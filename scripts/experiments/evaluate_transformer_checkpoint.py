#!/usr/bin/env python3
"""Evaluate a validation-selected transformer checkpoint after experiment settings are frozen."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from training.transformer import train_kepler_transformer as trainer


def evaluate(run_dir: Path, device_name: str = "auto") -> dict:
    run_dir = run_dir.resolve()
    cfg = trainer.TrainConfig(**json.loads((run_dir / "config.json").read_text()))
    if cfg.evaluate_test or (run_dir / "test_predictions.csv").exists():
        raise ValueError("This command requires a completed run with test evaluation deferred")
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary.get("test_evaluated") is not False:
        raise ValueError("Training summary must explicitly record deferred test evaluation")
    destination = run_dir / "test_evaluation"
    destination.mkdir(exist_ok=False)
    device = trainer.detect_device(device_name)
    evaluation_cfg = replace(cfg, evaluate_test=True, device=str(device))
    with trainer.tracking_run(evaluation_cfg, destination) as tracker:
        trainer.set_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")
        arrays, manifest = trainer.load_training_data(cfg)
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
        if checkpoint["epoch"] != summary["best_epoch"] or checkpoint["config"] != trainer.asdict(cfg):
            raise ValueError("Checkpoint differs from saved training configuration/selection")
        model = trainer.ExoFormer(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        y = arrays["y"][arrays["train_idx"]]
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum())/y.sum()], dtype=torch.float32, device=device))
        def loader(split):
            data = trainer.KeplerTensorDataset(arrays["x_global"], arrays["x_local"], arrays["x_scalar"],
                arrays["x_flat"], arrays["y"], arrays[split+"_idx"], x_aux=arrays.get("x_aux"))
            return trainer.make_dataloader(data, cfg.batch_size, False, cfg.num_workers, device)
        # Reproduce validation predictions to check the saved model before touching test inputs.
        validation = trainer.evaluate_split(model, loader("val"), loss_fn, device, summary["best_threshold"])
        saved = pd.read_csv(run_dir / "val_predictions.csv")
        np.testing.assert_allclose(validation["probs"], saved.probability, rtol=1e-5, atol=1e-6)
        result = trainer.evaluate_split(model, loader("test"), loss_fn, device, summary["best_threshold"])
        trainer.save_predictions_csv(run_dir / "test_predictions.csv", manifest, result["indices"],
                                     result["probs"], summary["best_threshold"])
        record = {"run_dir": str(run_dir), "phase": "evaluation after settings frozen", "device": str(device),
                  "seed": cfg.seed, "best_epoch": summary["best_epoch"], "best_threshold": summary["best_threshold"],
                  "test_metrics": result["metrics"], "test_size": len(result["targets"]),
                  "validation_checkpoint_replay_matches": True,
                  "checkpoint_sha256": hashlib.sha256((run_dir/"best_model.pt").read_bytes()).hexdigest(),
                  "training_config_sha256": hashlib.sha256((run_dir/"config.json").read_bytes()).hexdigest()}
        trainer.write_json(run_dir / "test_evaluation.json", record)
        if tracker is not None:
            tracker.config.update({"phase": record["phase"], "parent_run_dir": str(run_dir),
                                   "checkpoint_sha256": record["checkpoint_sha256"]})
            metrics = {"test/"+k:v for k,v in result["metrics"].items()}
            tracker.log(metrics); tracker.summary.update(metrics)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args.run_dir, args.device)
