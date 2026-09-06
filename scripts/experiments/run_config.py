#!/usr/bin/env python3
"""Run one explicit model configuration and save a stable run.json pointer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paths import configuration_paths

MODELS = {
    "cnn": ("training.cnn.train_kepler_cnn", "TrainConfig", "train_model"),
    "tabular": ("training.tabular.train_hgb", "TrainConfig", "train_model"),
    "blend": ("training.stacking.stack_cnn_hgb", "StackConfig", "run_stack"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cnn_record", type=Path)
    parser.add_argument("--hgb_record", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    module_name, config_class, function = MODELS[args.model]
    module = importlib.import_module(module_name)
    config = configuration_paths(json.loads(args.config.read_text()))
    directory = args.output_dir.resolve()
    config["output_root"] = str(directory)
    if args.model == "cnn":
        config["device"] = args.device
    if args.model == "blend":
        if not args.cnn_record or not args.hgb_record:
            parser.error("Blending requires explicit --cnn_record and --hgb_record")
        for key, record_path in (("cnn_run_dir", args.cnn_record), ("hgb_run_dir", args.hgb_record)):
            record = json.loads(record_path.read_text())
            if record["status"] != "complete":
                raise ValueError(f"Incomplete input run: {record_path}")
            config[key] = record["run_dir"]
    cfg = getattr(module, config_class)(**config)
    if args.dry_run:
        print(json.dumps(asdict(cfg), indent=2))
        return
    directory.mkdir(parents=True, exist_ok=False)
    record = {"model": args.model, "status": "running", "config": asdict(cfg)}
    record_path = directory / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        run_dir = getattr(module, function)(cfg)
        if args.model == "cnn":
            subprocess.run([sys.executable, str(ROOT / "scripts/analysis/plot_cnn_history.py"),
                            "--run_dir", str(run_dir), "--output_dir", str(directory / "figures")],
                           cwd=ROOT, check=True)
        record.update(status="complete", run_dir=str(run_dir.resolve()))
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Run record: {record_path}")


if __name__ == "__main__":
    main()
