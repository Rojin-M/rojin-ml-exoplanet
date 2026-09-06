#!/usr/bin/env python3
"""Repeat one saved CNN configuration on fixed data and splits, changing only its seed."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paths import configuration_paths
from training.cnn.train_kepler_cnn import TrainConfig  # noqa: E402


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_configs(source: dict, seeds: list, group_dir: Path, device: str, wandb_mode: str) -> list:
    base = configuration_paths(asdict(TrainConfig(**source)))
    for key in ("base_dir", "dataset_path", "splits_path", "manifest_path"):
        base[key] = str(Path(base[key]).expanduser().resolve())
    return [dict(base, seed=seed, output_root=str(group_dir / f"seed_{seed}"),
                 device=device, wandb_mode=wandb_mode, wandb_group=group_dir.name)
            for seed in seeds]


def check_sources(hashes: dict) -> None:
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"Source changed during the experiment: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_config", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline"), default="offline")
    parser.add_argument("--group_dir", type=Path, help="New experiment directory; must not already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print the plan without writing or training.")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be unique integers between 0 and 2**32 - 1")

    source_path = args.source_config.resolve()
    source = json.loads(source_path.read_text())
    group_dir = (args.group_dir or ROOT / "outputs" / "cnn" /
                 datetime.now(timezone.utc).strftime("multiseed_main_%Y%m%d_%H%M%S_utc")).resolve()
    configs = build_configs(source, args.seeds, group_dir, args.device, args.wandb_mode)
    paths = [source_path, Path(__file__).resolve(), ROOT / "training/cnn/train_kepler_cnn.py",
             ROOT / "scripts/analysis/plot_cnn_history.py"]
    paths.extend(Path(configs[0][key]) for key in ("dataset_path", "splits_path", "manifest_path"))
    hashes = {str(path): sha256(path) for path in paths}
    figures = Path(configs[0]["base_dir"]) / "outputs" / "figures" / "cnn" / group_dir.name
    plan = {
        "created_at": timestamp(), "status": "planned", "source_config": str(source_path),
        "source_sha256": hashes, "seeds": args.seeds, "group_dir": str(group_dir),
        "figure_dir": str(figures), "python": sys.version, "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "Fixed dataset and split; fixed hyperparameters; validation AP selects the checkpoint; "
                    "validation F1 selects its threshold; test evaluated once per completed run. "
                    "Training seeds vary; CUDA operations are not forced to be deterministic.",
        "runs": [{"seed": cfg["seed"], "status": "pending", "config": cfg} for cfg in configs],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    group_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = group_dir / "experiment.json"
    source_dir = group_dir / "source"
    source_dir.mkdir()
    for path in paths[:4]:
        shutil.copy2(path, source_dir / path.name)
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (group_dir / "requirements-frozen.txt").write_text(packages.stdout)
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                              "--format=csv"], capture_output=True, text=True, check=True)
        (group_dir / "gpu_environment.csv").write_text(gpu.stdout)
    plan["status"] = "running"
    write_json(manifest_path, plan)
    print(f"Experiment: {group_dir}", flush=True)

    # Each seed gets a fresh Python process, including independent data-loader workers and tracker.
    child_code = (
        "import json, sys; from training.cnn.train_kepler_cnn import TrainConfig, train_model; "
        "cfg = json.loads(open(sys.argv[1]).read()); train_model(TrainConfig(**cfg))"
    )
    active = None
    try:
        for record in plan["runs"]:
            active = record
            check_sources(hashes)
            seed_dir = Path(record["config"]["output_root"])
            seed_dir.mkdir()
            config_path = seed_dir / "requested_config.json"
            write_json(config_path, record["config"])
            command = [sys.executable, "-u", "-c", child_code, str(config_path)]
            record.update(status="running", started_at=timestamp(), command=command,
                          log_path=str(seed_dir / "training.log"))
            write_json(manifest_path, plan)
            print(f"Starting seed {record['seed']}; log: {record['log_path']}", flush=True)
            with Path(record["log_path"]).open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            summaries = list(seed_dir.glob("*/summary.json"))
            if len(summaries) != 1:
                raise RuntimeError(f"Expected exactly one completed run in {seed_dir}")
            run_dir = summaries[0].parent
            if json.loads((run_dir / "config.json").read_text()) != record["config"]:
                raise RuntimeError(f"Saved configuration differs from the plan: {run_dir}")
            for name in ("best_model.pt", "history.csv", "val_predictions.csv", "test_predictions.csv"):
                if not (run_dir / name).is_file() or not (run_dir / name).stat().st_size:
                    raise RuntimeError(f"Missing run artifact: {run_dir / name}")
            check_sources(hashes)
            record.update(status="complete", completed_at=timestamp(), run_dir=str(run_dir))
            write_json(manifest_path, plan)
            print(f"Completed seed {record['seed']}: {run_dir}", flush=True)

        active = None
        plan["status"] = "plotting"
        write_json(manifest_path, plan)
        plot_command = [sys.executable, str(ROOT / "scripts/analysis/plot_cnn_history.py"), "--run_dir",
                        *[r["run_dir"] for r in plan["runs"]], "--output_dir", str(figures)]
        with (group_dir / "plotting.log").open("w") as log:
            subprocess.run(plot_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        check_sources(hashes)
        plan.update(status="complete", completed_at=timestamp(), sources_unchanged=True)
        write_json(manifest_path, plan)
        print(f"All seeds and plots complete. Manifest: {manifest_path}", flush=True)
    except BaseException as exc:
        if active is not None and active["status"] == "running":
            active.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        plan.update(status="failed", error=f"{type(exc).__name__}: {exc}", stopped_at=timestamp())
        write_json(manifest_path, plan)
        raise


if __name__ == "__main__":
    main()
