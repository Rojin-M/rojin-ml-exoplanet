#!/usr/bin/env python3
"""Execute the documented Kepler experiments with explicit inputs and stable output records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("tabular", "cnn-baseline", "cnn-ablations", "blend", "cnn-v0", "cnn-v1",
          "cnn-comparison", "transformer", "comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/reproduce")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_uuids", nargs="+", help="Optional GPU UUIDs for parallel transformer runs")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline"), default="disabled")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    manifest = output / "workflow.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {"status": "running", "stages": {}}
    sources = [p for directory in ("configs", "scripts", "training") for p in (ROOT / directory).rglob("*")
               if p.suffix in (".py", ".json")]
    fingerprint = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    previous = state.get("source_sha256", fingerprint)
    changed = {p for p in previous.keys() | fingerprint.keys() if previous.get(p) != fingerprint.get(p)}
    analysis_only = lambda p: p.startswith("scripts/analysis/") or p == "scripts/experiments/reproduce.py"
    if any(not analysis_only(p) for p in changed):
        raise ValueError("Training, data, or configuration source changed; choose a new output directory")
    if changed:
        state.setdefault("analysis_source_revisions", []).append({
            "before_stage": args.stage,
            "changes": {p: {"before": previous.get(p), "after": fingerprint.get(p)} for p in sorted(changed)},
        })
    state["source_sha256"] = fingerprint
    execution = {"device": args.device, "gpu_uuids": args.gpu_uuids, "wandb_mode": args.wandb_mode}
    if state.get("execution", execution) != execution:
        raise ValueError("Use the same execution settings for every stage in this workflow")
    state["execution"] = execution

    def command(script, *options):
        argv = [sys.executable, str(ROOT / "scripts" / script), *map(str, options)]
        print(" ".join(argv), flush=True)
        if not args.dry_run:
            subprocess.run(argv, cwd=ROOT, check=True)

    def single(model, config, name, *options):
        command("experiments/run_config.py", "--model", model, "--config", ROOT / "configs" / (config + ".json"),
                "--output_dir", output / name, "--device", args.device, *options)

    for stage in STAGES if args.stage == "all" else [args.stage]:
        if state["stages"].get(stage, {}).get("status") == "complete":
            print(f"Already completed: {stage}", flush=True)
            continue
        if stage in state["stages"] and not args.dry_run:
            raise ValueError(f"Stage {stage} was interrupted or failed; preserve it and choose a new output directory")
        if not args.dry_run:
            state["stages"][stage] = {"status": "running"}
            manifest.write_text(json.dumps(state, indent=2) + "\n")
        try:
            if stage == "tabular":
                for mode in ("scalar", "full"):
                    single("tabular", "hgb_" + mode, "hgb_" + mode)
            elif stage == "cnn-baseline":
                single("cnn", "cnn_v0", "cnn_baseline")
            elif stage == "cnn-ablations":
                for variant in ("wide", "regularized"):
                    single("cnn", "cnn_" + variant, "cnn_" + variant)
            elif stage == "blend":
                single("blend", "blend", "blend", "--cnn_record", output / "cnn_baseline/run.json",
                       "--hgb_record", output / "hgb_full/run.json")
            elif stage in ("cnn-v0", "cnn-v1"):
                name = stage.replace("-", "_")
                command("experiments/run_cnn_multiseed.py", "--source_config", ROOT / "configs" / (name + ".json"),
                        "--group_dir", output / name, "--device", args.device, "--wandb_mode", args.wandb_mode)
                command("analysis/aggregate_cnn_multiseed.py", "--experiment", output / name / "experiment.json")
            elif stage == "cnn-comparison":
                command("analysis/compare_cnn_variants.py", "--baseline", output / "cnn_v0/experiment.json",
                        "--candidate", output / "cnn_v1/experiment.json", "--output_dir", output / "cnn_comparison/results",
                        "--figure_dir", output / "cnn_comparison/figures")
            elif stage == "transformer":
                options = ["--gpu_uuids", *args.gpu_uuids] if args.gpu_uuids else []
                existing = output / "transformer/experiment.json"
                if existing.exists() and not args.dry_run:
                    study = json.loads(existing.read_text())
                    if study["status"] != "complete":
                        raise ValueError("The standalone transformer study must finish before it can join this workflow")
                    for path, expected in study["source_sha256"].items():
                        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                            raise ValueError(f"Completed study source differs: {path}")
                    config_path = str(ROOT / "configs/transformer_base.json")
                    if study["source_sha256"].get(config_path) != hashlib.sha256(Path(config_path).read_bytes()).hexdigest():
                        raise ValueError("The study did not use the workflow's transformer configuration")
                    expected_device = "cuda:0" if args.gpu_uuids else args.device
                    if any(r["config"]["wandb_mode"] != args.wandb_mode or r["config"]["device"] != expected_device for r in study["runs"]):
                        raise ValueError("Completed study execution settings differ")
                    print("Verified completed standalone transformer study", flush=True)
                else:
                    command("experiments/run_transformer_study.py", "--source_config", ROOT / "configs/transformer_base.json",
                            "--study_dir", output / "transformer", "--device", args.device,
                            "--wandb_mode", args.wandb_mode, *options)
                command("analysis/analyze_transformer_study.py", "--study_dir", output / "transformer",
                        "--figure_dir", output / "transformer/figures", "--cnn_v0", output / "cnn_v0/experiment.json",
                        "--cnn_v1", output / "cnn_v1/experiment.json")
            elif stage == "comparison":
                command("analysis/compare_all_models.py", "--workflow_dir", output, "--output_dir", output / "comparison")
        except BaseException as exc:
            if not args.dry_run:
                state["stages"][stage].update(status="failed", error=repr(exc))
                state["status"] = "failed"
                manifest.write_text(json.dumps(state, indent=2) + "\n")
            raise
        if not args.dry_run:
            state["stages"][stage] = {"status": "complete"}
            state["status"] = "complete" if all(state["stages"].get(s, {}).get("status") == "complete" for s in STAGES) else "running"
            manifest.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
