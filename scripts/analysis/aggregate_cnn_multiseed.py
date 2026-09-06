#!/usr/bin/env python3
"""Aggregate every planned seed in a completed CNN experiment and export report figures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

METRICS = (
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("average_precision", "Average precision"),
    ("roc_auc", "ROC-AUC"),
)
SPLITS = ("train", "val", "test")
DISPLAY_SPLITS = {"train": "Training", "val": "Validation", "test": "Test"}
VARYING_CONFIG_KEYS = {"seed", "output_root", "wandb_mode", "wandb_project", "wandb_entity", "wandb_group"}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#A07800")
MARKERS = ("o", "s", "^", "v", "P")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiment(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reject incomplete, duplicated, or incompatible runs rather than silently filtering them."""
    sources: dict[str, Any] = {}

    def read_json(source: Path) -> dict[str, Any]:
        sources[str(source.resolve())] = file_hash(source)
        return json.loads(source.read_text())

    experiment = read_json(path)
    if experiment.get("status") != "complete":
        raise ValueError("Experiment must be complete before aggregation")
    seeds = experiment.get("seeds", [])
    if (len(seeds) < 2 or any(type(s) is not int for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("At least two unique planned integer seeds are required for sample SD")
    records = experiment.get("runs", [])
    actual_seeds = [record.get("seed") for record in records]
    if len(actual_seeds) != len(seeds) or set(actual_seeds) != set(seeds):
        raise ValueError("Run records must contain every planned seed exactly once")
    rows = []
    run_paths = set()
    reference_config = None
    reference_sizes = None
    for record in sorted(records, key=lambda r: r["seed"]):
        if record.get("status") != "complete":
            raise ValueError(f"Seed {record['seed']} is not complete")
        run_dir = Path(record["run_dir"]).resolve()
        if run_dir in run_paths:
            raise ValueError("A run directory is referenced more than once")
        run_paths.add(run_dir)
        config = read_json(run_dir / "config.json")
        summary = read_json(run_dir / "summary.json")
        if config != record["config"] or config["seed"] != record["seed"]:
            raise ValueError(f"Seed {record['seed']}: saved configuration differs from the experiment plan")
        fixed = {k: v for k, v in config.items() if k not in VARYING_CONFIG_KEYS}
        if reference_config is None:
            reference_config = fixed
        elif fixed != reference_config:
            raise ValueError("Seeds have different data paths, model settings, or training settings")
        for key in ("dataset_path", "splits_path", "manifest_path"):
            if summary[key] != config[key]:
                raise ValueError(f"Seed {record['seed']}: summary has an inconsistent {key}")
        sizes = {split: summary[f"{split}_size"] for split in SPLITS}
        if any(type(n) is not int or n <= 0 for n in sizes.values()):
            raise ValueError("Split sizes must be positive integers")
        if reference_sizes is None:
            reference_sizes = sizes
        elif sizes != reference_sizes:
            raise ValueError("Seeds have different split sizes")
        history_path = run_dir / "history.csv"
        sources[str(history_path)] = file_hash(history_path)
        history = pd.read_csv(history_path)
        if history.empty or not np.array_equal(history.epoch, np.arange(1, len(history) + 1)):
            raise ValueError(f"{run_dir}: history must contain consecutive epochs starting at 1")
        if not np.isfinite(history.val_pr_auc.to_numpy(dtype=float)).all():
            raise ValueError(f"{run_dir}: validation AP must be finite")
        selected = history.loc[history.val_pr_auc.idxmax()]
        if summary["best_epoch"] != selected.epoch:
            raise ValueError(f"{run_dir}: checkpoint does not match maximum validation AP")
        threshold = summary["best_threshold"]
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{run_dir}: invalid classification threshold")
        if not np.isclose(threshold, selected.val_threshold, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{run_dir}: threshold differs from selected validation epoch")
        row = {"seed": record["seed"], "run_name": run_dir.name, "run_dir": str(run_dir),
               "epochs_run": len(history), "selected_epoch": summary["best_epoch"],
               "validation_threshold": threshold}
        row.update({f"{split}_size": size for split, size in sizes.items()})
        for split in SPLITS:
            for metric, _ in METRICS:
                saved_name = "pr_auc" if metric == "average_precision" else metric
                value = float(summary[f"{split}_metrics"][saved_name])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{run_dir}: invalid {split} {metric}")
                row[f"{split}_{metric}"] = value
        if not np.isclose(row["val_average_precision"], selected.val_pr_auc, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{run_dir}: final validation AP differs from selected epoch")
        rows.append(row)
    return pd.DataFrame(rows), {
        "experiment_path": str(path.resolve()), "group_name": path.parent.name,
        "source_files_sha256": sources, "shared_configuration": reference_config,
        "split_sizes": reference_sizes, "seeds": sorted(seeds),
        "experiment_figure_dir": experiment.get("figure_dir"),
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for metric, label in METRICS:
            values = per_seed[f"{split}_{metric}"].to_numpy(dtype=float)
            rows.append({"split": split, "metric": metric, "label": label, "n": len(values),
                         "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                         "min": float(values.min()), "max": float(values.max())})
    return pd.DataFrame(rows)


def plot_summary(per_seed: pd.DataFrame, summary: pd.DataFrame, destination: Path,
                 metadata: dict[str, Any], dpi: int) -> dict[str, Any]:
    displayed = summary[summary.split.isin(("val", "test"))]
    lower = min(displayed["min"].min(), (displayed["mean"] - displayed["std"]).min())
    upper = max(displayed["max"].max(), (displayed["mean"] + displayed["std"]).max())
    y_min = math.floor((lower - 0.02) / 0.05) * 0.05
    y_max = math.ceil((upper + 0.02) / 0.05) * 0.05
    offsets = np.linspace(-0.25, 0.15, len(per_seed))
    mean_offset = 0.30
    handles = [Line2D([], [], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                      linestyle="none", markersize=6, label=f"Seed {int(row.seed)}")
               for i, row in enumerate(per_seed.itertuples())]
    handles.append(Line2D([], [], color="#222222", marker="D", markersize=6,
                          label="Mean ± sample SD"))
    files = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for splits, stem in ((("test",), "test_metrics"), (("val", "test"), "validation_test_metrics")):
            fig, axes = plt.subplots(1, len(splits), figsize=(8.8 if len(splits) == 1 else 12.4, 5.8),
                                     squeeze=False, sharey=True)
            for ax, split in zip(axes[0], splits):
                x = np.arange(len(METRICS))
                stats = summary[summary.split == split].set_index("metric").loc[[m for m, _ in METRICS]]
                for i, (_, row) in enumerate(per_seed.iterrows()):
                    ax.scatter(x + offsets[i], [row[f"{split}_{m}"] for m, _ in METRICS],
                               color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                               s=45, linewidths=0.5, edgecolors="white", zorder=3)
                ax.errorbar(x + mean_offset, stats["mean"], yerr=stats["std"], fmt="D", color="#222222",
                            markersize=5, capsize=6, elinewidth=1.6, capthick=1.6, zorder=4)
                ax.set_xticks(x, ["Accuracy", "F1", "Average\nprecision", "ROC-AUC"])
                ax.set(xlim=(-0.5, 3.5), ylim=(y_min, y_max),
                       title=f"{DISPLAY_SPLITS[split]} · {metadata['split_sizes'][split]:,} objects")
                ax.grid(axis="y", color="#DDE1E5", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0, 0].set_ylabel("Score")
            fig.suptitle("Kepler CNN: variation across training seeds", fontsize=15, fontweight="bold", y=0.98)
            fig.text(0.5, 0.92, f"Fixed model and star-grouped split · {len(per_seed)} fresh runs", ha="center", fontsize=10)
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False)
            fig.text(0.5, 0.045,
                     "Error bars: ±1 sample SD across runs (ddof=1). They are not confidence intervals.\n"
                     f"Score axis shown from {y_min:.2f} to {y_max:.2f}. Checkpoints and thresholds selected on validation.",
                     ha="center", fontsize=9, color="#444444")
            fig.subplots_adjust(left=0.085, right=0.98, top=0.70, bottom=0.23, wspace=0.17)
            for extension in ("png", "pdf"):
                filename = f"{stem}.{extension}"
                fig.savefig(destination / filename, dpi=dpi, facecolor="white", bbox_inches="tight")
                files.append(filename)
            plt.close(fig)
    return {"files": files, "y_limits": [y_min, y_max], "seed_offsets": offsets.tolist(), "mean_offset": mean_offset,
            "dpi": dpi, "error_bar": "sample standard deviation, ddof=1", "point_source": "per_seed_metrics.csv"}


def export(experiment_path: Path, output_dir: Path | None, figure_dir: Path | None, dpi: int) -> Path:
    per_seed, metadata = load_experiment(experiment_path)
    summary = summarize(per_seed)
    output_dir = (output_dir or experiment_path.parent / "aggregate").resolve()
    figure_dir = (figure_dir or Path(metadata["experiment_figure_dir"]) / "aggregate").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    plot_details = plot_summary(per_seed, summary, figure_dir, metadata, dpi)
    metadata.update({"generated_at": datetime.now(timezone.utc).isoformat(),
                     "script_sha256": file_hash(Path(__file__)), "n_runs": len(per_seed),
                     "aggregation": "Equal-weight arithmetic mean and sample standard deviation (ddof=1) of run metrics",
                     "metric_definition": "average_precision is sklearn average_precision_score, saved as pr_auc in runs",
                     "evaluation": "Final checkpoint in evaluation mode for all splits; validation AP selects checkpoint; "
                                   "validation F1 selects threshold. All planned seeds included; no probability averaging or seed selection.",
                     "uncertainty_scope": "Training-run variability on one previously consulted fixed split; "
                                          "does not measure dataset/split uncertainty. CUDA determinism was not forced.",
                     "output_dir": str(output_dir), "figure_dir": str(figure_dir),
                     "per_seed_records": per_seed.to_dict(orient="records"),
                     "summary_records": summary.to_dict(orient="records"), "plots": plot_details})
    for name in ("per_seed_metrics.csv", "summary_metrics.csv"):
        metadata.setdefault("export_sha256", {})[str(output_dir / name)] = file_hash(output_dir / name)
    for name in plot_details["files"]:
        metadata["export_sha256"][str(figure_dir / name)] = file_hash(figure_dir / name)
    (output_dir / "summary_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        f"# Multi-seed CNN figure captions\n\nExperiment: `{metadata['group_name']}`. "
        f"Training seeds: {', '.join(map(str, metadata['seeds']))}.\n\n"
        "Colored symbols show each training run; black diamonds and error bars show the equal-weight mean "
        "and ±1 sample standard deviation (ddof=1). Horizontal offsets separate seeds and carry no numerical meaning. "
        "The same seed colors and shapes are used in both panels. Score axes are zoomed to show the spread, "
        f"with limits {plot_details['y_limits'][0]:.2f}–{plot_details['y_limits'][1]:.2f}.\n\n"
        "All runs use the same architecture, hyperparameters, dataset, and star-grouped split. "
        "Each run uses its maximum-validation-AP checkpoint and a threshold selected by validation F1. "
        "Validation metrics are development results affected by that selection. Test thresholds are not retuned. "
        "Average precision is sklearn average_precision_score, not trapezoidal PR area.\n\n"
        "These error bars describe variability across training runs on one fixed, previously consulted split. "
        "They are not standard errors, confidence intervals, or estimates of uncertainty across new datasets. "
        "The figure is not an ensemble result and includes every planned seed.\n\n"
        f"Exact values, configuration, source hashes, and plot settings: `{output_dir / 'summary_metrics.json'}`.\n"
    )
    (figure_dir / "CAPTIONS.md").write_text(caption)
    report = ["# CNN multi-seed results", "", f"Seeds: {', '.join(map(str, metadata['seeds']))}; "
              f"test objects: {metadata['split_sizes']['test']:,}. Mean ± sample SD (ddof=1); all scores use a 0–1 scale.", "",
              "| Test metric | Mean ± SD | Minimum | Maximum |", "| --- | --- | --- | --- |"]
    for row in summary[summary.split == "test"].itertuples():
        report.append(f"| {row.label} | {row.mean:.4f} ± {row.std:.4f} | {row.min:.4f} | {row.max:.4f} |")
    report.extend(["", "| Seed | Selected epoch | Validation threshold | Test accuracy | Test F1 | Test AP | Test ROC-AUC |",
                   "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in per_seed.itertuples():
        report.append(f"| {row.seed} | {row.selected_epoch} | {row.validation_threshold:.3f} | "
                      f"{row.test_accuracy:.4f} | {row.test_f1:.4f} | {row.test_average_precision:.4f} | {row.test_roc_auc:.4f} |")
    report.extend(["", "Each run uses the same fixed split and hyperparameters, with validation-based checkpoint and threshold selection. "
                   "Standard deviations describe training-run variability, not confidence intervals or new-dataset uncertainty.", "",
                   f"Figures and captions: {figure_dir}", "",
                   "Detailed CSV tables include final training, validation, and test metrics; JSON records exact sources and hashes.", ""])
    (output_dir / "REPORT_SUMMARY.md").write_text("\n".join(report))
    for path, expected in metadata["source_files_sha256"].items():
        if file_hash(Path(path)) != expected:
            raise RuntimeError(f"Source changed while aggregating: {path}")
    print(f"Aggregated all {len(per_seed)} planned seeds. Tables: {output_dir}")
    print(f"Figures: {figure_dir}")
    print(summary[summary.split == "test"][["label", "n", "mean", "std"]].to_string(index=False))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path, help="Completed experiment.json from run_cnn_multiseed.py")
    parser.add_argument("--output_dir", type=Path, help="Default: <experiment directory>/aggregate")
    parser.add_argument("--figure_dir", type=Path, help="Default: <experiment figure directory>/aggregate")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    export(args.experiment.resolve(), args.output_dir, args.figure_dir, args.dpi)


if __name__ == "__main__":
    main()
