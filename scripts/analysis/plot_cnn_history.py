#!/usr/bin/env python3
"""Export CNN learning curves from saved histories, without loading data or models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


METRICS = (
    ("loss", "Weighted binary cross-entropy", "loss"),
    ("pr_auc", "Average precision (AP)", "average_precision"),
    ("f1", "F1 score", "f1"),
)
COLORS = {"train": "#0072B2", "val": "#D55E00"}
CURVE_NOTE = (
    "Training: evolving weights and dropout. Validation: evaluation mode.\n"
    "F1: thresholds tuned separately per split and epoch. Curves are unsmoothed."
)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], int]:
    history_path = run_dir / "history.csv"
    history = pd.read_csv(history_path)
    required = ["epoch"] + [f"{split}_{metric}" for metric, _, _ in METRICS for split in ("train", "val")]
    missing = sorted(set(required) - set(history.columns))
    if missing:
        raise ValueError(f"{history_path}: missing columns: {', '.join(missing)}")
    if history.empty:
        raise ValueError(f"{history_path}: empty history")
    values = history[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{history_path}: required metrics must be finite numbers")
    epochs = history["epoch"].to_numpy(dtype=float)
    if (epochs < 1).any() or (epochs != np.floor(epochs)).any() or (np.diff(epochs) <= 0).any():
        raise ValueError(f"{history_path}: epochs must be positive, increasing, unique integers")
    for metric, _, _ in METRICS:
        for split in ("train", "val"):
            series = history[f"{split}_{metric}"]
            if (series < 0).any() or (metric != "loss" and (series > 1).any()):
                raise ValueError(f"{history_path}: invalid {split}_{metric} values")

    def read_optional(name: str) -> dict[str, Any]:
        path = run_dir / name
        return json.loads(path.read_text()) if path.exists() else {}

    summary = read_optional("summary.json")
    config = read_optional("config.json")
    peak_epoch = int(history.loc[history["val_pr_auc"].idxmax(), "epoch"])
    selected = summary.get("best_epoch", peak_epoch)
    if selected not in epochs:
        raise ValueError(f"{run_dir}: selected epoch {selected} does not occur in history")
    selected = int(selected)
    selected_ap = float(history.loc[history["epoch"] == selected, "val_pr_auc"].iloc[0])
    if not np.isclose(selected_ap, history["val_pr_auc"].max(), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{run_dir}: selected epoch disagrees with maximum validation AP")
    return history, summary, config, selected


def draw_metric(ax: Any, history: pd.DataFrame, metric: str, label: str, selected: int) -> None:
    for split, name, style in (("train", "Training", "-"), ("val", "Validation", "--")):
        ax.plot(history["epoch"], history[f"{split}_{metric}"], style,
                color=COLORS[split], linewidth=1.8, label=name)
    ax.axvline(selected, color="#555555", linestyle=":", linewidth=1.3,
               label=f"Selected epoch {selected} (validation AP)")
    selected_value = float(history.loc[history["epoch"] == selected, f"val_{metric}"].iloc[0])
    ax.scatter([selected], [selected_value], s=38, color=COLORS["val"],
               edgecolor="white", linewidth=0.8, zorder=5)
    ax.set(xlabel="Epoch", ylabel=label)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    last_epoch = int(history["epoch"].iloc[-1])
    ax.set_xlim(0.5, last_epoch + 0.5)
    ax.set_ylim(0, max(1.0, 1.08 * history[["train_loss", "val_loss"]].to_numpy().max()) if metric == "loss" else 1.0)
    ax.grid(axis="y", color="#DDE1E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def export_run(run_dir: Path, output_dir: Path, dpi: int) -> dict[str, Any]:
    history, summary, config, selected = load_run(run_dir)
    destination = output_dir / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    match = re.search(r"cnn_\d{8}_\d{6}$", run_dir.name)
    short_name = match.group(0) if match else run_dir.name
    subtitle = f"{short_name}  |  training seed {config.get('seed', 'not recorded')}"
    if "train_size" in summary and "val_size" in summary:
        subtitle += f"  |  train {summary['train_size']:,}, validation {summary['val_size']:,}"
    files = []

    def save(fig: Any, stem: str) -> None:
        for extension in ("png", "pdf"):
            path = destination / f"{stem}.{extension}"
            fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
        for ax, (metric, label, _) in zip(axes, METRICS):
            draw_metric(ax, history, metric, label, selected)
        fig.suptitle("Kepler CNN learning curves", fontsize=15, fontweight="bold", y=0.98)
        fig.text(0.5, 0.91, subtitle, ha="center", fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=3, frameon=False)
        fig.text(0.5, 0.02, CURVE_NOTE, ha="center", va="bottom", fontsize=9, color="#444444")
        fig.subplots_adjust(left=0.065, right=0.985, top=0.73, bottom=0.23, wspace=0.29)
        save(fig, "learning_curves")

        for metric, label, stem in METRICS:
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            draw_metric(ax, history, metric, label, selected)
            fig.suptitle(f"Kepler CNN: {label}", fontsize=13, fontweight="bold", y=0.98)
            fig.text(0.5, 0.915, subtitle, ha="center", fontsize=8)
            ax.legend(loc="best", framealpha=0.95, fontsize=8)
            fig.text(0.5, 0.02, CURVE_NOTE, ha="center", va="bottom", fontsize=8, color="#444444")
            fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.22)
            save(fig, stem)

    selected_row = history.loc[history["epoch"] == selected].iloc[0]
    sources = {}
    for name in ("history.csv", "summary.json", "config.json"):
        path = run_dir / name
        if path.exists():
            sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {
        "run_name": run_dir.name,
        "training_seed": config.get("seed"),
        "source_files": sources,
        "selection_source": "summary.json" if "best_epoch" in summary else "maximum val_pr_auc in history.csv",
        "selected_epoch": selected,
        "last_epoch": int(history["epoch"].iloc[-1]),
        "selected_history_metrics": {key: float(selected_row[key]) for key in history.columns if key != "epoch"},
        "metric_definition": "pr_auc is sklearn.metrics.average_precision_score, not trapezoidal PR area",
        "interpretation": CURVE_NOTE.replace("\n", " "),
        "smoothing": "none",
        "dpi": dpi,
        "files": files,
    }
    (destination / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    caption = (
        f"# Learning-curve captions: {short_name}\n\n"
        f"Source run: `{run_dir.name}`. Training seed: {config.get('seed', 'not recorded')}.\n\n"
        f"The dotted line marks epoch {selected}, selected using maximum validation average precision; "
        f"the orange dot marks its validation value in each panel. The history ends at epoch {int(history['epoch'].iloc[-1])}. "
        "The marker does not select the minimum loss or maximum F1. No test metrics are used to draw these curves.\n\n"
        "Training metrics are accumulated while weights change during optimization, with dropout and any configured training regularization active. "
        "Validation metrics use evaluation mode after each epoch. Their difference is not a comparison of the same fixed model under identical conditions.\n\n"
        "Average precision uses scikit-learn's average_precision_score (stored as pr_auc). "
        "For F1, separate thresholds are tuned on training and validation predictions each epoch; "
        "validation F1 is therefore a tuned development metric. These curves do not use the fixed final classification threshold.\n\n"
        "These are unsmoothed single-run curves. No seed uncertainty or error bars are implied. "
        "Class weighting and any configured label smoothing affect the recorded loss.\n\n"
        "Exports: `learning_curves` (three panels), `loss`, `average_precision`, and `f1`, each as PNG and PDF. "
        "Use the PDF for vector graphics in the report. Exact sources and hashes are in `plot_metadata.json`.\n"
    )
    (destination / "CAPTIONS.md").write_text(caption)
    print(f"{short_name}: selected epoch {selected}, exported {len(files)} figures to {destination}")
    return {
        "run_name": run_dir.name, "training_seed": config.get("seed"),
        "selected_epoch": selected, "last_epoch": int(history["epoch"].iloc[-1]),
        **{f"selected_{split}_{metric}": float(selected_row[f"{split}_{metric}"])
           for metric, _, _ in METRICS for split in ("train", "val")},
        "figure_dir": str(destination.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, default=Path(__file__).resolve().parents[2])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run_dir", type=Path, nargs="+", help="One or more saved CNN run directories.")
    selection.add_argument("--all_cnn", action="store_true", help="Plot all histories directly under outputs/cnn/.")
    parser.add_argument("--output_dir", type=Path, help="Default: <base_dir>/outputs/figures/cnn.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    runs = args.run_dir or [p.parent for p in sorted((args.base_dir / "outputs" / "cnn").glob("*/history.csv"))]
    if not runs:
        parser.error("No CNN histories found")
    if len({p.name for p in runs}) != len(runs):
        parser.error("Run directory names must be distinct")
    output_dir = args.output_dir or args.base_dir / "outputs" / "figures" / "cnn"
    rows = [export_run(run, output_dir, args.dpi) for run in runs]
    pd.DataFrame(rows).to_csv(output_dir / "plotted_runs.csv", index=False)
    print(f"Run index: {output_dir / 'plotted_runs.csv'}")


if __name__ == "__main__":
    main()
