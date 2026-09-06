#!/usr/bin/env python3
"""Compare completed v0 and auxiliary-v1 CNN experiments on matched seeds and data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.paths import processed_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from aggregate_cnn_multiseed import COLORS, METRICS, file_hash, load_experiment

ALLOWED_DIFFERENCES = {"dataset_path", "splits_path", "manifest_path", "prepare_version", "use_aux_branch"}
SPLITS = ("val", "test")


def check_provenance(path: Path) -> dict:
    experiment = json.loads(path.read_text())
    verified = {}
    snapshot_names = {Path(experiment["source_config"]).name, "run_cnn_multiseed.py",
                      "train_kepler_cnn.py", "plot_cnn_history.py"}
    for source, expected in experiment["source_sha256"].items():
        original = processed_path(source)
        snapshot = path.parent / "source" / original.name
        if original.is_file() and file_hash(original) == expected:
            verified[source] = {"verified_path": str(original), "sha256": expected}
        elif original.name in snapshot_names and snapshot.is_file() and file_hash(snapshot) == expected:
            verified[source] = {"verified_path": str(snapshot), "sha256": expected}
        else:
            raise ValueError(f"Source hash mismatch with no matching saved snapshot: {original}")
    return verified


def check_inputs(v0: dict, v1: dict) -> tuple[pd.DataFrame, dict]:
    a = dict(v0)
    b = dict(v1)
    # Old v0 configurations predate both optional auxiliary fields.
    a.setdefault("use_aux_branch", False)
    a.setdefault("aux_dropout", 0.10)
    b.setdefault("aux_dropout", 0.10)
    if a.get("use_aux_branch") is not False or b.get("use_aux_branch") is not True:
        raise ValueError("Expected an auxiliary-disabled baseline and an auxiliary-enabled candidate")
    if {k: v for k, v in a.items() if k not in ALLOWED_DIFFERENCES} != {
            k: v for k, v in b.items() if k not in ALLOWED_DIFFERENCES}:
        raise ValueError("Shared hyperparameters or execution settings differ between variants")
    for key in ("manifest_path", "splits_path"):
        if processed_path(a[key]).read_bytes() != processed_path(b[key]).read_bytes():
            raise ValueError(f"Variants must use byte-identical {key} contents")
    manifest = pd.read_csv(processed_path(a["manifest_path"]))
    with np.load(processed_path(a["splits_path"])) as saved:
        splits = {s: saved[s + "_idx"].copy() for s in ("train", "val", "test")}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(manifest)))
    groups = [set(manifest.iloc[indices].kepid) for indices in splits.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Star groups overlap between splits")
    with np.load(processed_path(a["dataset_path"]), allow_pickle=True) as first, np.load(processed_path(b["dataset_path"]), allow_pickle=True) as second:
        for key in first.files:
            if key != "preprocessing_version":
                np.testing.assert_array_equal(first[key], second[key], err_msg=f"Original input changed: {key}")
        np.testing.assert_array_equal(first["y"], manifest.y.to_numpy())
        np.testing.assert_array_equal(second["aux_candidate_ids"].astype(str), manifest.candidate_id.astype(str))
        if second["X_aux"].shape[0] != len(manifest) or not np.isfinite(second["X_aux"]).all():
            raise ValueError("Invalid auxiliary inputs")
    return manifest, splits


def verify_predictions(rows: pd.DataFrame, manifest: pd.DataFrame, splits: dict,
                       config: dict, sources: dict) -> pd.DataFrame:
    rows = rows.copy()
    for i, row in rows.iterrows():
        summary = json.loads((Path(row.run_dir) / "summary.json").read_text())
        for split in SPLITS:
            path = Path(row.run_dir) / f"{split}_predictions.csv"
            sources[str(path)] = file_hash(path)
            prediction = pd.read_csv(path)
            expected = manifest.iloc[splits[split]].reset_index(drop=True)
            pd.testing.assert_frame_equal(prediction[manifest.columns], expected)
            if not prediction.candidate_id.is_unique or not prediction.probability.between(0, 1).all():
                raise ValueError(f"Invalid candidate IDs or probabilities: {path}")
            y = prediction.y.to_numpy()
            probabilities = prediction.probability.to_numpy()
            labels = (probabilities >= row.validation_threshold).astype(int)
            np.testing.assert_array_equal(prediction.prediction.to_numpy(), labels)
            metrics = {"accuracy": accuracy_score(y, labels), "precision": precision_score(y, labels, zero_division=0),
                       "recall": recall_score(y, labels, zero_division=0), "f1": f1_score(y, labels, zero_division=0),
                       "pr_auc": average_precision_score(y, probabilities), "roc_auc": roc_auc_score(y, probabilities)}
            for key, value in metrics.items():
                np.testing.assert_allclose(value, summary[split + "_metrics"][key], rtol=1e-10, atol=1e-12)
            rows.loc[i, split + "_false_positives"] = int(((labels == 1) & (y == 0)).sum())
            rows.loc[i, split + "_false_negatives"] = int(((labels == 0) & (y == 1)).sum())
            rows.loc[i, split + "_precision"] = metrics["precision"]
            rows.loc[i, split + "_recall"] = metrics["recall"]
            if split == "val":
                grid = np.linspace(0.05, 0.95, config["threshold_grid_size"])
                scores = [f1_score(y, probabilities >= threshold, zero_division=0) for threshold in grid]
                np.testing.assert_allclose(grid[int(np.argmax(scores))], row.validation_threshold, rtol=1e-10, atol=1e-12)
    return rows


def compare(baseline: Path, candidate: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    first, meta0 = load_experiment(baseline)
    second, meta1 = load_experiment(candidate)
    if meta0["seeds"] != meta1["seeds"]:
        raise ValueError("Variants must include the same complete set of planned seeds")
    manifest, splits = check_inputs(meta0["shared_configuration"], meta1["shared_configuration"])
    provenance = {"v0": check_provenance(baseline), "v1": check_provenance(candidate)}
    sources = dict(meta0["source_files_sha256"], **meta1["source_files_sha256"])
    sources.update({entry["verified_path"]: entry["sha256"] for variant in provenance.values() for entry in variant.values()})
    first = verify_predictions(first, manifest, splits, meta0["shared_configuration"], sources)
    second = verify_predictions(second, manifest, splits, meta1["shared_configuration"], sources)
    combined = pd.concat([first.assign(variant="v0"), second.assign(variant="v1")], ignore_index=True)
    paired = first.merge(second, on="seed", suffixes=("_v0", "_v1"), validate="one_to_one")
    statistics = []
    for split in SPLITS:
        for metric, label in METRICS:
            key = f"{split}_{metric}"
            x, y = paired[key + "_v0"], paired[key + "_v1"]
            delta = y - x
            paired[key + "_delta"] = delta
            statistics.append({"split": split, "metric": metric, "label": label, "n": len(delta),
                               "v0_mean": x.mean(), "v0_std": x.std(ddof=1), "v1_mean": y.mean(),
                               "v1_std": y.std(ddof=1), "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1),
                               "delta_min": delta.min(), "delta_max": delta.max(),
                               "improved_seeds": int((delta > 0).sum()), "tied_seeds": int((delta == 0).sum())})
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "seeds": meta0["seeds"],
                "baseline": str(baseline.resolve()), "candidate": str(candidate.resolve()),
                "split_sizes": meta0["split_sizes"], "shared_v0_configuration": meta0["shared_configuration"],
                "shared_v1_configuration": meta1["shared_configuration"], "verified_source_provenance": provenance,
                "source_files_sha256": sources, "prediction_metrics_recomputed": True,
                "original_arrays_equal": True, "split_and_manifest_bytes_equal": True, "star_groups_disjoint": True,
                "error_bars": "Sample SD across runs or paired-by-seed differences (ddof=1), not confidence intervals",
                "limitations": "Same previously consulted split. Training-seed variability does not measure new-dataset uncertainty. "
                               "Architecture changes alter random-number consumption. This compares the combined metadata/branch "
                               "extension; it does not isolate individual features or model capacity."}
    return combined, paired, pd.DataFrame(statistics), metadata


def plot(paired: pd.DataFrame, stats: pd.DataFrame, destination: Path, dpi: int) -> list[str]:
    files = []
    test_stats = stats[stats.split == "test"].set_index("metric")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=(12, 5.5), sharey=True)
        values = []
        for ax, (metric, label) in zip(axes, METRICS):
            summary = test_stats.loc[metric]
            for j, row in enumerate(paired.itertuples()):
                y = [getattr(row, "test_" + metric + suffix) for suffix in ("_v0", "_v1")]
                values.extend(y)
                ax.plot([0, 1], y, "o-", color=COLORS[j % len(COLORS)], alpha=0.8, markersize=4,
                        label=f"Seed {row.seed}")
            ax.errorbar([0.17, 1.17], [summary.v0_mean, summary.v1_mean],
                        yerr=[summary.v0_std, summary.v1_std], fmt="D", color="#222222", capsize=5,
                        label="Mean ± sample SD", zorder=5)
            values.extend([summary.v0_mean-summary.v0_std, summary.v1_mean-summary.v1_std,
                           summary.v0_mean+summary.v0_std, summary.v1_mean+summary.v1_std])
            ax.set(xticks=[0, 1], xticklabels=["v0", "v1 + aux"], xlim=(-0.3, 1.4), title=label)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set(ylim=(max(0, min(values)-0.025), min(1.01, max(values)+0.02)), ylabel="Test score")
        fig.suptitle("Kepler CNN: auxiliary metadata comparison", fontsize=15, fontweight="bold")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False)
        fig.text(0.5, 0.04, "Lines connect matching seed numbers on the same test split. Black diamonds: mean ± sample SD.\n"
                 "Checkpoints and thresholds selected on validation. Score axis is truncated; error bars are not confidence intervals.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(top=0.73, bottom=0.2, wspace=0.18)
        for extension in ("png", "pdf"):
            name = "test_comparison." + extension
            fig.savefig(destination/name, dpi=dpi, bbox_inches="tight", facecolor="white")
            files.append(name)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for j, row in enumerate(paired.itertuples()):
            ax.scatter(np.arange(4)+(j-(len(paired)-1)/2)*0.065,
                       [100*getattr(row, "test_"+metric+"_delta") for metric, _ in METRICS],
                       color=COLORS[j % len(COLORS)], label=f"Seed {row.seed}", zorder=3)
        ordered = test_stats.loc[[m for m, _ in METRICS]]
        ax.errorbar(np.arange(4)+0.27, ordered.delta_mean*100, yerr=ordered.delta_std*100,
                    fmt="D", color="#222222", capsize=5, label="Mean difference ± sample SD")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set(xticks=np.arange(4), xticklabels=[label for _, label in METRICS],
               ylabel="v1 − v0 (percentage points)", title="Change in test scores for each matched seed")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        fig.text(0.5, 0.015, f"Positive values favor v1. Error bars describe variability across {len(paired)} seed pairs, not confidence intervals.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(bottom=0.28, top=0.88)
        for extension in ("png", "pdf"):
            name = "test_paired_differences." + extension
            fig.savefig(destination/name, dpi=dpi, bbox_inches="tight", facecolor="white")
            files.append(name)
        plt.close(fig)
    return files


def export(baseline: Path, candidate: Path, output: Path, figures: Path, dpi: int = 300) -> None:
    combined, paired, stats, metadata = compare(baseline, candidate)
    if output.exists() or figures.exists():
        raise ValueError("Use new output and figure directories to preserve existing comparisons")
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    combined.to_csv(output/"per_seed_metrics.csv", index=False)
    paired.to_csv(output/"paired_seed_metrics.csv", index=False)
    stats.to_csv(output/"comparison_metrics.csv", index=False)
    metadata["figures"] = plot(paired, stats, figures, dpi)
    metadata["comparison_metrics"] = stats.to_dict(orient="records")
    metadata["figure_directory"] = str(figures.resolve())
    (output/"comparison.json").write_text(json.dumps(metadata, indent=2)+"\n")
    report = ["# Matched v0/v1 CNN comparison", "", f"Scores are mean ± sample SD across the same {len(paired)} training seeds. "
              "AP is average precision. Differences are v1 minus v0, in percentage points.", "",
              "| Test metric | v0 | v1 + auxiliary | Difference ± SD (pp) | Seeds improved |",
              "| --- | --- | --- | --- | --- |"]
    for row in stats[stats.split == "test"].itertuples():
        report.append(f"| {row.label} | {row.v0_mean:.4f} ± {row.v0_std:.4f} | {row.v1_mean:.4f} ± {row.v1_std:.4f} | "
                      f"{row.delta_mean*100:+.2f} ± {row.delta_std*100:.2f} | {row.improved_seeds}/{row.n} |")
    report += ["", "Both variants use identical candidates, labels, original inputs, star-grouped split, and shared training settings. "
               "Checkpoints maximize validation AP; thresholds maximize validation F1. All planned seeds are included.", "",
               metadata["limitations"], "", "The added inputs are catalog metadata, including vetting-related centroid diagnostics. "
               "Results describe catalog-assisted Kepler classification, not blind discovery or Kepler-to-TESS transfer.", "",
               f"Figures: {figures.resolve()}", ""]
    (output/"REPORT_SUMMARY.md").write_text("\n".join(report))
    (figures/"CAPTIONS.md").write_text("# Comparison figures\n\n"
        "`test_comparison`: per-seed v0/v1 test scores, with lines joining equal seed numbers; black diamonds show mean ± sample SD. "
        "The score axis is truncated and identical across metric panels.\n\n"
        "`test_paired_differences`: v1 minus v0 per seed in percentage points. Positive values favor v1. "
        "Black diamonds show the mean paired difference ± its sample SD.\n\n"
        "Both figures use all planned seeds. Error bars use ddof=1 and are not confidence intervals. "
        "Matching seeds does not imply identical random draws across different architectures. "
        "Checkpoints and thresholds are selected on validation.\n")
    for source, expected in metadata["source_files_sha256"].items():
        if file_hash(Path(source)) != expected:
            raise RuntimeError(f"Source changed during comparison: {source}")
    print(stats[stats.split == "test"].to_string(index=False))
    print(f"Comparison tables: {output}\nFigures: {figures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--figure_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    export(args.baseline, args.candidate, args.output_dir, args.figure_dir, args.dpi)


if __name__ == "__main__":
    main()
