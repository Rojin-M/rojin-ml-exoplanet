#!/usr/bin/env python3
"""Verify prediction metrics and compare every model in a completed reproduction workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

METRICS = (("accuracy", "Accuracy"), ("f1", "F1"), ("pr_auc", "Average precision"), ("roc_auc", "ROC-AUC"))


def run_row(path, model, evidence, seed=None):
    summary = json.loads((path / "summary.json").read_text())
    cfg = json.loads((path / "config.json").read_text())
    evaluation = path / "test_evaluation.json"
    test = json.loads(evaluation.read_text())["test_metrics"] if evaluation.exists() else summary.get("test_metrics")
    row = {"model": model, "evidence": evidence, "run_id": path.name,
           "seed": cfg.get("seed", seed), "best_epoch": summary.get("best_epoch"),
           "threshold": summary["best_threshold"]}
    provenance = {}
    for split in ("val", "test"):
        expected = summary["val_metrics"] if split == "val" else test
        if expected is None:
            continue
        p = path / (split + "_predictions.csv")
        frame = pd.read_csv(p)
        if "probability_stacked" in frame.columns:
            frame = frame.rename(columns={"probability_stacked": "probability", "prediction_stacked": "prediction"})
        if not frame.candidate_id.is_unique or not frame.probability.between(0, 1).all():
            raise ValueError(f"Invalid predictions: {p}")
        manifest = pd.read_csv(ROOT / "data/processed/kepler/v0/manifest_kepler_v0.csv")
        with np.load(ROOT / "data/processed/kepler/v0/splits_kepler_v0_seed42.npz") as saved:
            expected_rows = manifest.iloc[saved[split + "_idx"]]
        cols = ["candidate_id", "kepid", "y"]
        pd.testing.assert_frame_equal(frame[cols].sort_values("candidate_id").reset_index(drop=True),
                                      expected_rows[cols].sort_values("candidate_id").reset_index(drop=True))
        y, probability = frame.y.to_numpy(), frame.probability.to_numpy()
        prediction = probability >= row["threshold"]
        np.testing.assert_array_equal(prediction.astype(int), frame.prediction)
        actual = {"accuracy": accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0),
                  "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0),
                  "pr_auc": average_precision_score(y, probability), "roc_auc": roc_auc_score(y, probability)}
        for key, value in actual.items():
            np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12)
            row[split + "_" + key] = value
        row[split + "_false_positives"] = int(((y == 0) & prediction).sum())
        row[split + "_false_negatives"] = int(((y == 1) & ~prediction).sum())
        provenance[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for metric, _ in METRICS:
        row["train_" + metric] = summary.get("train_metrics", {}).get(metric)
    for p in (path / "summary.json", path / "config.json"):
        provenance[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return row, provenance


def export(rows, output, provenance):
    output.mkdir(parents=True, exist_ok=False)
    data = pd.DataFrame(rows)
    data.to_csv(output / "runs.csv", index=False)
    summaries = []
    for (model, evidence), group in data.groupby(["model", "evidence"], sort=False):
        for metric, label in METRICS:
            scores = group["test_" + metric].dropna()
            if len(scores):
                summaries.append({"model": model, "evidence": evidence, "metric": metric, "label": label,
                                  "n": len(scores), "mean": scores.mean(), "std": scores.std(ddof=1) if len(scores) > 1 else None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    lines = ["# Model comparison", "", "AP is average precision. Values are test scores on the same previously consulted Kepler split.",
             "Five-seed entries report mean ± sample SD; individual runs have no seed uncertainty estimate.", "",
             "| Model | Evidence | Runs | Accuracy | F1 | AP | ROC-AUC |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (model, evidence), group in summary.groupby(["model", "evidence"], sort=False):
        group = group.set_index("metric")
        values = [f"{group.loc[k, 'mean']:.4f}" + (f" ± {group.loc[k, 'std']:.4f}" if group.loc[k, 'n'] > 1 else "") for k, _ in METRICS]
        lines.append(f"| {model} | {evidence} | {int(group.iloc[0]['n'])} | " + " | ".join(values) + " |")
    lines += ["", "Architecture, inputs, training recipes, and numerical precision can differ between model families. This is a descriptive comparison.",
              "The paired v0/v1 studies provide stronger evidence about their combined metadata-and-branch extension; they do not isolate individual features or added capacity.",
              "Repeated runs on this split do not create an untouched test set. Sample SD measures training-seed variability, not uncertainty on a new mission.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(summary) / 9)))
    for ax, metric in zip(axes, ("pr_auc", "f1")):
        scores = summary[summary.metric == metric].reset_index(drop=True)
        for i, row in scores.iterrows():
            ax.errorbar(row["mean"], i, xerr=row["std"] if row["n"] > 1 else None,
                        fmt="D" if row["n"] > 1 else "o", color="#0072B2" if row["n"] > 1 else "#777777", capsize=4)
        ax.set(yticks=np.arange(len(scores)), yticklabels=scores.model, xlabel="Test " + dict(METRICS)[metric], xlim=(0, 1))
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    fig.suptitle("Kepler model comparison")
    fig.text(.5, .01, "Diamonds: five-seed mean ± sample SD. Circles: individual runs. Same split; different recipes/inputs. Descriptive comparison.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("comparison." + extension), dpi=220, bbox_inches="tight")
    plt.close(fig)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Comparison exported:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.workflow_dir.resolve()
    rows, provenance = [], {}

    def add(path, name, evidence, seed=None):
        row, sources = run_row(path, name, evidence, seed)
        rows.append(row); provenance[path.name] = sources

    for name, label in (("hgb_scalar", "HGB scalar"), ("hgb_full", "HGB full"), ("cnn_baseline", "CNN baseline"),
                        ("cnn_wide", "CNN wide branch"), ("cnn_regularized", "CNN regularized"), ("blend", "CNN + HGB blend")):
        record = json.loads((work / name / "run.json").read_text())
        if record["status"] != "complete":
            raise ValueError(f"Incomplete run: {name}")
        add(Path(record["run_dir"]), label, "individual run")
    for name, label in (("cnn_v0", "CNN v0"), ("cnn_v1", "CNN v1 + metadata")):
        experiment = json.loads((work / name / "experiment.json").read_text())
        if experiment["status"] != "complete" or experiment["seeds"] != [42,43,44,45,46]:
            raise ValueError(f"Incomplete five-seed experiment: {name}")
        for record in experiment["runs"]:
            add(Path(record["run_dir"]), label, "five matched seeds")
    experiment = json.loads((work / "transformer/experiment.json").read_text())
    if experiment["status"] != "complete":
        raise ValueError("Incomplete transformer study")
    selected = experiment["selection"]["selected_variant"]
    for record in experiment["runs"]:
        if record["variant"] in (selected, "selected_aux"):
            if record.get("evaluation_status") != "complete":
                raise ValueError("Missing final transformer evaluation")
            label = "Transformer v1 + metadata" if record["variant"] == "selected_aux" else "Transformer v0"
            add(Path(record["run_dir"]), label, "five matched seeds")
    export(rows, args.output_dir.resolve(), provenance)


if __name__ == "__main__":
    main()
