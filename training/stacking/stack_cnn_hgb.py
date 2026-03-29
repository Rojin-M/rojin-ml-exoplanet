#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class StackConfig:
    base_dir: str
    cnn_run_dir: str
    hgb_run_dir: str
    output_root: str
    method: str = "blend"
    selection_metric: str = "f1"
    threshold_grid_size: int = 181
    weight_grid_size: int = 201


def auto_find_latest_run(outputs_dir: Path, summary_pattern: str) -> Path:
    summaries = sorted(outputs_dir.glob(summary_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summaries:
        raise FileNotFoundError(f"No run summaries matching {summary_pattern} in {outputs_dir}")
    return summaries[0].parent


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def best_f1_threshold_and_score(y_true: np.ndarray, probs: np.ndarray, grid_size: int) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, grid_size)
    y = y_true.astype(bool)[:, None]
    preds = probs[:, None] >= thresholds[None, :]

    tp = np.sum(preds & y, axis=0)
    fp = np.sum(preds & ~y, axis=0)
    fn = np.sum(~preds & y, axis=0)
    denom = (2 * tp) + fp + fn
    f1_scores = np.where(denom > 0, (2.0 * tp) / denom, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def merge_prediction_frames(cnn_path: Path, hgb_path: Path) -> pd.DataFrame:
    merge_cols = ["candidate_id", "kepid"]
    cnn = pd.read_csv(cnn_path)
    hgb = pd.read_csv(hgb_path)

    merged = cnn.merge(
        hgb[merge_cols + ["probability", "prediction"]],
        on=merge_cols,
        how="inner",
        suffixes=("_cnn", "_hgb"),
    )
    if merged.empty:
        raise ValueError(f"No overlapping prediction rows between {cnn_path} and {hgb_path}")

    if "y" in merged.columns:
        merged["y"] = merged["y"].astype(np.int64)
    elif "label" in merged.columns:
        merged["y"] = merged["label"].astype(np.int64)
    else:
        raise KeyError("Predictions are missing a label column. Expected 'y' or 'label'.")

    return merged


def metric_for_selection(y_true: np.ndarray, probs: np.ndarray, threshold: float, selection_metric: str) -> float:
    if selection_metric == "f1":
        _, score = best_f1_threshold_and_score(y_true, probs, 181)
        return score
    if selection_metric == "pr_auc":
        return float(average_precision_score(y_true, probs))
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def find_best_blend(
    y_val: np.ndarray,
    cnn_val: np.ndarray,
    hgb_val: np.ndarray,
    cfg: StackConfig,
) -> Tuple[float, float, Dict[str, float]]:
    best_weight = 0.5
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] | None = None

    for weight in np.linspace(0.0, 1.0, cfg.weight_grid_size):
        stacked_val = weight * cnn_val + (1.0 - weight) * hgb_val
        threshold, best_f1 = best_f1_threshold_and_score(y_val, stacked_val, cfg.threshold_grid_size)
        score = best_f1 if cfg.selection_metric == "f1" else metric_for_selection(y_val, stacked_val, threshold, cfg.selection_metric)
        if score > best_score:
            best_score = score
            best_weight = float(weight)
            best_threshold = float(threshold)
            best_metrics = compute_metrics(y_val, stacked_val, best_threshold)

    assert best_metrics is not None
    return best_weight, best_threshold, best_metrics


def save_predictions_csv(
    path: Path,
    frame: pd.DataFrame,
    stacked_probs: np.ndarray,
    threshold: float,
) -> None:
    out = frame.copy()
    out["probability_stacked"] = stacked_probs
    out["prediction_stacked"] = (stacked_probs >= threshold).astype(np.int64)
    out.to_csv(path, index=False)


def infer_dataset_name(cnn_summary: Dict[str, Any]) -> str:
    dataset_path = str(cnn_summary.get("dataset_path", ""))
    if dataset_path:
        return Path(dataset_path).stem
    match = re.search(r"(dataset_kepler_[^/]+)", str(cnn_summary.get("run_name", "")))
    if match:
        return match.group(1)
    return "dataset_kepler"


def run_stack(cfg: StackConfig) -> Path:
    cnn_run_dir = Path(cfg.cnn_run_dir)
    hgb_run_dir = Path(cfg.hgb_run_dir)

    cnn_summary = load_json(cnn_run_dir / "summary.json")
    hgb_summary = load_json(hgb_run_dir / "summary.json")

    val = merge_prediction_frames(cnn_run_dir / "val_predictions.csv", hgb_run_dir / "val_predictions.csv")
    test = merge_prediction_frames(cnn_run_dir / "test_predictions.csv", hgb_run_dir / "test_predictions.csv")

    y_val = val["y"].to_numpy(dtype=np.int64)
    y_test = test["y"].to_numpy(dtype=np.int64)
    cnn_val = val["probability_cnn"].to_numpy(dtype=np.float64)
    hgb_val = val["probability_hgb"].to_numpy(dtype=np.float64)
    cnn_test = test["probability_cnn"].to_numpy(dtype=np.float64)
    hgb_test = test["probability_hgb"].to_numpy(dtype=np.float64)

    extra_summary: Dict[str, Any] = {}
    if cfg.method == "blend":
        weight, threshold, val_metrics = find_best_blend(y_val, cnn_val, hgb_val, cfg)
        stacked_val = weight * cnn_val + (1.0 - weight) * hgb_val
        stacked_test = weight * cnn_test + (1.0 - weight) * hgb_test
        extra_summary["cnn_weight"] = weight
        extra_summary["hgb_weight"] = float(1.0 - weight)
    elif cfg.method == "logreg":
        model = LogisticRegression(max_iter=2000)
        model.fit(np.column_stack([cnn_val, hgb_val]), y_val)
        stacked_val = model.predict_proba(np.column_stack([cnn_val, hgb_val]))[:, 1]
        stacked_test = model.predict_proba(np.column_stack([cnn_test, hgb_test]))[:, 1]
        threshold, _ = best_f1_threshold_and_score(y_val, stacked_val, cfg.threshold_grid_size)
        val_metrics = compute_metrics(y_val, stacked_val, threshold)
        extra_summary["stacker_coefficients"] = model.coef_.tolist()
        extra_summary["stacker_intercept"] = model.intercept_.tolist()
    else:
        raise ValueError(f"Unsupported stacking method: {cfg.method}")

    dataset_name = infer_dataset_name(cnn_summary)
    run_name = f"{dataset_name}_stack_{cfg.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_predictions_csv(run_dir / "val_predictions.csv", val, stacked_val, threshold)
    save_predictions_csv(run_dir / "test_predictions.csv", test, stacked_test, threshold)

    summary = {
        "run_name": run_name,
        "method": cfg.method,
        "selection_metric": cfg.selection_metric,
        "cnn_run_dir": str(cnn_run_dir),
        "hgb_run_dir": str(hgb_run_dir),
        "best_threshold": float(threshold),
        "base_model_metrics": {
            "cnn_val_metrics": cnn_summary.get("val_metrics"),
            "cnn_test_metrics": cnn_summary.get("test_metrics"),
            "hgb_val_metrics": hgb_summary.get("val_metrics"),
            "hgb_test_metrics": hgb_summary.get("test_metrics"),
        },
        "val_metrics": val_metrics,
        "test_metrics": compute_metrics(y_test, stacked_test, threshold),
        "val_size": int(len(val)),
        "test_size": int(len(test)),
        **extra_summary,
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", asdict(cfg))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved run artifacts to: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend or stack saved CNN and HGB prediction runs.")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--cnn_run_dir", type=str, default="")
    parser.add_argument("--hgb_run_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--method", choices=["blend", "logreg"], default="blend")
    parser.add_argument("--selection_metric", choices=["f1", "pr_auc"], default="f1")
    parser.add_argument("--threshold_grid_size", type=int, default=181)
    parser.add_argument("--weight_grid_size", type=int, default=201)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StackConfig:
    base_dir = Path(args.base_dir)
    outputs_dir = base_dir / "outputs"
    cnn_run_dir = Path(args.cnn_run_dir) if args.cnn_run_dir else auto_find_latest_run(outputs_dir / "cnn", "*/summary.json")
    hgb_run_dir = Path(args.hgb_run_dir) if args.hgb_run_dir else auto_find_latest_run(outputs_dir / "tabular", "*/summary.json")
    output_root = Path(args.output_root) if args.output_root else outputs_dir / "stacking"
    return StackConfig(
        base_dir=str(base_dir),
        cnn_run_dir=str(cnn_run_dir),
        hgb_run_dir=str(hgb_run_dir),
        output_root=str(output_root),
        method=args.method,
        selection_metric=args.selection_metric,
        threshold_grid_size=args.threshold_grid_size,
        weight_grid_size=args.weight_grid_size,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_stack(cfg)


if __name__ == "__main__":
    main()
