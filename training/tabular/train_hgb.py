#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.paths import processed_path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class TrainConfig:
    base_dir: str
    dataset_path: str
    splits_path: str
    manifest_path: str
    output_root: str
    feature_set: str = "scalar"
    seed: int = 42
    max_iter: int = 400
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    max_depth: int = 0
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 30
    threshold_grid_size: int = 181
    regularization_search: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def auto_find_path(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {base_dir}")
    return matches[0]


def infer_manifest_path(processed_dir: Path, dataset_path: Path) -> Path:
    match = re.search(r"dataset_kepler_(v\d+)_", dataset_path.name)
    if match:
        candidate = processed_dir / f"manifest_kepler_{match.group(1)}.csv"
        if candidate.exists():
            return candidate
    return processed_dir / "manifest_kepler_v0.csv"


def build_threshold(y_true: np.ndarray, probs: np.ndarray, grid_size: int) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, grid_size):
        preds = (probs >= threshold).astype(np.int64)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


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


def load_feature_matrix(dataset: np.lib.npyio.NpzFile, feature_set: str) -> np.ndarray:
    if feature_set == "scalar":
        return dataset["X_scalar"].astype(np.float32)
    if feature_set == "full":
        return dataset["X"].astype(np.float32)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def save_predictions_csv(
    path: Path,
    manifest: pd.DataFrame,
    indices: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> None:
    part = manifest.iloc[indices].copy()
    part["probability"] = probs
    part["prediction"] = (probs >= threshold).astype(np.int64)
    part.to_csv(path, index=False)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_candidate_configs(cfg: TrainConfig) -> List[Dict[str, Any]]:
    base = {
        "name": "base",
        "learning_rate": cfg.learning_rate,
        "max_iter": cfg.max_iter,
        "max_leaf_nodes": cfg.max_leaf_nodes,
        "max_depth": None if cfg.max_depth <= 0 else cfg.max_depth,
        "min_samples_leaf": cfg.min_samples_leaf,
        "l2_regularization": cfg.l2_regularization,
    }
    if not cfg.regularization_search:
        return [base]

    if cfg.feature_set == "full":
        candidates = [
            base,
            {
                "name": "reg_mild",
                "learning_rate": min(cfg.learning_rate, 0.04),
                "max_iter": max(cfg.max_iter, 450),
                "max_leaf_nodes": min(cfg.max_leaf_nodes, 31),
                "max_depth": 8,
                "min_samples_leaf": max(cfg.min_samples_leaf, 40),
                "l2_regularization": max(cfg.l2_regularization, 1.0),
            },
            {
                "name": "reg_balanced",
                "learning_rate": min(cfg.learning_rate, 0.03),
                "max_iter": max(cfg.max_iter, 500),
                "max_leaf_nodes": 15,
                "max_depth": 6,
                "min_samples_leaf": max(cfg.min_samples_leaf, 50),
                "l2_regularization": max(cfg.l2_regularization, 3.0),
            },
            {
                "name": "reg_strong",
                "learning_rate": min(cfg.learning_rate, 0.02),
                "max_iter": max(cfg.max_iter, 550),
                "max_leaf_nodes": 15,
                "max_depth": 5,
                "min_samples_leaf": max(cfg.min_samples_leaf, 80),
                "l2_regularization": max(cfg.l2_regularization, 8.0),
            },
        ]
    else:
        candidates = [
            base,
            {
                "name": "reg_mild",
                "learning_rate": min(cfg.learning_rate, 0.04),
                "max_iter": max(cfg.max_iter, 450),
                "max_leaf_nodes": min(cfg.max_leaf_nodes, 31),
                "max_depth": 8,
                "min_samples_leaf": max(cfg.min_samples_leaf, 30),
                "l2_regularization": max(cfg.l2_regularization, 0.5),
            },
            {
                "name": "reg_balanced",
                "learning_rate": min(cfg.learning_rate, 0.03),
                "max_iter": max(cfg.max_iter, 500),
                "max_leaf_nodes": 15,
                "max_depth": 6,
                "min_samples_leaf": max(cfg.min_samples_leaf, 40),
                "l2_regularization": max(cfg.l2_regularization, 2.0),
            },
        ]

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate["learning_rate"],
            candidate["max_iter"],
            candidate["max_leaf_nodes"],
            candidate["max_depth"],
            candidate["min_samples_leaf"],
            candidate["l2_regularization"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def build_hgb_model(candidate: Dict[str, Any], cfg: TrainConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(candidate["learning_rate"]),
        max_iter=int(candidate["max_iter"]),
        max_leaf_nodes=int(candidate["max_leaf_nodes"]),
        max_depth=candidate["max_depth"],
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        l2_regularization=float(candidate["l2_regularization"]),
        early_stopping=cfg.early_stopping,
        validation_fraction=cfg.validation_fraction,
        n_iter_no_change=cfg.n_iter_no_change,
        random_state=cfg.seed,
    )


def candidate_rank_key(result: Dict[str, Any]) -> Tuple[float, float, float, float]:
    gap = result["train_metrics"]["pr_auc"] - result["val_metrics"]["pr_auc"]
    complexity_penalty = float(result["candidate"]["max_leaf_nodes"])
    return (
        float(result["val_metrics"]["pr_auc"]),
        float(result["val_metrics"]["f1"]),
        float(-gap),
        -complexity_penalty,
    )


def train_model(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    dataset = np.load(processed_path(cfg.dataset_path), allow_pickle=True)
    splits = np.load(processed_path(cfg.splits_path))
    manifest = pd.read_csv(processed_path(cfg.manifest_path))

    x = load_feature_matrix(dataset, cfg.feature_set)
    y = dataset["y"].astype(np.int64)
    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    candidates = make_candidate_configs(cfg)
    candidate_results: List[Dict[str, Any]] = []
    best_model: HistGradientBoostingClassifier | None = None
    best_threshold = 0.5
    best_val_probs: np.ndarray | None = None
    best_result: Dict[str, Any] | None = None

    for candidate in candidates:
        model = build_hgb_model(candidate, cfg)
        model.fit(x_train, y_train, sample_weight=sample_weight)
        train_probs = model.predict_proba(x_train)[:, 1]
        val_probs = model.predict_proba(x_val)[:, 1]
        threshold = build_threshold(y_val, val_probs, cfg.threshold_grid_size)
        result = {
            "candidate": candidate,
            "train_metrics": compute_metrics(y_train, train_probs, threshold),
            "val_metrics": compute_metrics(y_val, val_probs, threshold),
            "threshold": threshold,
            "n_iter_": int(getattr(model, "n_iter_", candidate["max_iter"])),
        }
        candidate_results.append(result)
        print(
            "candidate="
            f"{candidate['name']} val_pr_auc={result['val_metrics']['pr_auc']:.4f} "
            f"val_f1={result['val_metrics']['f1']:.4f} "
            f"train_val_pr_gap={result['train_metrics']['pr_auc'] - result['val_metrics']['pr_auc']:.4f}"
        )
        if best_result is None or candidate_rank_key(result) > candidate_rank_key(best_result):
            best_result = result
            best_model = model
            best_threshold = threshold
            best_val_probs = val_probs

    assert best_model is not None
    assert best_result is not None
    assert best_val_probs is not None

    train_probs = best_model.predict_proba(x_train)[:, 1]
    val_probs = best_val_probs
    test_probs = best_model.predict_proba(x_test)[:, 1]

    dataset_name = Path(cfg.dataset_path).stem
    run_name = f"{dataset_name}_hgb_{cfg.feature_set}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": run_name,
        "dataset_path": cfg.dataset_path,
        "splits_path": cfg.splits_path,
        "manifest_path": cfg.manifest_path,
        "feature_set": cfg.feature_set,
        "best_threshold": best_threshold,
        "selected_candidate": best_result["candidate"],
        "candidate_results": candidate_results,
        "train_metrics": compute_metrics(y_train, train_probs, best_threshold),
        "val_metrics": compute_metrics(y_val, val_probs, best_threshold),
        "test_metrics": compute_metrics(y_test, test_probs, best_threshold),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "n_iter_": int(getattr(best_model, "n_iter_", best_result["candidate"]["max_iter"])),
    }

    save_predictions_csv(run_dir / "val_predictions.csv", manifest, val_idx, val_probs, best_threshold)
    save_predictions_csv(run_dir / "test_predictions.csv", manifest, test_idx, test_probs, best_threshold)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", asdict(cfg))
    write_json(run_dir / "candidate_results.json", {"candidate_results": candidate_results, "selected_candidate": best_result["candidate"]})

    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved run artifacts to: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HistGradientBoosting baselines on processed Kepler artifacts.")
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--splits_path", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--feature_set", choices=["scalar", "full"], default="scalar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_leaf_nodes", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=0)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--l2_regularization", type=float, default=0.0)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--n_iter_no_change", type=int, default=30)
    parser.add_argument("--single_config", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    base_dir = Path(args.base_dir)
    processed_dir = base_dir / "data" / "processed" / "kepler" / "v0"
    dataset_path = Path(args.dataset_path) if args.dataset_path else auto_find_path(processed_dir, "dataset_kepler_*.npz")
    splits_path = Path(args.splits_path) if args.splits_path else auto_find_path(processed_dir, "splits_kepler_*.npz")
    manifest_path = Path(args.manifest_path) if args.manifest_path else infer_manifest_path(processed_dir, dataset_path)
    output_root = Path(args.output_root) if args.output_root else base_dir / "outputs" / "tabular"
    return TrainConfig(
        base_dir=str(base_dir),
        dataset_path=str(dataset_path),
        splits_path=str(splits_path),
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        feature_set=args.feature_set,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        regularization_search=not args.single_config,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    train_model(cfg)


if __name__ == "__main__":
    main()
