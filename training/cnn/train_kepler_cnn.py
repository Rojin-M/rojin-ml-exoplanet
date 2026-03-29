#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    import torch
    from torch import nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for training. Activate the project's virtual environment "
        "and run `python setup/setup_torch_cuda.py` if torch is not installed there yet."
    ) from exc


@dataclass
class TrainConfig:
    base_dir: str
    dataset_path: str
    splits_path: str
    manifest_path: str
    output_root: str
    seed: int = 42
    batch_size: int = 256
    epochs: int = 60
    patience: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.20
    scalar_dropout: float = 0.10
    wide_dropout: float = 0.15
    num_workers: int = 4
    prepare_version: str = "unknown"
    threshold_metric: str = "f1"
    threshold_grid_size: int = 181
    compile_model: bool = False
    device: str = "auto"
    gradient_clip_norm: float = 1.0
    use_wide_branch: bool = False
    label_smoothing: float = 0.0
    view_noise_std: float = 0.0
    scalar_noise_std: float = 0.0
    flat_dropout_prob: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def normalize_view_channels(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> np.ndarray:
    out = x_other.copy()
    channel_mean = x_train[:, :, :2].mean(axis=(0, 1), keepdims=True)
    channel_std = x_train[:, :, :2].std(axis=(0, 1), keepdims=True)
    channel_std = np.clip(channel_std, 1e-6, None)
    out[:, :, :2] = (out[:, :, :2] - channel_mean) / channel_std
    return out.astype(np.float32)


def normalize_scalar_features(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.clip(x_train.std(axis=0, keepdims=True), 1e-6, None)
    return ((x_other - mean) / std).astype(np.float32)


def normalize_flat_features(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.clip(x_train.std(axis=0, keepdims=True), 1e-6, None)
    return ((x_other - mean) / std).astype(np.float32)


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


def regularize_training_batch(
    x_global: torch.Tensor,
    x_local: torch.Tensor,
    x_scalar: torch.Tensor,
    x_flat: torch.Tensor,
    y: torch.Tensor,
    *,
    view_noise_std: float,
    scalar_noise_std: float,
    flat_dropout_prob: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if view_noise_std > 0:
        x_global = x_global.clone()
        x_local = x_local.clone()
        x_global[:, :, :2] = x_global[:, :, :2] + torch.randn_like(x_global[:, :, :2]) * view_noise_std
        x_local[:, :, :2] = x_local[:, :, :2] + torch.randn_like(x_local[:, :, :2]) * view_noise_std

    if scalar_noise_std > 0:
        x_scalar = x_scalar + torch.randn_like(x_scalar) * scalar_noise_std

    if flat_dropout_prob > 0:
        keep_prob = max(1e-6, 1.0 - flat_dropout_prob)
        drop_mask = (torch.rand_like(x_flat) < keep_prob).float()
        x_flat = x_flat * drop_mask / keep_prob

    if label_smoothing > 0:
        y_for_loss = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
    else:
        y_for_loss = y
    return x_global, x_local, x_scalar, x_flat, y_for_loss


class KeplerTensorDataset(Dataset):
    def __init__(
        self,
        x_global: np.ndarray,
        x_local: np.ndarray,
        x_scalar: np.ndarray,
        x_flat: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x_global = torch.from_numpy(x_global[indices]).float()
        self.x_local = torch.from_numpy(x_local[indices]).float()
        self.x_scalar = torch.from_numpy(x_scalar[indices]).float()
        self.x_flat = torch.from_numpy(x_flat[indices]).float()
        self.y = torch.from_numpy(y[indices]).float()
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x_global[idx],
            self.x_local[idx],
            self.x_scalar[idx],
            self.x_flat[idx],
            self.y[idx],
            self.indices[idx],
        )


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.dropout(out)
        return self.activation(x + out)


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return torch.sum(x * weights, dim=-1)


class TransitBranch(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, dropout: float) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base_channels, kernel_size=7)
        self.stage1 = nn.Sequential(
            ResidualConvBlock(base_channels, kernel_size=7, dropout=dropout),
            ResidualConvBlock(base_channels, kernel_size=5, dropout=dropout),
        )
        self.down1 = ConvNormAct(base_channels, base_channels * 2, kernel_size=5, stride=2)
        self.stage2 = nn.Sequential(
            ResidualConvBlock(base_channels * 2, kernel_size=5, dropout=dropout),
            ResidualConvBlock(base_channels * 2, kernel_size=3, dropout=dropout),
        )
        self.down2 = ConvNormAct(base_channels * 2, base_channels * 4, kernel_size=3, stride=2)
        self.stage3 = nn.Sequential(
            ResidualConvBlock(base_channels * 4, kernel_size=3, dropout=dropout),
            ResidualConvBlock(base_channels * 4, kernel_size=3, dropout=dropout),
        )
        self.attn_pool = AttentionPool1d(base_channels * 4)
        self.head = nn.Sequential(
            nn.Linear(base_channels * 12, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        pooled = torch.cat(
            [
                torch.mean(x, dim=-1),
                torch.amax(x, dim=-1),
                self.attn_pool(x),
            ],
            dim=1,
        )
        return self.head(pooled)


class ScalarEncoder(nn.Module):
    def __init__(self, in_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, features: int, hidden_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, features),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(features)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.net(x)))


class WideEncoder(nn.Module):
    def __init__(self, in_features: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block1 = ResidualMLPBlock(features=512, hidden_features=768, dropout=dropout)
        self.block2 = ResidualMLPBlock(features=512, hidden_features=768, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


class ExoMinerStyleCNN(nn.Module):
    def __init__(
        self,
        scalar_features: int,
        flat_features: int,
        dropout: float,
        scalar_dropout: float,
        wide_dropout: float,
        use_wide_branch: bool,
    ) -> None:
        super().__init__()
        self.global_branch = TransitBranch(in_channels=3, base_channels=40, dropout=dropout)
        self.local_branch = TransitBranch(in_channels=3, base_channels=40, dropout=dropout)
        self.scalar_branch = ScalarEncoder(scalar_features, dropout=scalar_dropout)
        self.use_wide_branch = use_wide_branch
        self.wide_branch = WideEncoder(flat_features, dropout=wide_dropout) if use_wide_branch else None
        fusion_dim = 128 + 128 + 128 + 128 + 64 + (128 if use_wide_branch else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(
        self,
        x_global: torch.Tensor,
        x_local: torch.Tensor,
        x_scalar: torch.Tensor,
        x_flat: torch.Tensor,
    ) -> torch.Tensor:
        global_embed = self.global_branch(x_global)
        local_embed = self.local_branch(x_local)
        scalar_embed = self.scalar_branch(x_scalar)
        fused_parts = [
            global_embed,
            local_embed,
            torch.abs(global_embed - local_embed),
            global_embed * local_embed,
            scalar_embed,
        ]
        if self.wide_branch is not None:
            fused_parts.append(self.wide_branch(x_flat))
        fused = torch.cat(fused_parts, dim=1)
        return self.classifier(fused).squeeze(1)


def make_dataloader(
    dataset: KeplerTensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    gradient_clip_norm: Optional[float] = None,
    label_smoothing: float = 0.0,
    view_noise_std: float = 0.0,
    scalar_noise_std: float = 0.0,
    flat_dropout_prob: float = 0.0,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []

    context = torch.enable_grad if train_mode else torch.no_grad
    with context():
        for x_global, x_local, x_scalar, x_flat, y, indices in loader:
            x_global = x_global.to(device, non_blocking=True)
            x_local = x_local.to(device, non_blocking=True)
            x_scalar = x_scalar.to(device, non_blocking=True)
            x_flat = x_flat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_for_loss = y

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                x_global, x_local, x_scalar, x_flat, y_for_loss = regularize_training_batch(
                    x_global,
                    x_local,
                    x_scalar,
                    x_flat,
                    y,
                    view_noise_std=view_noise_std,
                    scalar_noise_std=scalar_noise_std,
                    flat_dropout_prob=flat_dropout_prob,
                    label_smoothing=label_smoothing,
                )

            amp_enabled = scaler is not None and device.type == "cuda"
            with autocast(enabled=amp_enabled):
                logits = model(x_global, x_local, x_scalar, x_flat)
                loss = loss_fn(logits, y_for_loss)

            if train_mode:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if gradient_clip_norm and gradient_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm and gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            total_loss += float(loss.detach().cpu().item()) * y.shape[0]
            all_probs.append(probs)
            all_targets.append(y.detach().cpu().numpy())
            all_indices.append(indices.numpy())

    n = len(loader.dataset)
    mean_loss = total_loss / max(1, n)
    return (
        mean_loss,
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_targets, axis=0).astype(np.int64),
        np.concatenate(all_indices, axis=0).astype(np.int64),
    )


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    loss, probs, targets, indices = run_epoch(
        model,
        loader,
        optimizer=None,
        loss_fn=loss_fn,
        device=device,
        scaler=None,
    )
    metrics = compute_metrics(targets, probs, threshold=threshold)
    metrics["loss"] = float(loss)
    return {"metrics": metrics, "probs": probs, "targets": targets, "indices": indices}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def load_training_data(cfg: TrainConfig) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    dataset = np.load(cfg.dataset_path, allow_pickle=True)
    splits = np.load(cfg.splits_path)
    manifest = pd.read_csv(cfg.manifest_path)

    x_global = dataset["X_global"].astype(np.float32)
    x_local = dataset["X_local"].astype(np.float32)
    x_scalar = dataset["X_scalar"].astype(np.float32)
    x_flat = dataset["X"].astype(np.float32)
    y = dataset["y"].astype(np.int64)

    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)
    test_idx = splits["test_idx"].astype(np.int64)

    x_global = normalize_view_channels(x_global[train_idx], x_global)
    x_local = normalize_view_channels(x_local[train_idx], x_local)
    x_scalar = normalize_scalar_features(x_scalar[train_idx], x_scalar)
    x_flat = normalize_flat_features(x_flat[train_idx], x_flat)

    arrays = {
        "x_global": x_global,
        "x_local": x_local,
        "x_scalar": x_scalar,
        "x_flat": x_flat,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return arrays, manifest


def train_model(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = detect_device(cfg.device)
    arrays, manifest = load_training_data(cfg)

    dataset_name = Path(cfg.dataset_path).stem
    run_name = f"{dataset_name}_cnn_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["train_idx"],
    )
    val_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["val_idx"],
    )
    test_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["test_idx"],
    )

    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, device=device)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, device=device)
    test_loader = make_dataloader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, device=device)

    scalar_features = int(arrays["x_scalar"].shape[1])
    flat_features = int(arrays["x_flat"].shape[1])
    model = ExoMinerStyleCNN(
        scalar_features=scalar_features,
        flat_features=flat_features,
        dropout=cfg.dropout,
        scalar_dropout=cfg.scalar_dropout,
        wide_dropout=cfg.wide_dropout,
        use_wide_branch=cfg.use_wide_branch,
    ).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    y_train = arrays["y"][arrays["train_idx"]]
    pos_count = max(1, int(y_train.sum()))
    neg_count = max(1, int(len(y_train) - pos_count))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler: Optional[GradScaler] = GradScaler(enabled=True) if device.type == "cuda" else None

    history: List[Dict[str, Any]] = []
    best_val_pr_auc = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint = run_dir / "best_model.pt"

    train_threshold = 0.5
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_probs, train_targets, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            gradient_clip_norm=cfg.gradient_clip_norm,
            label_smoothing=cfg.label_smoothing,
            view_noise_std=cfg.view_noise_std,
            scalar_noise_std=cfg.scalar_noise_std,
            flat_dropout_prob=cfg.flat_dropout_prob,
        )
        scheduler.step()
        train_threshold = build_threshold(train_targets, train_probs, cfg.threshold_grid_size)
        train_metrics = compute_metrics(train_targets, train_probs, threshold=train_threshold)
        train_metrics["loss"] = float(train_loss)

        val_eval = evaluate_split(model, val_loader, loss_fn, device=device, threshold=train_threshold)
        val_metrics = val_eval["metrics"]
        val_tuned_threshold = build_threshold(val_eval["targets"], val_eval["probs"], cfg.threshold_grid_size)
        val_metrics = compute_metrics(val_eval["targets"], val_eval["probs"], threshold=val_tuned_threshold) | {
            "loss": val_metrics["loss"],
            "threshold": val_tuned_threshold,
        }

        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_f1": train_metrics["f1"],
            "train_threshold": train_threshold,
            "val_loss": val_metrics["loss"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
            "val_threshold": val_tuned_threshold,
        }
        history.append(row)

        improved = val_metrics["pr_auc"] > best_val_pr_auc
        if improved:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_pr_auc": best_val_pr_auc,
                    "val_threshold": val_tuned_threshold,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_pr_auc={train_metrics['pr_auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping at epoch {epoch} after {cfg.patience} epochs without validation PR-AUC improvement.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_threshold = float(checkpoint.get("val_threshold", 0.5))

    train_eval = evaluate_split(model, train_loader, loss_fn, device=device, threshold=best_threshold)
    val_eval = evaluate_split(model, val_loader, loss_fn, device=device, threshold=best_threshold)
    test_eval = evaluate_split(model, test_loader, loss_fn, device=device, threshold=best_threshold)

    save_predictions_csv(run_dir / "val_predictions.csv", manifest, val_eval["indices"], val_eval["probs"], best_threshold)
    save_predictions_csv(run_dir / "test_predictions.csv", manifest, test_eval["indices"], test_eval["probs"], best_threshold)

    summary = {
        "run_name": run_name,
        "dataset_path": cfg.dataset_path,
        "splits_path": cfg.splits_path,
        "manifest_path": cfg.manifest_path,
        "device": str(device),
        "use_wide_branch": cfg.use_wide_branch,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(test_ds)),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", asdict(cfg))

    print("\nBest validation/test summary")
    print(json.dumps(summary, indent=2))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ExoMiner-inspired Kepler CNN on processed dataset artifacts.")
    parser.add_argument("--base_dir", type=str, default="/local00/student/moradian/rojin-ml-exoplanet")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--splits_path", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--scalar_dropout", type=float, default=0.10)
    parser.add_argument("--wide_dropout", type=float, default=0.15)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wide_branch", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--view_noise_std", type=float, default=0.0)
    parser.add_argument("--scalar_noise_std", type=float, default=0.0)
    parser.add_argument("--flat_dropout_prob", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    base_dir = Path(args.base_dir)
    processed_dir = base_dir / "data" / "processed" / "kepler"
    dataset_path = Path(args.dataset_path) if args.dataset_path else auto_find_path(processed_dir, "dataset_kepler_*.npz")
    splits_path = Path(args.splits_path) if args.splits_path else auto_find_path(processed_dir, "splits_kepler_*.npz")
    manifest_path = Path(args.manifest_path) if args.manifest_path else infer_manifest_path(processed_dir, dataset_path)
    output_root = Path(args.output_root) if args.output_root else base_dir / "outputs" / "cnn"
    version_match = re.search(r"dataset_kepler_(v\d+)_", dataset_path.name)
    prepare_version = version_match.group(1) if version_match else "unknown"
    return TrainConfig(
        base_dir=str(base_dir),
        dataset_path=str(dataset_path),
        splits_path=str(splits_path),
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        scalar_dropout=args.scalar_dropout,
        wide_dropout=args.wide_dropout,
        num_workers=args.num_workers,
        prepare_version=prepare_version,
        compile_model=args.compile_model,
        device=args.device,
        gradient_clip_norm=args.gradient_clip_norm,
        use_wide_branch=args.wide_branch,
        label_smoothing=args.label_smoothing,
        view_noise_std=args.view_noise_std,
        scalar_noise_std=args.scalar_noise_std,
        flat_dropout_prob=args.flat_dropout_prob,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_dir = train_model(cfg)
    print(f"\nSaved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
