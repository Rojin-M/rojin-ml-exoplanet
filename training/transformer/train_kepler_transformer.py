#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.cuda.amp import GradScaler
import pandas as pd

from training.cnn.train_kepler_cnn import (
    KeplerTensorDataset,
    auto_find_path,
    build_threshold,
    compute_metrics,
    detect_device,
    evaluate_split,
    infer_manifest_path,
    load_training_data,
    make_dataloader,
    run_epoch,
    save_predictions_csv,
    set_seed,
    tracking_run,
    write_json,
)


@dataclass
class TrainConfig:
    base_dir: str
    dataset_path: str
    splits_path: str
    manifest_path: str
    output_root: str
    seed: int = 42
    batch_size: int = 192
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    dropout: float = 0.15
    attention_dropout: float = 0.10
    token_dropout: float = 0.05
    scalar_dropout: float = 0.10
    num_workers: int = 4
    prepare_version: str = "unknown"
    threshold_grid_size: int = 181
    compile_model: bool = False
    device: str = "auto"
    gradient_clip_norm: float = 1.0
    d_model: int = 128
    num_heads: int = 8
    global_depth: int = 4
    local_depth: int = 3
    mlp_ratio: float = 4.0
    label_smoothing: float = 0.0
    view_noise_std: float = 0.0
    scalar_noise_std: float = 0.0
    flat_dropout_prob: float = 0.0
    use_input_norm: bool = True
    use_aux_branch: bool = False
    aux_dropout: float = 0.10
    evaluate_test: bool = True
    wandb_mode: str = "disabled"
    wandb_project: str = "kepler-practical"
    wandb_entity: str = ""
    wandb_group: str = ""
    use_amp: bool = True


class ScalarEncoder(nn.Module):
    def __init__(self, in_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class TransformerBranch(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        token_dropout: float,
        use_input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_dropout = token_dropout
        self.use_input_norm = use_input_norm
        self.input_norm = nn.LayerNorm(input_channels)
        self.input_norm.requires_grad_(use_input_norm)
        self.input_proj = nn.Linear(input_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.embed_dropout = nn.Dropout(dropout)

        # Preserve the historical effective attention dropout for controlled ablations.
        # attention_dropout is a legacy recorded-only argument; dropout controls attention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_pool = AttentionPool(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _apply_token_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout <= 0:
            return x
        keep_prob = max(1e-6, 1.0 - self.token_dropout)
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < keep_prob).float()
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = self.input_norm(x)
        x = self.input_proj(x)
        x = self._apply_token_dropout(x)

        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.embed_dropout(x)
        x = self.encoder(x)
        x = self.final_norm(x)

        cls_embed = x[:, 0, :]
        tokens = x[:, 1:, :]
        mean_embed = tokens.mean(dim=1)
        max_embed = tokens.amax(dim=1)
        attn_embed = self.attn_pool(tokens)
        pooled = torch.cat([cls_embed, mean_embed, max_embed, attn_embed], dim=1)
        return self.head(pooled)


class ExoFormer(nn.Module):
    def __init__(
        self,
        *,
        global_seq_len: int,
        local_seq_len: int,
        scalar_features: int,
        d_model: int,
        num_heads: int,
        global_depth: int,
        local_depth: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        token_dropout: float,
        scalar_dropout: float,
        use_input_norm: bool = True,
        use_aux_branch: bool = False,
        aux_features: int = 0,
        aux_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if use_aux_branch and aux_features <= 0:
            raise ValueError("An auxiliary branch requires a positive feature count")
        self.global_branch = TransformerBranch(
            input_channels=3,
            seq_len=global_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            depth=global_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            token_dropout=token_dropout,
            use_input_norm=use_input_norm,
        )
        self.local_branch = TransformerBranch(
            input_channels=3,
            seq_len=local_seq_len,
            d_model=d_model,
            num_heads=num_heads,
            depth=local_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            token_dropout=token_dropout,
            use_input_norm=use_input_norm,
        )
        self.scalar_branch = ScalarEncoder(scalar_features, dropout=scalar_dropout)
        self.cross_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )
        self.aux_branch = ScalarEncoder(aux_features, dropout=aux_dropout) if use_aux_branch else None
        fusion_dim = 128 + 128 + 128 + 128 + 128 + 64 + (64 if use_aux_branch else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(
        self,
        x_global: torch.Tensor,
        x_local: torch.Tensor,
        x_scalar: torch.Tensor,
        x_flat: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del x_flat
        if self.aux_branch is not None and x_aux is None:
            raise ValueError("X_aux is required when the auxiliary branch is enabled")
        if self.aux_branch is None and x_aux is not None:
            raise ValueError("X_aux was supplied while the auxiliary branch is disabled")
        global_embed = self.global_branch(x_global)
        local_embed = self.local_branch(x_local)
        scalar_embed = self.scalar_branch(x_scalar)
        gate = self.cross_gate(torch.cat([global_embed, local_embed], dim=1))
        mixed_embed = gate * global_embed + (1.0 - gate) * local_embed
        parts = [
                global_embed,
                local_embed,
                mixed_embed,
                torch.abs(global_embed - local_embed),
                global_embed * local_embed,
                scalar_embed,
            ]
        if self.aux_branch is not None:
            parts.append(self.aux_branch(x_aux))
        fused = torch.cat(parts, dim=1)
        return self.classifier(fused).squeeze(1)


def train_model(cfg: TrainConfig) -> Path:
    dataset_name = Path(cfg.dataset_path).stem
    run_name = f"{dataset_name}_transformer_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", asdict(cfg))
    with tracking_run(cfg, run_dir) as tracker:
        return _train_model(cfg, run_dir, tracker)


def _train_model(cfg: TrainConfig, run_dir: Path, tracker: Any) -> Path:
    set_seed(cfg.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = detect_device(cfg.device)
    arrays, manifest = load_training_data(cfg)

    run_name = run_dir.name

    train_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["train_idx"],
        x_aux=arrays.get("x_aux"),
    )
    val_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["val_idx"],
        x_aux=arrays.get("x_aux"),
    )
    test_ds = KeplerTensorDataset(
        arrays["x_global"],
        arrays["x_local"],
        arrays["x_scalar"],
        arrays["x_flat"],
        arrays["y"],
        arrays["test_idx"],
        x_aux=arrays.get("x_aux"),
    ) if cfg.evaluate_test else None

    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, device=device)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, device=device)
    test_loader = (make_dataloader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, device=device)
                   if test_ds is not None else None)

    model_config = dict(
        global_seq_len=int(arrays["x_global"].shape[1]),
        local_seq_len=int(arrays["x_local"].shape[1]),
        scalar_features=int(arrays["x_scalar"].shape[1]),
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        global_depth=cfg.global_depth,
        local_depth=cfg.local_depth,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        token_dropout=cfg.token_dropout,
        scalar_dropout=cfg.scalar_dropout,
        use_input_norm=cfg.use_input_norm, use_aux_branch=cfg.use_aux_branch,
        aux_features=int(arrays["x_aux"].shape[1]) if cfg.use_aux_branch else 0,
        aux_dropout=cfg.aux_dropout,
    )
    auxiliary_preprocessing = json.loads(str(arrays["aux_preprocessing_json"].item())) if cfg.use_aux_branch else None
    write_json(run_dir / "model_config.json", model_config)
    if auxiliary_preprocessing is not None:
        write_json(run_dir / "aux_preprocessing.json", auxiliary_preprocessing)
    if tracker is not None:
        tracker.config.update({"resolved_device": str(device), "effective_attention_dropout": cfg.dropout,
                               "aux_feature_count": model_config["aux_features"],
                               "aux_feature_names": arrays["aux_feature_names"].tolist() if cfg.use_aux_branch else [],
                               "train_size": len(train_ds), "val_size": len(val_ds)})
    model = ExoFormer(**model_config).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    y_train = arrays["y"][arrays["train_idx"]]
    pos_count = max(1, int(y_train.sum()))
    neg_count = max(1, int(len(y_train) - pos_count))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler: Optional[GradScaler] = GradScaler(enabled=True) if device.type == "cuda" and cfg.use_amp else None

    history: List[Dict[str, Any]] = []
    best_val_pr_auc = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
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
        val_threshold = build_threshold(val_eval["targets"], val_eval["probs"], cfg.threshold_grid_size)
        val_metrics = compute_metrics(val_eval["targets"], val_eval["probs"], threshold=val_threshold) | {
            "loss": val_eval["metrics"]["loss"],
            "threshold": val_threshold,
        }

        history.append(
            {
                "epoch": epoch,
                "learning_rate": epoch_learning_rate,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_pr_auc": train_metrics["pr_auc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "train_f1": train_metrics["f1"],
                "train_threshold": train_threshold,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_f1": val_metrics["f1"],
                "val_threshold": val_threshold,
            }
        )
        if tracker is not None:
            tracker.log({key.replace("train_", "train/", 1).replace("val_", "val/", 1): value
                         for key, value in history[-1].items()}, step=epoch)

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
                    "val_threshold": val_threshold,
                    "model_config": model_config,
                    "aux_preprocessing": auxiliary_preprocessing,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_pr_auc={train_metrics['pr_auc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} lr={epoch_learning_rate:.2e}"
        )

        if epochs_without_improvement >= cfg.patience:
            print(f"Early stopping at epoch {epoch} after {cfg.patience} epochs without validation PR-AUC improvement.")
            break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    best_threshold = float(checkpoint.get("val_threshold", 0.5))

    train_eval = evaluate_split(model, train_loader, loss_fn, device=device, threshold=best_threshold)
    val_eval = evaluate_split(model, val_loader, loss_fn, device=device, threshold=best_threshold)
    test_eval = (evaluate_split(model, test_loader, loss_fn, device=device, threshold=best_threshold)
                 if test_loader is not None else None)

    save_predictions_csv(run_dir / "val_predictions.csv", manifest, val_eval["indices"], val_eval["probs"], best_threshold)
    if test_eval is not None:
        save_predictions_csv(run_dir / "test_predictions.csv", manifest, test_eval["indices"], test_eval["probs"], best_threshold)

    summary = {
        "run_name": run_name,
        "dataset_path": cfg.dataset_path,
        "splits_path": cfg.splits_path,
        "manifest_path": cfg.manifest_path,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "train_size": int(len(train_ds)),
        "val_size": int(len(val_ds)),
        "test_size": int(len(arrays["test_idx"])),
        "test_evaluated": test_eval is not None,
        "use_input_norm": cfg.use_input_norm,
        "use_aux_branch": cfg.use_aux_branch,
        "aux_feature_count": model_config["aux_features"],
        "aux_feature_names": arrays["aux_feature_names"].tolist() if cfg.use_aux_branch else [],
        "effective_attention_dropout": cfg.dropout,
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "global_depth": cfg.global_depth,
        "local_depth": cfg.local_depth,
    }
    if test_eval is not None:
        summary["test_metrics"] = test_eval["metrics"]
    if tracker is not None:
        final = {"best_epoch": best_epoch, "best_threshold": best_threshold}
        evaluations = [("final_train", train_eval), ("final_val", val_eval)]
        if test_eval is not None:
            evaluations.append(("test", test_eval))
        for prefix, evaluation in evaluations:
            final.update({f"{prefix}/{key}": value for key, value in evaluation["metrics"].items()})
        tracker.log(final, step=len(history)+1)
        tracker.summary.update(final)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "config.json", asdict(cfg))

    print("\nBest validation/test summary")
    print(json.dumps(summary, indent=2))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modern transformer-based Kepler classifier on processed dataset artifacts.")
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--splits_path", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attention_dropout", type=float, default=0.10,
                        help="Legacy recorded-only value; effective attention dropout equals --dropout.")
    parser.add_argument("--no_input_norm", action="store_false", dest="use_input_norm",
                        help="Bypass only the initial LayerNorm(3), preserving all other model operations.")
    parser.add_argument("--aux_branch", action="store_true")
    parser.add_argument("--aux_dropout", type=float, default=0.10)
    parser.add_argument("--skip_test", action="store_false", dest="evaluate_test",
                        help="Train/select using train and validation only; save no test predictions or metrics.")
    parser.add_argument("--wandb_mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb_project", default="kepler-practical")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Use float32 training; useful on GPUs where autocast is slower.")
    parser.add_argument("--token_dropout", type=float, default=0.05)
    parser.add_argument("--scalar_dropout", type=float, default=0.10)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--global_depth", type=int, default=4)
    parser.add_argument("--local_depth", type=int, default=3)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--view_noise_std", type=float, default=0.0)
    parser.add_argument("--scalar_noise_std", type=float, default=0.0)
    parser.add_argument("--flat_dropout_prob", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    if args.aux_branch and not all((args.dataset_path, args.splits_path, args.manifest_path)):
        raise ValueError("--aux_branch requires explicit dataset, split, and manifest paths from the same auxiliary build")
    base_dir = Path(args.base_dir)
    processed_dir = base_dir / "data" / "processed" / "kepler" / "v0"
    dataset_path = Path(args.dataset_path) if args.dataset_path else auto_find_path(processed_dir, "dataset_kepler_*.npz")
    splits_path = Path(args.splits_path) if args.splits_path else auto_find_path(processed_dir, "splits_kepler_*.npz")
    manifest_path = Path(args.manifest_path) if args.manifest_path else infer_manifest_path(processed_dir, dataset_path)
    output_root = Path(args.output_root) if args.output_root else base_dir / "outputs" / "transformer"
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
        attention_dropout=args.attention_dropout,
        token_dropout=args.token_dropout,
        scalar_dropout=args.scalar_dropout,
        num_workers=args.num_workers,
        prepare_version=prepare_version,
        compile_model=args.compile_model,
        device=args.device,
        gradient_clip_norm=args.gradient_clip_norm,
        d_model=args.d_model,
        num_heads=args.num_heads,
        global_depth=args.global_depth,
        local_depth=args.local_depth,
        mlp_ratio=args.mlp_ratio,
        label_smoothing=args.label_smoothing,
        view_noise_std=args.view_noise_std,
        scalar_noise_std=args.scalar_noise_std,
        flat_dropout_prob=args.flat_dropout_prob,
        use_input_norm=args.use_input_norm, use_aux_branch=args.aux_branch, aux_dropout=args.aux_dropout,
        evaluate_test=args.evaluate_test, wandb_mode=args.wandb_mode, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, wandb_group=args.wandb_group,
        use_amp=args.use_amp,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_dir = train_model(cfg)
    print(f"\nSaved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
