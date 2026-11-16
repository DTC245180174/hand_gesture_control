"""
BƯỚC 5 (AIR-INK): HUẤN LUYỆN MODEL 3 (AIRINKRESCNN - 9 LỚP HÌNH HỌC) TỪ ĐẦU
==============================================================================
Huấn luyện mạng CNN Residual + Quadrant Spatial Pooling (`AirInkResCNN`, 9 lớp hình học)
từ trọng số khởi tạo ngẫu nhiên trên bộ dữ liệu 64x64 đã chuẩn hoá:
- Stratified Balanced Cross-Entropy + Label Smoothing (0.05) (không dùng Mixup ảnh).
- AdamW Optimizer + CosineAnnealingLR + Gradient Clipping (max_norm=1.0).
- Early Stopping theo Val Macro-F1 + Lưu tại `outputs/models/best_air_ink_model.pt`.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.config import (
    AIR_INK_CLASSES,
    AIR_INK_IMG_SIZE,
    AIR_INK_PROCESSED_DIR,
    AIR_INK_SHAPE_IDS,
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    NUM_AIR_INK_CLASSES,
)
from step4_model_selection.models import build_air_ink_model
from step5_training.air_ink_dataset import AirInkDataset, compute_air_ink_class_weights
from step5_training.train import evaluate_epoch, set_seed, train_one_epoch


def _save_learning_curves(history: Dict[str, list], best_epoch: int, best_val_acc: float) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    ep_range = np.arange(1, len(history["train_loss"]) + 1)

    ax1.plot(ep_range, history["train_loss"], label="Train Loss (Aug)", color="#2980b9", linewidth=2)
    ax1.plot(ep_range, history["val_loss"], label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.axvline(best_epoch, color="#27ae60", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Model 3 (AirInkResCNN): Cross-Entropy Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep_range, history["train_acc"], label="Train Acc (%)", color="#2980b9", linewidth=2)
    ax2.plot(ep_range, history["val_acc"], label="Val Acc 9-cls (%)", color="#e67e22", linewidth=2)
    ax2.plot(ep_range, history["val_f1"], label="Val Macro-F1 (%)", color="#27ae60", linewidth=2, linestyle="-.")
    ax2.axvline(best_epoch, color="#27ae60", linestyle="--")
    ax2.set_title(f"9-Shape Accuracy & Macro-F1 (Best Val Acc: {best_val_acc:.2f}%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig_path = FIGURES_DIR / "fig9_air_ink_learning_curves.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def train_air_ink_model(
    epochs: int = 14,
    batch_size: int = 160,
    lr: float = 2.0e-3,
    weight_decay: float = 5e-4,
    dropout: float = 0.25,
    patience: int = 6,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Huấn luyện từ đầu Model 3 (`AirInkResCNN`) trên dữ liệu 9 lớp Hình học 64x64,
    lưu trọng số tốt nhất tại `outputs/models/best_air_ink_model.pt`.
    """
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(AIR_INK_PROCESSED_DIR / "X_train.npy")
    y_train = np.load(AIR_INK_PROCESSED_DIR / "y_train.npy")
    X_val = np.load(AIR_INK_PROCESSED_DIR / "X_val.npy")
    y_val = np.load(AIR_INK_PROCESSED_DIR / "y_val.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = AirInkDataset(X_train, y_train, augment=True, class_names=AIR_INK_CLASSES)
    val_ds = AirInkDataset(X_val, y_val, augment=False, class_names=AIR_INK_CLASSES)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_air_ink_class_weights(y_train, NUM_AIR_INK_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = build_air_ink_model(
        in_channels=1, num_classes=NUM_AIR_INK_CLASSES, dropout=dropout
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 84)
    print(f"BƯỚC 5 (AIR-INK): HUẤN LUYỆN MODEL 3 (AIRINKRESCNN - {NUM_AIR_INK_CLASSES} LỚP HÌNH HỌC)")
    print(f"  - Thiết bị       : {device} (CPU threads={torch.get_num_threads()})")
    print(f"  - Tham số Model  : {num_params:,d} (~{num_params * 4 / 1024:.1f} KB)")
    print(f"  - Tập Train      : {X_train.shape} ảnh 64x64 (Augment=True, Batch={batch_size})")
    print(f"  - Tập Validation : {X_val.shape} ảnh 64x64")
    print(f"  - Regularization : Dropout={dropout} | WeightDecay={weight_decay} | LabelSmooth=0.05")
    print("=" * 84)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_metrics: Dict[str, float] = {}
    best_model_path = MODELS_DIR / "best_air_ink_model.pt"

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "epoch_sec": [],
    }

    t_start = time.perf_counter()
    print(
        f"{'Epoch':>5} | {'LR':>8} | {'TrLoss':>7} | {'TrAcc':>7} | "
        f"{'ValLoss':>7} | {'ValAcc':>7} | {'ValF1':>7} | {'Time':>6} | Status"
    )
    print("-" * 84)

    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha=0.0
        )
        va_loss, va_acc, va_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_sec = time.perf_counter() - t_ep

        history["train_loss"].append(round(tr_loss, 5))
        history["train_acc"].append(round(tr_acc * 100.0, 2))
        history["train_f1"].append(round(tr_f1 * 100.0, 2))
        history["val_loss"].append(round(va_loss, 5))
        history["val_acc"].append(round(va_acc * 100.0, 2))
        history["val_f1"].append(round(va_f1 * 100.0, 2))
        history["epoch_sec"].append(round(ep_sec, 1))

        status_str = ""
        is_higher_f1 = va_f1 > best_val_f1 + 1e-6
        is_tied_lower_loss = abs(va_f1 - best_val_f1) <= 1e-6 and va_loss < best_val_loss - 1e-4
        if is_higher_f1 or is_tied_lower_loss:
            best_val_f1 = va_f1
            best_val_loss = va_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "val_loss": round(va_loss, 4),
                "train_acc": round(tr_acc * 100.0, 2),
                "train_f1_macro": round(tr_f1 * 100.0, 2),
                "val_acc": round(va_acc * 100.0, 2),
                "val_f1_macro": round(va_f1 * 100.0, 2),
                "acc_gap_train_minus_val": round((tr_acc - va_acc) * 100.0, 2),
                "f1_gap_train_minus_val": round((tr_f1 - va_f1) * 100.0, 2),
            }
            torch.save(
                {
                    "arch_name": "air_ink",
                    "model_class": "AirInkResCNN",
                    "in_channels": 1,
                    "img_size": AIR_INK_IMG_SIZE,
                    "num_classes": NUM_AIR_INK_CLASSES,
                    "class_names": AIR_INK_CLASSES,
                    "shape_ids": list(AIR_INK_SHAPE_IDS),
                    "dropout": dropout,
                    "model_state_dict": model.state_dict(),
                    "best_epoch": epoch,
                    "best_metrics": best_metrics,
                },
                best_model_path,
            )
            status_str = "* LƯU BEST AIR-INK"
        else:
            epochs_no_improve += 1

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>5d} | {curr_lr:>8.1e} | {tr_loss:>7.4f} | {tr_acc*100:>6.2f}% | "
            f"{va_loss:>7.4f} | {va_acc*100:>6.2f}% | {va_f1*100:>6.2f}% | "
            f"{ep_sec:>5.1f}s | {status_str}",
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Dừng sớm tại Epoch {epoch} (không cải thiện sau {patience} epochs).")
            break

    total_time = time.perf_counter() - t_start
    fig_path = _save_learning_curves(history, best_epoch, best_metrics["val_acc"])
    report_path = DIAGNOSTICS_DIR / "air_ink_training_history.json"
    summary = {
        "model_class": "AirInkResCNN",
        "num_classes": NUM_AIR_INK_CLASSES,
        "device": str(device),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "total_epochs_run": len(history["train_loss"]),
        "training_time_sec": round(total_time, 1),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "class_weights_min_max": [
            round(float(class_weights.min().item()), 4),
            round(float(class_weights.max().item()), 4),
        ],
        "history": history,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("-" * 84)
    print(f"[HOÀN TẤT HUẤN LUYỆN MODEL 3] Tổng thời gian: {total_time:.1f}s")
    print(
        f"[BEST @ Epoch {best_epoch}] Val Acc ({NUM_AIR_INK_CLASSES} lớp hình) = {best_metrics['val_acc']:.2f}% | "
        f"Val Macro-F1 = {best_metrics['val_f1_macro']:.2f}% | "
        f"Gap (Train-Val Acc) = {best_metrics['acc_gap_train_minus_val']:+.2f}%"
    )
    print(f"[MODEL SAVED]  {best_model_path}")
    print(f"[FIGURE SAVED] {fig_path}")
    print(f"[REPORT SAVED] {report_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_air_ink_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )
