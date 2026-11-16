"""
BƯỚC 5: HUẤN LUYỆN MÔ HÌNH HỌC SÂU CHUỖI THỜI GIAN TỪ ĐẦU (TRAINING FROM SCRATCH)
=================================================================================
Huấn luyện mô hình từ trọng số ngẫu nhiên trên dữ liệu đã làm sạch:
- Balanced Cross-Entropy Loss + Label Smoothing (0.05)
- AdamW Optimizer + ReduceLROnPlateau Scheduler
- Early Stopping (patience=20)
- Lưu mô hình tốt nhất vào `outputs/models/best_gesture_model.pt`.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common.config import (
    CLASS_NAMES,
    DIAGNOSTICS_DIR,
    FEATURE_DIM,
    FIGURES_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_POSE_CLASSES,
    POSE_CLASS_NAMES,
    POSE_FEATURE_DIM,
    PROCESSED_DIR,
    SEQ_LEN,
)
from step4_model_selection.models import build_model, build_pose_model
from step5_training.dataset import HandGestureSequenceDataset
from step5_training.pose_dataset import (
    StaticPoseDataset,
    extract_single_frame_pose_samples,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y_train: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    weights = np.sqrt(total / (counts.astype(np.float32) * num_classes + 1e-6))
    # Giảm nhẹ trọng số lớp 0 (idle) để mô hình không bị thiên vị đoán nhầm các lệnh hành động về idle
    weights[0] *= 0.82
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float = 0.15,
) -> Tuple[float, float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Áp dụng Sequence Mixup nhẹ (xác suất 25%) và so khớp nhãn với thành phần chiếm ưu thế (lam >= 0.5)
        if mixup_alpha > 0.0 and len(y_batch) > 1 and np.random.rand() < 0.25:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(len(y_batch), device=device)
            X_mix = lam * X_batch + (1.0 - lam) * X_batch[perm]
            logits = model(X_mix)
            loss = lam * criterion(logits, y_batch) + (1.0 - lam) * criterion(logits, y_batch[perm])
            dominant_targets = y_batch if lam >= 0.5 else y_batch[perm]
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            dominant_targets = y_batch

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == dominant_targets).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(dominant_targets.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


def run_training_pipeline(
    arch: str = "hybrid",
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 8e-4,
    dropout: float = 0.28,
    patience: int = 20,
    seed: int = 42,
) -> None:
    set_seed(seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HandGestureSequenceDataset(X_train, y_train, augment=True)
    val_dataset = HandGestureSequenceDataset(X_val, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weights(y_train, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = build_model(arch, input_dim=FEATURE_DIM, num_classes=NUM_CLASSES, dropout=dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"BƯỚC 5: HUẤN LUYỆN MÔ HÌNH HỌC SÂU ({arch.upper()}) TỪ ĐẦU (CÂN BẰNG CHỐNG OVERFITTING)")
    print(f"  - Thiết bị        : {device}")
    print(f"  - Số tham số Model: {num_params:,d} tham số (~{num_params * 4 / 1024:.1f} KB)")
    print(f"  - Tập Train       : {X_train.shape} chuỗi độc lập")
    print(f"  - Tập Validation  : {X_val.shape} chuỗi (6 người độc lập trong IPN HandS)")
    print(f"  - Regularization  : Dropout={dropout} | WeightDecay={weight_decay} | Mixup=0.20 | LabelSmooth=0.05")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_metrics = {}
    best_model_path = MODELS_DIR / "best_gesture_model.pt"

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }

    t_start = time.perf_counter()
    print(f"{'Epoch':>5} | {'LR':>8} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>8} | Status")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.20)
        va_loss, va_acc, va_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(va_f1)

        history["train_loss"].append(round(tr_loss, 5))
        history["train_acc"].append(round(tr_acc * 100, 2))
        history["train_f1"].append(round(tr_f1 * 100, 2))
        history["val_loss"].append(round(va_loss, 5))
        history["val_acc"].append(round(va_acc * 100, 2))
        history["val_f1"].append(round(va_f1 * 100, 2))

        status_str = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": round(va_loss, 4),
                "val_acc": round(va_acc * 100, 2),
                "val_f1_macro": round(va_f1 * 100, 2),
                "train_acc": round(tr_acc * 100, 2),
                "train_f1_macro": round(tr_f1 * 100, 2),
            }
            torch.save({
                "arch_name": arch,
                "input_dim": FEATURE_DIM,
                "seq_len": SEQ_LEN,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "model_state_dict": model.state_dict(),
                "best_epoch": epoch,
                "best_metrics": best_metrics,
            }, best_model_path)
            status_str = "* LƯU BEST MODEL"
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or status_str or epoch == epochs:
            curr_lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>5d} | {curr_lr:>8.1e} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | {va_loss:>9.4f} | {va_acc*100:>7.2f}% | {va_f1*100:>7.2f}% | {status_str}")

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Dừng sớm tại Epoch {epoch} (không cải thiện sau {patience} epochs).")
            break

    total_time = time.perf_counter() - t_start
    print("-" * 80)
    print(f"[HOÀN TẤT HUẤN LUYỆN] Tổng thời gian: {total_time:.1f} giây")
    print(f"[KẾT QUẢ TỐT NHẤT @ Epoch {best_epoch}] Val Acc = {best_metrics['val_acc']:.2f}% | Val Macro-F1 = {best_metrics['val_f1_macro']:.2f}%")
    print(f"[MODEL SAVED] Đã lưu trọng số tại: {best_model_path}")

    # Đánh giá trên tập Webcam sạch (nếu có)
    x_web_path = PROCESSED_DIR / "X_webcam_clean.npy"
    y_web_path = PROCESSED_DIR / "y_webcam_clean.npy"
    if x_web_path.exists() and y_web_path.exists():
        X_web = np.load(x_web_path)
        y_web = np.load(y_web_path)
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        web_loader = DataLoader(HandGestureSequenceDataset(X_web, y_web, augment=False), batch_size=batch_size, shuffle=False)
        _, web_acc, web_f1 = evaluate_epoch(model, web_loader, criterion, device)
        best_metrics["webcam_clean_acc"] = round(web_acc * 100, 2)
        best_metrics["webcam_clean_f1"] = round(web_f1 * 100, 2)
        checkpoint["best_metrics"] = best_metrics
        torch.save(checkpoint, best_model_path)
        print(f"[KIỂM ĐỊNH ĐỘC LẬP 20% MẪU WEBCAM VAL ({len(y_web)} mẫu chưa từng học)] Webcam Val Acc = {web_acc*100:.2f}% | Webcam Val F1 = {web_f1*100:.2f}%")

    # Vẽ biểu đồ đường cong huấn luyện fig5_learning_curves.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    ep_range = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(ep_range, history["train_loss"], label="Train Loss", color="#2980b9", linewidth=2)
    ax1.plot(ep_range, history["val_loss"], label="Val Loss", color="#e74c3c", linewidth=2)
    ax1.axvline(best_epoch, color="#27ae60", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Cross-Entropy Loss (Train vs Validation)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep_range, history["train_acc"], label="Train Acc (%)", color="#2980b9", linewidth=2)
    ax2.plot(ep_range, history["val_acc"], label="Val Acc (%)", color="#e67e22", linewidth=2)
    ax2.plot(ep_range, history["val_f1"], label="Val Macro-F1 (%)", color="#27ae60", linewidth=2, linestyle="-.")
    ax2.axvline(best_epoch, color="#27ae60", linestyle="--")
    ax2.set_title(f"Accuracy & Macro-F1 (Best Val Acc: {best_metrics['val_acc']:.2f}%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig5_path = FIGURES_DIR / "fig5_learning_curves.png"
    fig.tight_layout()
    fig.savefig(fig5_path)
    plt.close(fig)
    print(f"[FIGURE SAVED] Đã xuất biểu đồ huấn luyện tại: {fig5_path}")

    with open(DIAGNOSTICS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "arch_name": arch,
            "device": str(device),
            "total_epochs_run": len(history["train_loss"]),
            "training_time_sec": round(total_time, 1),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "class_weights": [round(float(w), 4) for w in class_weights.cpu().numpy()],
            "history": history,
        }, f, indent=2, ensure_ascii=False)
    print(f"[REPORT SAVED] Đã lưu lịch sử huấn luyện tại: {DIAGNOSTICS_DIR / 'training_history.json'}")

    # Huấn luyện Model 1 (StaticPoseNet - 1 frame) cho kiến trúc Dual-Model (Option B)
    train_static_pose_model(device=device, seed=seed)


def train_static_pose_model(
    device: torch.device,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 4e-4,
    dropout: float = 0.15,
    patience: int = 18,
    seed: int = 42,
) -> None:
    """
    Huấn luyện từ đầu Model 1 (StaticPoseNet - 1 frame) nhận diện 3 tư thế tĩnh:
    [0: other_pose, 1: point_1f, 2: point_2f], lưu tại outputs/models/best_pose_model.pt.
    """
    set_seed(seed)
    raw_train = np.load(PROCESSED_DIR / "raw30_train.npy")
    y_train_seq = np.load(PROCESSED_DIR / "y_train.npy")
    raw_val = np.load(PROCESSED_DIR / "raw30_val.npy")
    y_val_seq = np.load(PROCESSED_DIR / "y_val.npy")

    f_tr, y_tr = extract_single_frame_pose_samples(raw_train, y_train_seq, is_train=True)
    f_va, y_va = extract_single_frame_pose_samples(raw_val, y_val_seq, is_train=False)

    train_ds = StaticPoseDataset(f_tr, y_tr, augment=True, base_aspect=4.0 / 3.0)
    val_ds = StaticPoseDataset(f_va, y_va, augment=False, base_aspect=4.0 / 3.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    counts = np.bincount(y_tr, minlength=NUM_POSE_CLASSES)
    weights = np.power(len(y_tr) / (counts.astype(np.float32) * NUM_POSE_CLASSES + 1e-6), 0.68)
    weights = weights / weights.mean()
    class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)

    pose_model = build_pose_model(
        in_channels=POSE_FEATURE_DIM, num_classes=NUM_POSE_CLASSES, dropout=dropout
    ).to(device)
    num_params = sum(p.numel() for p in pose_model.parameters() if p.requires_grad)

    print("\n" + "=" * 80)
    print("BƯỚC 5B: HUẤN LUYỆN MODEL 1 (STATICPOSENET - 1 FRAME TỪ ĐẦU - OPTION B)")
    print(f"  - Tham số Model : {num_params:,d} (~{num_params * 4 / 1024:.1f} KB)")
    print(f"  - Khung hình Tr : {len(y_tr)} frames | Phân bố: {counts.tolist()}")
    print(f"  - Khung hình Val: {len(y_va)} frames (IPN Val độc lập)")
    print("=" * 80)

    optimizer = torch.optim.AdamW(pose_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    best_metrics = {}
    best_pose_path = MODELS_DIR / "best_pose_model.pt"

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        _tr_loss, tr_acc, _tr_f1 = train_one_epoch(
            pose_model, train_loader, criterion, optimizer, device, mixup_alpha=0.0
        )
        va_loss, va_acc, va_f1 = evaluate_epoch(pose_model, val_loader, criterion, device)
        scheduler.step()

        improved = va_f1 > best_val_f1
        if improved:
            best_val_f1 = va_f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": round(va_loss, 4),
                "val_acc": round(va_acc * 100, 2),
                "val_f1_macro": round(va_f1 * 100, 2),
                "train_acc": round(tr_acc * 100, 2),
            }
            torch.save({
                "arch_name": "static_pose",
                "in_channels": POSE_FEATURE_DIM,
                "num_classes": NUM_POSE_CLASSES,
                "class_names": POSE_CLASS_NAMES,
                "base_aspect": 4.0 / 3.0,
                "model_state_dict": pose_model.state_dict(),
                "best_epoch": epoch,
                "best_metrics": best_metrics,
            }, best_pose_path)
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved or epoch == epochs:
            mark = "* LUU BEST POSE" if improved else ""
            print(
                f"Ep {epoch:>2d} | TrAcc: {tr_acc*100:>6.2f}% | "
                f"ValAcc: {va_acc*100:>6.2f}% | ValF1: {va_f1*100:>6.2f}% | {mark}"
            )

        if epochs_no_improve >= patience:
            print(f"[EARLY STOPPING] StaticPoseNet dừng tại Epoch {epoch}.")
            break

    elapsed = time.perf_counter() - t0
    print("-" * 80)
    print(f"[HOÀN TẤT BƯỚC 5B] Thời gian: {elapsed:.1f}s | Best @ Ep {best_epoch}: Val Acc = {best_metrics['val_acc']:.2f}% | Val F1 = {best_metrics['val_f1_macro']:.2f}%")

    raw_web_path = PROCESSED_DIR / "raw30_webcam_clean.npy"
    y_web_path = PROCESSED_DIR / "y_webcam_clean.npy"
    if raw_web_path.exists() and y_web_path.exists():
        raw_web = np.load(raw_web_path)
        y_web_seq = np.load(y_web_path)
        f_web, y_web_pose = extract_single_frame_pose_samples(raw_web, y_web_seq, is_train=False)
        if len(y_web_pose) > 0:
            ckpt_p = torch.load(best_pose_path, map_location=device, weights_only=False)
            pose_model.load_state_dict(ckpt_p["model_state_dict"])
            web_loader = DataLoader(
                StaticPoseDataset(f_web, y_web_pose, augment=False, base_aspect=16.0 / 9.0),
                batch_size=batch_size,
                shuffle=False,
            )
            _, web_acc, web_f1 = evaluate_epoch(pose_model, web_loader, criterion, device)
            best_metrics["webcam_clean_acc"] = round(web_acc * 100, 2)
            best_metrics["webcam_clean_f1"] = round(web_f1 * 100, 2)
            ckpt_p["best_metrics"] = best_metrics
            torch.save(ckpt_p, best_pose_path)
            print(f"[WEBCAM POSE VAL ({len(y_web_pose)} frames)] Acc = {web_acc*100:.2f}% | F1 = {web_f1*100:.2f}%")
    print(f"[POSE MODEL SAVED] {best_pose_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="hybrid")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pose-only", action="store_true")
    args = parser.parse_args()
    if args.pose_only:
        train_static_pose_model(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        run_training_pipeline(arch=args.arch, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

