"""
BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH ĐỘC LẬP TRÊN TẬP TEST (MODEL EVALUATION & SAFETY AUDIT)
================================================================================
Đánh giá mô hình đã huấn luyện (outputs/models/best_gesture_model.pt) trên tập Test độc lập:
1. Báo cáo phân loại chi tiết 8 lớp (Precision, Recall, F1-Score).
2. Ma trận nhầm lẫn 8x8 (Confusion Matrix).
3. Kiểm định cổng an toàn Confidence Threshold (Idle False Positive Rate).
4. Đo tốc độ suy luận CPU (Latency ms/frame).
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common.config import (
    CLASS_NAMES,
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    POSE_CLASS_NAMES,
    PROCESSED_DIR,
)
from step4_model_selection.models import build_model, build_pose_model
from step5_training.pose_dataset import (
    StaticPoseDataset,
    extract_single_frame_pose_samples,
)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, X_test: np.ndarray, device: torch.device
) -> Dict[str, np.ndarray]:
    model.eval()
    x_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)

    if hasattr(model, "attention"):
        logits, attn_weights = model(x_tensor, return_attn=True)
        attn_np = attn_weights.cpu().numpy()
    else:
        logits = model(x_tensor)
        attn_np = None

    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)
    confidences = probs.max(axis=-1)
    return {
        "probs": probs,
        "preds": preds,
        "confidences": confidences,
        "attn_weights": attn_np,
    }


def evaluate_confidence_thresholds(
    y_true: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    thresholds: List[float] = [0.0, 0.70, 0.80, 0.85, 0.90],
) -> Dict[str, Dict[str, float]]:
    results = {}
    num_idle = int(np.sum(y_true == 0))

    for th in thresholds:
        gated_preds = np.where(confidences >= th, preds, 0)
        overall_acc = float(accuracy_score(y_true, gated_preds)) * 100.0
        macro_f1 = float(f1_score(y_true, gated_preds, average="macro", zero_division=0)) * 100.0
        false_triggers_on_idle = int(np.sum((y_true == 0) & (gated_preds != 0)))
        idle_fpr = (false_triggers_on_idle / max(num_idle, 1)) * 100.0

        active_mask = gated_preds != 0
        active_precision = float(np.mean(y_true[active_mask] == gated_preds[active_mask])) * 100.0 if np.any(active_mask) else 100.0

        results[f"{th:.2f}"] = {
            "threshold": th,
            "gated_accuracy_pct": round(overall_acc, 2),
            "gated_macro_f1_pct": round(macro_f1, 2),
            "idle_false_triggers_count": false_triggers_on_idle,
            "idle_total_count": num_idle,
            "idle_false_positive_rate_pct": round(idle_fpr, 2),
            "active_command_precision_pct": round(active_precision, 2),
        }
    return results


def run_evaluation_pipeline(checkpoint_path: Path = MODELS_DIR / "best_gesture_model.pt") -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch_name = ckpt.get("arch_name", "hybrid")

    model = build_model(arch_name, input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 80)
    print(f"BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH ĐỘC LẬP TRÊN TẬP TEST ({len(y_test)} MẪU)")
    print(f"  - Model: {arch_name.upper()} @ Epoch {ckpt.get('best_epoch', 0)}")
    print("=" * 80)

    res = run_inference(model, X_test, device)
    preds = res["preds"]
    confidences = res["confidences"]

    acc = float(accuracy_score(y_test, preds)) * 100.0
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0)) * 100.0
    weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0)) * 100.0

    print(f"\n1. TỔNG HỢP:")
    print(f"  - Test Accuracy : {acc:.2f}% ({int((preds == y_test).sum())}/{len(y_test)} mẫu đúng)")
    print(f"  - Test Macro-F1 : {macro_f1:.2f}%")
    print(f"  - Test Weighted : {weighted_f1:.2f}%")

    print("\n2. BẢNG PHÂN LOẠI CHI TIẾT:")
    report_dict = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=2, output_dict=True)
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=2))

    # Benchmark độ trễ
    latencies = []
    x_single = torch.from_numpy(X_test[:1].astype(np.float32)).to(device)
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model(x_single)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency = float(np.median(latencies[5:]))
    print(f"[LATENCY BENCHMARK] Thời gian suy luận 1 cửa sổ trượt: {avg_latency:.3f} ms (~{int(1000/avg_latency)} FPS)")

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, preds, labels=np.arange(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix (Test Acc: {acc:.2f}%)", fontweight="bold")
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_confusion_matrix.png")
    plt.close(fig)

    threshold_results = evaluate_confidence_thresholds(y_test, preds, confidences)

    pose_eval_report = {}
    pose_ckpt_path = checkpoint_path.parent / "best_pose_model.pt"
    raw_test_path = PROCESSED_DIR / "raw30_test.npy"
    if pose_ckpt_path.exists() and raw_test_path.exists():
        raw_test = np.load(raw_test_path)
        f_pose_test, y_pose_test = extract_single_frame_pose_samples(raw_test, y_test, is_train=False)
        pose_ckpt = torch.load(pose_ckpt_path, map_location=device, weights_only=False)
        pose_model = build_pose_model(
            in_channels=pose_ckpt["in_channels"], num_classes=pose_ckpt["num_classes"]
        ).to(device)
        pose_model.load_state_dict(pose_ckpt["model_state_dict"])
        pose_model.eval()

        pose_ds = StaticPoseDataset(f_pose_test, y_pose_test, augment=False, base_aspect=4.0 / 3.0)
        pose_x = torch.stack([pose_ds[i][0] for i in range(len(pose_ds))], dim=0).to(device)
        with torch.no_grad():
            p_logits = pose_model(pose_x)
            p_probs = F.softmax(p_logits, dim=-1).cpu().numpy()
        p_preds = p_probs.argmax(axis=-1)
        p_acc = float(accuracy_score(y_pose_test, p_preds)) * 100.0
        p_f1 = float(f1_score(y_pose_test, p_preds, average="macro", zero_division=0)) * 100.0

        px_single = pose_x[:1]
        p_lats = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.perf_counter()
                _ = pose_model(px_single)
                p_lats.append((time.perf_counter() - t0) * 1000.0)
        p_med_lat = float(np.median(p_lats[5:]))

        print("\n" + "=" * 80)
        print(f"BƯỚC 6B: ĐÁNH GIÁ MODEL 1 (STATICPOSENET - 1 FRAME) TRÊN {len(y_pose_test)} TEST FRAMES")
        print(f"  - Pose Test Accuracy : {p_acc:.2f}% | Macro-F1: {p_f1:.2f}% | Latency: {p_med_lat:.3f} ms")
        print("=" * 80)
        pose_rep_dict = classification_report(
            y_pose_test, p_preds, target_names=POSE_CLASS_NAMES, digits=2, output_dict=True
        )
        print(classification_report(y_pose_test, p_preds, target_names=POSE_CLASS_NAMES, digits=2))
        pose_cm = confusion_matrix(y_pose_test, p_preds, labels=np.arange(len(POSE_CLASS_NAMES)))
        pose_eval_report = {
            "best_epoch_loaded": pose_ckpt.get("best_epoch", 0),
            "test_frames": len(y_pose_test),
            "test_accuracy_pct": round(p_acc, 2),
            "test_macro_f1_pct": round(p_f1, 2),
            "single_frame_latency_ms": round(p_med_lat, 3),
            "per_class_metrics": pose_rep_dict,
            "confusion_matrix": pose_cm.tolist(),
        }

    with open(DIAGNOSTICS_DIR / "test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "arch_name": arch_name,
            "best_epoch_loaded": ckpt.get("best_epoch", 0),
            "test_samples": len(y_test),
            "test_accuracy_pct": round(acc, 2),
            "test_macro_f1_pct": round(macro_f1, 2),
            "single_window_latency_ms": round(avg_latency, 3),
            "per_class_metrics": report_dict,
            "confusion_matrix": cm.tolist(),
            "confidence_threshold_audit": threshold_results,
            "static_pose_model": pose_eval_report,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_evaluation_pipeline()

