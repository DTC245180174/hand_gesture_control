"""
BƯỚC 6 (AIR-INK): ĐÁNH GIÁ ĐỘC LẬP MODEL 3 (AIRINKRESCNN - 9 LỚP HÌNH HỌC)
===========================================================================
Đánh giá `outputs/models/best_air_ink_model.pt` trên tập Test độc lập (`2,520` mẫu)
và tập Webcam Holdout thật (`X_webcam_holdout.npy`, nếu có):
1. Hiệu năng tổng thể 9 lớp hình học (Accuracy, Top-2, Macro-F1, Weighted-F1).
2. Kiểm định tỷ lệ kích hoạt nhầm từ nét tự do (`freeform`) & ma trận nhầm lẫn 9x9.
3. Kiểm định ngưỡng tin cậy (Confidence Gate Audit: Precision vs Coverage).
4. Đo độ trễ suy luận CPU thực tế (batch=1) + Xuất biểu đồ `fig10` & JSON.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from common.config import (
    AIR_INK_CLASSES,
    AIR_INK_NAME_TO_ID,
    AIR_INK_PROCESSED_DIR,
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    NUM_AIR_INK_CLASSES,
)
from step4_model_selection.models import build_air_ink_model


@torch.no_grad()
def _run_batched_logits(
    model: torch.nn.Module,
    X_u8: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    logits_chunks: List[torch.Tensor] = []
    n = len(X_u8)
    for start in range(0, n, batch_size):
        batch = X_u8[start : start + batch_size].astype(np.float32) * (1.0 / 255.0)
        xt = torch.from_numpy(batch[:, None, :, :]).to(device)
        logits_chunks.append(model(xt).cpu())
    return torch.cat(logits_chunks, dim=0).numpy()


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    return exp_l / np.maximum(exp_l.sum(axis=1, keepdims=True), 1e-12)


def _audit_thresholds(
    y_true: np.ndarray,
    preds: np.ndarray,
    confs: np.ndarray,
    thresholds: Sequence[float],
    freeform_id: int,
) -> List[Dict[str, float]]:
    """Đo trade-off giữa Coverage (tỷ lệ chấp nhận) và Accepted Precision theo ngưỡng."""
    rows: List[Dict[str, float]] = []
    for th in thresholds:
        accepted = confs >= float(th)
        cov = float(np.mean(accepted) * 100.0)
        acc_prec = float(np.mean(preds[accepted] == y_true[accepted]) * 100.0) if np.any(accepted) else 100.0
        row: Dict[str, float] = {
            "threshold": round(float(th), 2),
            "coverage_pct": round(cov, 2),
            "accepted_precision_pct": round(acc_prec, 2),
        }
        ff_mask = y_true == freeform_id
        if np.any(ff_mask):
            false_shape = (preds[ff_mask] != freeform_id) & (confs[ff_mask] >= float(th))
            row["freeform_false_shape_pct"] = round(float(np.mean(false_shape) * 100.0), 2)
        rows.append(row)
    return rows


def _save_eval_figure(
    cm: np.ndarray,
    labels: Sequence[str],
    acc_pct: float,
    shapes_thresh: List[Dict[str, float]],
) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=140)

    ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    ax1.set_title(f"AIR-INK 9 HINH HOC — Test Acc: {acc_pct:.2f}%", fontweight="bold")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=32, ha="right", fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    thresh_val = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh_val else "black",
            )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Ground Truth")

    ths = [r["threshold"] for r in shapes_thresh]
    prec = [r["accepted_precision_pct"] for r in shapes_thresh]
    cov = [r["coverage_pct"] for r in shapes_thresh]
    ax2.plot(ths, prec, "o-", color="#2ecc71", lw=2, label="Accepted Precision (%)")
    ax2.plot(ths, cov, "o--", color="#27ae60", alpha=0.75, lw=1.6, label="Coverage (%)")
    ax2.axvline(0.72, color="#34495e", linestyle=":", lw=1.5, label="Deploy Gate (0.72)")
    ax2.set_title("Confidence Threshold Audit (9 Shape Classes)", fontweight="bold")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Score (%)")
    ax2.set_ylim(45, 102)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="lower left")

    fig10_path = FIGURES_DIR / "fig10_air_ink_evaluation.png"
    fig.tight_layout()
    fig.savefig(fig10_path)
    plt.close(fig)
    return fig10_path


def run_air_ink_evaluation_pipeline(
    checkpoint_path: Path = MODELS_DIR / "best_air_ink_model.pt",
) -> Dict[str, object]:
    """Chạy kiểm định độc lập toàn diện Model 3 (9 lớp hình học + Webcam Holdout)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = np.load(AIR_INK_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(AIR_INK_PROCESSED_DIR / "y_test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_air_ink_model(
        in_channels=int(ckpt.get("in_channels", 1)),
        num_classes=int(ckpt.get("num_classes", NUM_AIR_INK_CLASSES)),
        dropout=float(ckpt.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 84)
    print(f"BƯỚC 6 (AIR-INK): ĐÁNH GIÁ ĐỘC LẬP MODEL 3 TRÊN TẬP TEST ({len(y_test):,d} MẪU)")
    print(f"  - Checkpoint : {checkpoint_path.name} @ Epoch {ckpt.get('best_epoch', 0)}")
    print("=" * 84)

    logits_np = _run_batched_logits(model, X_test, device, batch_size=256)
    probs = _softmax_np(logits_np)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    acc = float(accuracy_score(y_test, preds) * 100.0)
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0) * 100.0)

    ranked = np.argsort(-logits_np, axis=1)
    top2_acc = float(np.mean([y_test[i] in ranked[i, :2] for i in range(len(y_test))]) * 100.0)
    top3_acc = float(np.mean([y_test[i] in ranked[i, :3] for i in range(len(y_test))]) * 100.0)

    labels = list(AIR_INK_CLASSES)
    rep = classification_report(
        y_test,
        preds,
        labels=list(range(NUM_AIR_INK_CLASSES)),
        target_names=labels,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(NUM_AIR_INK_CLASSES)))

    top_confusions: List[Dict[str, object]] = []
    for i in range(cm.shape[0]):
        row_sum = int(cm[i].sum())
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                top_confusions.append({
                    "true_label": labels[i],
                    "pred_label": labels[j],
                    "count": int(cm[i, j]),
                    "rate_of_true_pct": round(float(cm[i, j]) / max(row_sum, 1) * 100.0, 2),
                })
    top_confusions.sort(key=lambda d: (-int(d["count"]), -float(d["rate_of_true_pct"])))

    # Đo độ trễ suy luận 1 mẫu trên CPU
    x_single = torch.from_numpy((X_test[:1].astype(np.float32) * (1.0 / 255.0))[:, None, :, :]).to(device)
    lats: List[float] = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(x_single)
            lats.append((time.perf_counter() - t0) * 1000.0)
    med_lat_ms = float(np.median(lats[5:]))

    freeform_id = AIR_INK_NAME_TO_ID["freeform"]
    shapes_thresh = _audit_thresholds(
        y_test, preds, confs, thresholds=[0.0, 0.70, 0.72, 0.85, 0.90], freeform_id=freeform_id
    )

    print(f"\n1. TỔNG HỢP {NUM_AIR_INK_CLASSES} LỚP HÌNH HỌC:")
    print(f"  - Test Accuracy : {acc:.2f}% ({int((preds == y_test).sum()):,d}/{len(y_test):,d})")
    print(f"  - Top-2 / Top-3 : {top2_acc:.2f}% / {top3_acc:.2f}%")
    print(f"  - Macro-F1      : {macro_f1:.2f}% | Weighted-F1: {weighted_f1:.2f}%")
    print(f"  - CPU Latency   : {med_lat_ms:.3f} ms/cụm nét (~{int(1000.0 / max(med_lat_ms, 1e-3))} FPS)")

    if top_confusions:
        print("\n2. NHẦM LẪN TRÊN TẬP TEST:")
        for item in top_confusions[:5]:
            print(f"  - '{item['true_label']}' -> '{item['pred_label']}': {item['count']} mẫu ({item['rate_of_true_pct']}%)")
    else:
        print("\n2. NHẦM LẪN TRÊN TẬP TEST: 0 lỗi nhầm lẫn!")

    print("\n3. KIỂM ĐỊNH NGƯỠNG TIN CẬY (CONFIDENCE GATE AUDIT):")
    for r in shapes_thresh:
        print(
            f"  th={r['threshold']:.2f} -> Coverage={r['coverage_pct']:>6.2f}% | "
            f"Accepted Prec={r['accepted_precision_pct']:>6.2f}% | "
            f"Freeform False Convert={r.get('freeform_false_shape_pct', 0.0):>5.2f}%"
        )

    webcam_report: Dict[str, object] = {"samples": 0, "status": "Chưa có mẫu Webcam Holdout"}
    wc_x_path = AIR_INK_PROCESSED_DIR / "X_webcam_holdout.npy"
    wc_y_path = AIR_INK_PROCESSED_DIR / "y_webcam_holdout.npy"
    if wc_x_path.exists() and wc_y_path.exists():
        X_wc = np.load(wc_x_path)
        y_wc = np.load(wc_y_path)
        if len(y_wc) > 0:
            wc_logits = _run_batched_logits(model, X_wc, device, batch_size=128)
            wc_preds = wc_logits.argmax(axis=1)
            wc_acc = float(accuracy_score(y_wc, wc_preds) * 100.0)
            webcam_report = {"samples": int(len(y_wc)), "accuracy_pct": round(wc_acc, 2)}
            print(f"\n4. WEBCAM HOLDOUT THỰC TẾ ({len(y_wc)} mẫu): Acc = {wc_acc:.2f}%")

    fig10_path = _save_eval_figure(cm, labels, acc, shapes_thresh)

    report_out = {
        "model_class": "AirInkResCNN",
        "num_classes": NUM_AIR_INK_CLASSES,
        "best_epoch_loaded": int(ckpt.get("best_epoch", 0)),
        "test_samples": int(len(y_test)),
        "accuracy_pct": round(acc, 2),
        "top2_accuracy_pct": round(top2_acc, 2),
        "top3_accuracy_pct": round(top3_acc, 2),
        "macro_f1_pct": round(macro_f1, 2),
        "weighted_f1_pct": round(weighted_f1, 2),
        "single_sample_cpu_latency_ms": round(med_lat_ms, 3),
        "top_confusions": top_confusions[:15],
        "per_class_f1_pct": {
            lb: round(float(rep[lb]["f1-score"]) * 100.0, 2)
            for lb in labels
        },
        "confidence_threshold_audit": shapes_thresh,
        "webcam_holdout": webcam_report,
    }
    report_path = DIAGNOSTICS_DIR / "air_ink_test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2, ensure_ascii=False)

    print("-" * 84)
    print(f"[FIGURE SAVED] {fig10_path}")
    print(f"[REPORT SAVED] {report_path}")
    return report_out


if __name__ == "__main__":
    run_air_ink_evaluation_pipeline()
