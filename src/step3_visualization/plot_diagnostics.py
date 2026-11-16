"""
BƯỚC 3: TRỰC QUAN HOÁ DỮ LIỆU CỬ CHỈ ĐỘNG (DATA VISUALIZATION & EDA)
====================================================================
Sinh ra đủ 4 biểu đồ kiểm tra chất lượng dữ liệu lưu tại `outputs/figures/`:
1. fig1_class_and_length_distribution.png: Phân bố số lượng mẫu và độ dài.
2. fig2_hand_skeletons_8classes.png: Cấu trúc khung xương 21 khớp của 8 lớp.
3. fig3_velocity_and_aperture_profiles.png: Đồ thị vận tốc ngang và độ mở ngón.
4. fig4_feature_space_projection.png: Phân cụm không gian đặc trưng (PCA 2D).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.config import (
    CLASS_NAMES,
    DIAGNOSTICS_DIR,
    FIGURES_DIR,
    HAND_CONNECTIONS,
    PROCESSED_DIR,
)

CLASS_COLORS = [
    "#7f8c8d",  # 0: idle (Xám)
    "#e74c3c",  # 1: swipe_left (Đỏ)
    "#2980b9",  # 2: swipe_right (Xanh dương)
    "#8e44ad",  # 3: point_1f (Tím)
    "#16a085",  # 4: point_2f (Xanh mòng két)
    "#e67e22",  # 5: zoom_in (Cam)
    "#27ae60",  # 6: zoom_out (Xanh lá)
    "#f1c40f",  # 7: click (Vàng sáng)
]


def run_visualizations(
    processed_dir: str = str(PROCESSED_DIR),
    diag_dir: str = str(DIAGNOSTICS_DIR),
    figures_dir: str = str(FIGURES_DIR),
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    report_path = os.path.join(diag_dir, "data_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Không tìm thấy {report_path}. Hãy chạy Bước 2 trước!")

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    raw30_train_path = os.path.join(processed_dir, "raw30_train.npy")
    raw30_train = np.load(raw30_train_path) if os.path.exists(raw30_train_path) else None

    print("=" * 84)
    print("BƯỚC 3: TRỰC QUAN HOÁ DỮ LIỆU & ĐẶC TRƯNG ĐỘNG HỌC")
    print("=" * 84)

    # 1. Fig 1: Phân bố số lượng mẫu Train / Val / Test
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    indices = np.arange(len(CLASS_NAMES))
    train_counts = [int(np.sum(y_train == i)) for i in range(len(CLASS_NAMES))]
    val_counts = [int(np.sum(y_val == i)) for i in range(len(CLASS_NAMES))]
    test_counts = [int(np.sum(y_test == i)) for i in range(len(CLASS_NAMES))]

    bar_w = 0.25
    ax.bar(indices - bar_w, train_counts, width=bar_w, label="Train", color="#3498db")
    ax.bar(indices, val_counts, width=bar_w, label="Val", color="#f39c12")
    ax.bar(indices + bar_w, test_counts, width=bar_w, label="Test", color="#2ecc71")

    ax.set_xticks(indices)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title(f"Phân Bố Số Lượng Mẫu {len(CLASS_NAMES)} Lớp Cử Chỉ (Train / Val / Test)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    f1_path = os.path.join(figures_dir, "fig1_class_and_length_distribution.png")
    fig.tight_layout()
    fig.savefig(f1_path)
    plt.close(fig)
    print(f"[*] Đã xuất: {f1_path}")

    # 2. Fig 2: Cấu trúc khung xương 21 khớp của các lớp (tại frame giữa t=15)
    if raw30_train is not None:
        fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(2.6 * len(CLASS_NAMES), 3.2), dpi=140)
        for cid, cname in enumerate(CLASS_NAMES):
            ax = axes[cid]
            mask = y_train == cid
            if np.any(mask):
                sample_pts = raw30_train[mask][0, 15].reshape(21, 3)
                for s_idx, e_idx in HAND_CONNECTIONS:
                    ax.plot(
                        [sample_pts[s_idx, 0], sample_pts[e_idx, 0]],
                        [sample_pts[s_idx, 1], sample_pts[e_idx, 1]],
                        color=CLASS_COLORS[cid % len(CLASS_COLORS)],
                        linewidth=2,
                    )
                ax.scatter(sample_pts[:, 0], sample_pts[:, 1], color="#2c3e50", s=18, zorder=3)
            ax.invert_yaxis()
            ax.set_title(f"{cid}: {cname}", fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        f2_path = os.path.join(figures_dir, "fig2_hand_skeletons_8classes.png")
        fig.tight_layout()
        fig.savefig(f2_path)
        plt.close(fig)
        print(f"[*] Đã xuất: {f2_path}")

    # 3. Fig 3: Đồ thị vận tốc ngang dX và độ mở kẹp Cái-Trỏ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
    for cid, cname in enumerate(CLASS_NAMES):
        mask = y_train == cid
        if not np.any(mask):
            continue
        feats_cls = X_train[mask]
        dx_mean = np.mean(feats_cls[:, :, 63], axis=0)
        ap_mean = np.mean(feats_cls[:, :, 0], axis=0)

        t_steps = np.arange(1, 31)
        ax1.plot(t_steps, dx_mean, label=cname, color=CLASS_COLORS[cid], linewidth=2)
        ax2.plot(t_steps, ap_mean, label=cname, color=CLASS_COLORS[cid], linewidth=2)

    ax1.set_title("Vận Tốc Ngang Cổ Tay (dx_t) qua 30 Frames", fontweight="bold")
    ax1.set_xlabel("Frame (1..30)")
    ax1.set_ylabel("dx (pixel / palm_size)")
    ax1.axhline(0, color="gray", linestyle=":")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_title("Độ Mở Kẹp Cái - Trỏ (d_4_8) qua 30 Frames", fontweight="bold")
    ax2.set_xlabel("Frame (1..30)")
    ax2.set_ylabel("Khoảng cách chuẩn hoá")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    f3_path = os.path.join(figures_dir, "fig3_velocity_and_aperture_profiles.png")
    fig.tight_layout()
    fig.savefig(f3_path)
    plt.close(fig)
    print(f"[*] Đã xuất: {f3_path}")

    # 4. Fig 4: Chiếu không gian đặc trưng 2D bằng SVD / PCA
    flat_X = X_train.reshape(len(X_train), -1)
    flat_X_centered = flat_X - np.mean(flat_X, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(flat_X_centered, full_matrices=False)
    proj_2d = u[:, :2] * s[:2]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    for cid, cname in enumerate(CLASS_NAMES):
        mask = y_train == cid
        if np.any(mask):
            ax.scatter(
                proj_2d[mask, 0],
                proj_2d[mask, 1],
                label=cname,
                color=CLASS_COLORS[cid],
                alpha=0.7,
                s=22,
            )
    ax.set_title("Phân Cụm Không Gian Đặc Trưng 66Dx30 (PCA 2D)", fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    f4_path = os.path.join(figures_dir, "fig4_feature_space_projection.png")
    fig.tight_layout()
    fig.savefig(f4_path)
    plt.close(fig)
    print(f"[*] Đã xuất: {f4_path}")

    print(f"[HOÀN TẤT BƯỚC 3] Đã tạo toàn bộ biểu đồ tại: {figures_dir}")


if __name__ == "__main__":
    run_visualizations()
