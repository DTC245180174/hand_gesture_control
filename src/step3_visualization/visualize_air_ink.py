"""
BƯỚC 3 (AIR-INK): TRỰC QUAN HOÁ & KIỂM ĐỊNH DỮ LIỆU 9 LỚP HÌNH HỌC 64x64
==========================================================================
Xuất biểu đồ `outputs/figures/fig8_air_ink_domain_check.png` kiểm tra:
1. Mẫu chuẩn hoá 64x64 của cả 9 lớp Hình học trong tập Test.
2. Đa dạng biến thể nét vẽ (độ dày/run tay/góc nghiêng) trong tập Train.
3. Phân bố mật độ pixel nét trắng (>40/255) giữa tập Train và Test.
"""

from __future__ import annotations
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
    AIR_INK_CLASSES,
    AIR_INK_NAME_TO_ID,
    AIR_INK_PROCESSED_DIR,
    FIGURES_DIR,
)


def run_air_ink_visualizations() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train = np.load(AIR_INK_PROCESSED_DIR / "X_train.npy")
    y_train = np.load(AIR_INK_PROCESSED_DIR / "y_train.npy")
    X_test = np.load(AIR_INK_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(AIR_INK_PROCESSED_DIR / "y_test.npy")

    fig = plt.figure(figsize=(15, 9), facecolor="#0F172A")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.35, 0.95], hspace=0.45)

    # Hàng 1: 9 lớp Hình học trên tập Test
    ax_row1 = fig.add_subplot(gs[0])
    ax_row1.set_axis_off()
    ax_row1.set_title(
        "1. Dai dien 9 Lop Hinh Hoc Tren Tap Test (Rasterized 64x64)",
        color="#F8FAFC", fontsize=12, fontweight="bold", pad=10,
    )
    sub1 = gs[0].subgridspec(1, len(AIR_INK_CLASSES), wspace=0.10)
    for col, cname in enumerate(AIR_INK_CLASSES):
        cid = AIR_INK_NAME_TO_ID[cname]
        idx = int(np.where(y_test == cid)[0][0])
        a = fig.add_subplot(sub1[0, col])
        a.imshow(X_test[idx], cmap="magma")
        a.set_title(cname, color="#34D399", fontsize=10, fontweight="bold")
        a.axis("off")

    # Hàng 2: Biến thể nét vẽ Train (2 biến thể mỗi lớp)
    ax_row2 = fig.add_subplot(gs[1])
    ax_row2.set_axis_off()
    ax_row2.set_title(
        "2. Da Dang Bien The Net Ve Tap Train (Do day 1-3px, rung tay, goc xoay)",
        color="#F8FAFC", fontsize=12, fontweight="bold", pad=10,
    )
    sub2 = gs[1].subgridspec(2, len(AIR_INK_CLASSES), wspace=0.10, hspace=0.14)
    for col, cname in enumerate(AIR_INK_CLASSES):
        cid = AIR_INK_NAME_TO_ID[cname]
        idxs = np.where(y_train == cid)[0]
        for r in range(2):
            pick = int(idxs[min(r * 7 + 1, len(idxs) - 1)])
            a = fig.add_subplot(sub2[r, col])
            a.imshow(X_train[pick], cmap="gray")
            if r == 0:
                a.set_title(cname, color="#38BDF8", fontsize=9)
            a.axis("off")

    # Hàng 3: Phân bố mật độ pixel nét trắng (>40)
    ax_row3 = fig.add_subplot(gs[2])
    ax_row3.set_facecolor("#1E293B")
    fg_tr = (X_train > 40).mean(axis=(1, 2)) * 100.0
    fg_te = (X_test > 40).mean(axis=(1, 2)) * 100.0

    ax_row3.hist(fg_tr, bins=28, alpha=0.55, density=True, label=f"Train (1-3px, TB={fg_tr.mean():.1f}%)", color="#38BDF8")
    ax_row3.hist(fg_te, bins=28, alpha=0.60, density=True, label=f"Test (2px, TB={fg_te.mean():.1f}%)", color="#34D399")
    ax_row3.set_title(
        "3. Phan Bo Mat Do Diem Anh Net Trang (>40/255) Tren Khung Hinh 64x64",
        color="#F8FAFC", fontsize=11, fontweight="bold",
    )
    ax_row3.set_xlabel("% Dien tich net ve (>40/255) tren khung 64x64", color="#CBD5E1")
    ax_row3.set_ylabel("Mat do", color="#CBD5E1")
    ax_row3.tick_params(colors="#CBD5E1")
    ax_row3.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")

    out_path = FIGURES_DIR / "fig8_air_ink_domain_check.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[FIGURE SAVED] Đã xuất biểu đồ kiểm định dữ liệu Air-Ink tại: {out_path}")
    return out_path


if __name__ == "__main__":
    run_air_ink_visualizations()
