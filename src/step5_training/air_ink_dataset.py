"""
DATASET & TĂNG CƯỜNG DỮ LIỆU ẢNH NÉT VẼ 64x64 CHO MODEL 3 (AIR-INK)
=====================================================================
Quản lý tập ảnh nét vẽ xám `(N, 64, 64)` gồm 9 lớp hình học vẽ không trung
(`line`, `circle`, `rectangle`, `triangle`, `arrow`, `checkmark`, `cross`, `star`, `freeform`):
1. Biến đổi Affine đơn nhất (Xoay an toàn theo lớp + Co giãn dị hướng + Dịch chuyển).
   - Lớp nhạy góc trục (`line`): giữ xoay trong ±7° để bảo toàn hướng gạch chuẩn.
   - Các lớp còn lại: xoay trong ±11° mô phỏng nghiêng cổ tay vẽ tự nhiên.
2. Biến thiên độ mảnh/dày nét bằng hình thái học (`dilate`/`erode` nhẹ).
3. Cutout cục bộ (5x5..8x8) mô phỏng rớt 1-2 frame MediaPipe khi vẽ trên không.
4. Biến thiên cường độ sáng nét & nhiễu cảm biến nhẹ.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import FrozenSet, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.config import AIR_INK_CLASSES, AIR_INK_IMG_SIZE, NUM_AIR_INK_CLASSES

_TIGHT_ROT_CLASS_NAMES: FrozenSet[str] = frozenset({"line"})
_KERNEL_2x2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_KERNEL_3x3_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def compute_air_ink_class_weights(
    y_train: np.ndarray, num_classes: int = NUM_AIR_INK_CLASSES
) -> torch.Tensor:
    """Tính trọng số cân bằng nghịch đảo căn bậc 2 chuẩn hoá về trung bình 1.0."""
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes)
    total = float(len(y_train))
    weights = np.sqrt(total / (counts.astype(np.float32) * float(num_classes) + 1e-6))
    weights = weights / np.maximum(weights.mean(), 1e-6)
    return torch.from_numpy(weights.astype(np.float32))


class AirInkDataset(Dataset):
    """Dataset ảnh nét vẽ không gian 1x64x64 lưu uint8 tiết kiệm RAM và augment tức thì."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        class_names: Sequence[str] = AIR_INK_CLASSES,
    ) -> None:
        if X.dtype != np.uint8:
            max_val = float(X.max()) if X.size > 0 else 0.0
            scale = 255.0 if max_val <= 1.5 else 1.0
            self.X = np.clip(X * scale, 0, 255).astype(np.uint8)
        else:
            self.X = X
        self.y = y.astype(np.int64)
        self.augment = bool(augment)
        self.tight_rot_ids: FrozenSet[int] = frozenset(
            idx for idx, name in enumerate(class_names) if name in _TIGHT_ROT_CLASS_NAMES
        )

    def __len__(self) -> int:
        return len(self.y)

    def _augment_image(self, img_u8: np.ndarray, label: int) -> np.ndarray:
        h, w = img_u8.shape
        cx = 0.5 * (w - 1)
        cy = 0.5 * (h - 1)
        out = img_u8

        # 1. Phép biến đổi Affine gộp (Xoay an toàn + Co giãn + Dịch chuyển)
        if np.random.rand() < 0.80:
            max_deg = 7.0 if label in self.tight_rot_ids else 11.0
            theta = np.deg2rad(float(np.random.uniform(-max_deg, max_deg)))
            cos_t = float(np.cos(theta))
            sin_t = float(np.sin(theta))
            sx = float(np.random.uniform(0.86, 1.12))
            sy = float(np.clip(sx * np.random.uniform(0.92, 1.08), 0.84, 1.14))
            tx = float(np.random.uniform(-3.5, 3.5))
            ty = float(np.random.uniform(-3.5, 3.5))

            affine_mat = np.array(
                [
                    [sx * cos_t, -sy * sin_t, (1.0 - sx * cos_t) * cx + sy * sin_t * cy + tx],
                    [sx * sin_t, sy * cos_t, -sx * sin_t * cx + (1.0 - sy * cos_t) * cy + ty],
                ],
                dtype=np.float32,
            )
            out = cv2.warpAffine(
                out,
                affine_mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # 2. Thay đổi độ mảnh/dày nét nhẹ để bất biến kích thước bút viết
        r_morph = float(np.random.rand())
        if r_morph < 0.18:
            out = cv2.dilate(out, _KERNEL_3x3_CROSS, iterations=1)
        elif r_morph < 0.30 and int((out > 40).sum()) > 80:
            out = cv2.erode(out, _KERNEL_2x2, iterations=1)

        # 3. Cutout đứt nét cục bộ (mô phỏng rớt khung hình khi viết nhanh)
        if np.random.rand() < 0.25:
            out = out.copy()
            bh = int(np.random.randint(4, 8))
            bw = int(np.random.randint(4, 8))
            margin = max(6, int(round(min(h, w) * 0.12)))
            y0 = int(np.random.randint(margin, max(margin + 1, h - margin - bh)))
            x0 = int(np.random.randint(margin, max(margin + 1, w - margin - bw)))
            out[y0 : y0 + bh, x0 : x0 + bw] = 0

        # Tránh tình huống ảnh bị trống hoàn toàn sau augment cực đoan
        if int(out.max()) < 25:
            out = img_u8

        out_f32 = out.astype(np.float32) * (1.0 / 255.0)
        if np.random.rand() < 0.50:
            gain = float(np.random.uniform(0.82, 1.12))
            noise = np.random.normal(0.0, 0.015, size=out_f32.shape).astype(np.float32)
            out_f32 = np.clip(out_f32 * gain + noise, 0.0, 1.0)
        return out_f32

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_u8 = self.X[idx]
        label = int(self.y[idx])

        if self.augment:
            img_f32 = self._augment_image(img_u8, label)
        else:
            img_f32 = img_u8.astype(np.float32) * (1.0 / 255.0)

        if img_f32.shape != (AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE):
            img_f32 = cv2.resize(
                img_f32, (AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )

        return torch.from_numpy(img_f32[None, :, :]), torch.tensor(label, dtype=torch.long)
