"""
DATASET & TĂNG CƯỜNG DỮ LIỆU CHUỖI KHUNG XƯƠNG ĐỘNG (SEQUENCE DATASET & AUGMENTATION)
====================================================================================
Quản lý chuỗi thời gian (T=30, F=66) và áp dụng các phép biến đổi vật lý hợp lệ:
1. Scale Jitter: Co giãn nhẹ tỷ lệ bàn tay (0.90 -> 1.10).
2. Gaussian Noise: Mô phỏng sai số trích xuất khớp MediaPipe.
3. Velocity Amplitude Scaling: Biến thiên tốc độ chuyển động (0.75 -> 1.25).
4. Drift Augmentation: Mô phỏng trôi ngón tay khi đang rê bút vẽ chữ (point_1f / point_2f).
5. Sliding-Window Speed Warping: Mô phỏng tốc độ thực hiện nhanh / chậm trong cửa sổ trượt.
6. Temporal Shift: Dịch chuỗi +-1..4 frames dọc theo trục thời gian.
7. Random Frame Dropout: Che ngẫu nhiên 1-2 frames.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from torch.utils.data import Dataset

from common.config import SEQ_LEN


class HandGestureSequenceDataset(Dataset):
    """Dataset quản lý chuỗi đặc trưng động (T=30, F=66)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def _augment_sequence(self, seq: np.ndarray, label: int = 0) -> Tuple[np.ndarray, int]:
        out = seq.copy()
        norm_pts = out[:, :63].reshape(SEQ_LEN, 21, 3)
        joints = norm_pts[:, 1:21, :]  # 20 khớp ngón tay (1..20), gốc tại cổ tay (0,0,0)
        vel = out[:, 63:66]

        # 0. Lật gương ngang ngẫu nhiên (50%) -> học đồng thời tay trái & tay phải, đảo nhãn swipe 1 <-> 2
        if np.random.rand() < 0.50:
            joints[:, :, 0] *= -1.0
            vel[:, 0] *= -1.0
            if label == 1:
                label = 2
            elif label == 2:
                label = 1

        # 1. Phép xoay mặt phẳng 2D (+-40 độ) giúp mô hình động học bất biến với độ nghiêng cổ tay
        if np.random.rand() < 0.70:
            max_deg = 40.0
            theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            joints[:, :, :2] = joints[:, :, :2] @ rot_mat.T
            if label in (1, 2):
                vel[:, :2] = vel[:, :2] @ rot_mat.T

        # 2. Co giãn bất đẳng hướng & nghiêng phối cảnh 3D theo trục X/Y/Z (Mô phỏng xoay cổ tay 3D)
        if np.random.rand() < 0.70:
            sx = float(np.random.uniform(0.85, 1.15))
            sy = float(np.random.uniform(0.85, 1.15))
            sz = float(np.random.uniform(0.82, 1.18))
            joints[:, :, 0] *= sx
            joints[:, :, 1] *= sy
            joints[:, :, 2] *= sz

        # 3. Gaussian Noise trên 20 khớp ngón tay (Mô phỏng rung khớp MediaPipe)
        if np.random.rand() < 0.65:
            noise = np.random.normal(0.0, 0.008, size=joints.shape).astype(np.float32)
            joints = joints + noise

        # Cập nhật lại 20 khớp và tính lại chính xác 3 đặc trưng động học tại slot 0
        norm_pts[:, 1:21, :] = joints
        pinch_dist = np.linalg.norm(norm_pts[:, 8, :] - norm_pts[:, 4, :], axis=1)
        idx_ext = np.linalg.norm(norm_pts[:, 8, :], axis=1)
        mid_ext = np.linalg.norm(norm_pts[:, 12, :], axis=1)
        norm_pts[:, 0, 0] = pinch_dist
        norm_pts[:, 0, 2] = idx_ext - mid_ext
        pose = norm_pts.reshape(SEQ_LEN, 63)

        # 4. Biến thiên vận tốc cổ tay (0.82 -> 1.18)
        if np.random.rand() < 0.60:
            vel_scale = np.random.uniform(0.82, 1.18)
            vel = vel * vel_scale
            vel_noise = np.random.normal(0.0, 0.0012, size=vel.shape).astype(np.float32)
            vel = vel + vel_noise

        # 5. Active Drawing / Instant Pose Augmentation cho point_1f (3) và point_2f (4):
        #    Hỗ trợ cả suy luận tức thời 1-frame (vận tốc = 0) lẫn lúc đang rê bút vẽ chữ trên Slide
        if label in (3, 4):
            r_mode = np.random.rand()
            if r_mode < 0.35:
                vel[:] = 0.0
            elif r_mode < 0.80:
                t_axis = np.linspace(0.0, np.pi * np.random.uniform(0.8, 2.0), SEQ_LEN, dtype=np.float32)
                phase = float(np.random.uniform(0.0, np.pi))
                drift_dx = np.random.uniform(-0.0085, 0.0085) * np.sin(t_axis + phase)
                drift_dy = np.random.uniform(-0.0095, 0.0095) * np.cos(t_axis)
                vel[:, 0] = vel[:, 0] * 0.5 + drift_dx
                vel[:, 1] = vel[:, 1] * 0.5 + drift_dy

        out = np.concatenate([pose, vel], axis=-1)

        # 6. Speed Warping vừa phải (22..29 frames) để không làm mất xung cử chỉ
        if np.random.rand() < 0.35:
            sub_len = int(np.random.randint(22, 29))
            src_t = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)
            dst_t = np.linspace(0.0, 1.0, sub_len, dtype=np.float32)
            warped = np.zeros((sub_len, out.shape[1]), dtype=np.float32)
            for c in range(out.shape[1]):
                warped[:, c] = np.interp(dst_t, src_t, out[:, c])
            warped[:, 63:66] *= float(SEQ_LEN) / float(sub_len)

            pad_total = SEQ_LEN - sub_len
            pad_left = int(np.random.randint(0, pad_total + 1))
            pad_right = pad_total - pad_left

            left_part = np.repeat(warped[:1], pad_left, axis=0)
            left_part[:, 63:66] = 0.0
            right_part = np.repeat(warped[-1:], pad_right, axis=0)
            right_part[:, 63:66] = 0.0
            out = np.concatenate([left_part, warped, right_part], axis=0)

        # 7. Temporal Shift nhẹ (+-1..2 frames)
        if np.random.rand() < 0.35:
            shift = np.random.choice([-2, -1, 1, 2])
            if shift > 0:
                out = np.concatenate([np.repeat(out[:1], shift, axis=0), out[:-shift]], axis=0)
                out[:shift, 63:66] = 0.0
            elif shift < 0:
                s = abs(shift)
                out = np.concatenate([out[s:], np.repeat(out[-1:], s, axis=0)], axis=0)
                out[-s:, 63:66] = 0.0

        # 8. Random Frame Dropout
        if np.random.rand() < 0.35:
            num_mask = np.random.randint(1, 3)
            mask_indices = np.random.choice(SEQ_LEN, size=num_mask, replace=False)
            for idx in mask_indices:
                if 0 < idx < SEQ_LEN - 1:
                    out[idx] = 0.5 * (out[idx - 1] + out[idx + 1])

        # Đồng bộ lại đạo hàm vận tốc bung/khép ngón (cột 1) từ khoảng cách Cái-Trỏ (cột 0)
        out[0, 1] = 0.0
        out[1:, 1] = (out[1:, 0] - out[:-1, 0]) * 5.0
        return out.astype(np.float32), label

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.X[index]
        label = int(self.y[index])
        if self.augment:
            seq, label = self._augment_sequence(seq, label=label)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)

