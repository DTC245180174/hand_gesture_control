"""
BƯỚC 5 (PHẦN B): DATASET TƯ THẾ TĨNH 1 FRAME CHO STATICPOSENET (OPTION B - MODEL 1)
================================================================================
Trích xuất từng khung hình (21 khớp 3D) từ raw30_*.npy để huấn luyện StaticPoseNet
phân biệt 3 lớp tư thế cốt lõi từ đầu:
  - 0: other_pose  (idle, swipe_left, swipe_right, zoom_in, zoom_out, click)
  - 1: point_1f    (1 ngón trỏ - vùng ổn định giữa chuỗi)
  - 2: point_2f    (2 ngón Trỏ + Giữa - vùng ổn định giữa chuỗi)
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

from common.config import POSE_FEATURE_DIM
from step2_clean_diagnose.feature_extractor import extract_static_pose_features


def extract_single_frame_pose_samples(
    raw30: np.ndarray,
    y_seq: np.ndarray,
    is_train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuyển tập chuỗi thô (N, 30, 63) thành tập mẫu khung hình đơn (M, 21, 3) & nhãn (M,).
    - point_1f (3) & point_2f (4): lấy vùng ổn định [4:26] ở Train để tránh frame chuyển tiếp đầu/cuối;
      ở Val/Test lấy vùng tâm ổn định [8:22:2] để đo độ chính xác tư thế tinh khiết.
    - Các lớp khác (0, 1, 2, 5, 6, 7): lấy stride đều để giữ cân bằng số lượng khung hình.
    """
    frames_list = []
    labels_list = []

    frame_slice_ptr = range(6, 24) if is_train else range(8, 22, 2)
    frame_slice_idle = range(0, 30, 2) if is_train else range(5, 25, 3)
    frame_slice_other = range(0, 30, 3) if is_train else range(5, 25, 4)
    frame_slice_zoom = range(0, 30, 4) if is_train else range(5, 25, 4)

    for seq_idx in range(len(raw30)):
        cid = int(y_seq[seq_idx])
        seq = raw30[seq_idx].reshape(30, 21, 3)

        if cid == 3:
            pose_lbl = 1
            indices = frame_slice_ptr
        elif cid == 4:
            pose_lbl = 2
            indices = frame_slice_ptr
        elif cid == 0:
            pose_lbl = 0
            indices = frame_slice_idle
        elif cid in (5, 6):
            pose_lbl = 0
            indices = frame_slice_zoom
        else:
            pose_lbl = 0
            indices = frame_slice_other

        for t in indices:
            pts = seq[t]
            if float(np.linalg.norm(pts)) < 1e-4:
                continue
            if float(np.linalg.norm(pts[9] - pts[0])) < 1e-4:
                continue
            frames_list.append(pts.astype(np.float32))
            labels_list.append(pose_lbl)

    if not frames_list:
        return np.zeros((0, 21, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(frames_list, axis=0), np.array(labels_list, dtype=np.int64)


class StaticPoseDataset(Dataset):
    """Dataset khung hình đơn kèm Data Augmentation cho StaticPoseNet."""

    def __init__(
        self,
        frames_21x3: np.ndarray,
        y_pose: np.ndarray,
        augment: bool = False,
        base_aspect: float = 4.0 / 3.0,
    ) -> None:
        self.frames = frames_21x3.astype(np.float32)
        self.y = torch.from_numpy(y_pose.astype(np.int64))
        self.augment = augment
        self.base_aspect = float(base_aspect)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pts = self.frames[idx].copy()
        aspect = self.base_aspect

        if self.augment:
            # 1. Lật gương ngang ngẫu nhiên (50%) -> học cả tay trái và tay phải
            if np.random.rand() < 0.50:
                pts[:, 0] = 1.0 - pts[:, 0]

            # 2. Biến thiên nhẹ tỷ lệ khung hình thực tế (4:3 <-> 16:9) mà không làm méo cấu trúc ngón
            if np.random.rand() < 0.45:
                aspect = float(np.random.uniform(1.30, 1.80))

            # 3. Nghiêng phối cảnh 3D vừa phải trong không gian đẳng hướng (tránh gập quá mức thành nắm đấm)
            wrist = pts[0:1, :].copy()
            rel = pts - wrist
            if np.random.rand() < 0.50:
                yaw = np.deg2rad(np.random.uniform(-22.0, 22.0))
                pitch = np.deg2rad(np.random.uniform(-25.0, 25.0))
                cy, sy = np.cos(yaw), np.sin(yaw)
                cp, sp = np.cos(pitch), np.sin(pitch)
                x_iso = rel[:, 0] * self.base_aspect
                y_iso = rel[:, 1]
                z_iso = rel[:, 2] * self.base_aspect
                x1 = x_iso * cy - z_iso * sy
                z1 = x_iso * sy + z_iso * cy
                y2 = y_iso * cp - z1 * sp
                z2 = y_iso * sp + z1 * cp
                rel[:, 0] = x1 / self.base_aspect
                rel[:, 1] = y2
                rel[:, 2] = z2 / self.base_aspect

            # 4. Biến thiên nhẹ tỷ lệ chiều dài từng ngón tay (người lạ tay dài/ngắn khác nhau)
            if np.random.rand() < 0.40:
                for start_j, end_j in ((1, 5), (5, 9), (9, 13), (13, 17), (17, 21)):
                    f_scale = float(np.random.uniform(0.92, 1.08))
                    rel[start_j:end_j] *= f_scale

            # 5. Nhiễu vi mô tại từng khớp MediaPipe
            if np.random.rand() < 0.45:
                palm = float(np.linalg.norm(rel[9])) + 1e-6
                rel += np.random.normal(0.0, 0.008 * palm, size=rel.shape).astype(np.float32)

            pts = rel + wrist

        feat = extract_static_pose_features(pts, aspect=aspect)
        if feat.shape != (POSE_FEATURE_DIM, 21):
            feat = np.zeros((POSE_FEATURE_DIM, 21), dtype=np.float32)
        return torch.from_numpy(feat), self.y[idx]
