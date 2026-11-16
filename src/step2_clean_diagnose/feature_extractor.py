"""
MODULE NỘI SUY, LẤY MẪU THỜI GIAN & TRÍCH XUẤT 66 ĐẶC TRƯNG ĐỘNG
==================================================================
1. interpolate_missing_frames: Nội suy tuyến tính các frame khuyết (tối đa 25% thời lượng).
2. temporal_resample: Co giãn chuỗi từ độ dài bất kỳ về đúng T=30 frames.
3. extract_66d_dynamic_features: Trích xuất 63 đặc trưng hình thái bàn tay chuẩn hoá
   + 3 đặc trưng động học nhúng cổ tay + 3 đặc trưng vận tốc dịch chuyển toàn cục.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import SEQ_LEN


def interpolate_missing_frames(
    seq_raw: np.ndarray, max_missing_ratio: float = 0.25
) -> tuple[np.ndarray, int, float, bool]:
    """
    Phát hiện các frame bị mất dấu bàn tay và nội suy tuyến tính.
    Trả về: (seq_clean, num_missing, missing_ratio, is_valid)
    """
    t_len, num_coords = seq_raw.shape
    has_nan = np.isnan(seq_raw).any(axis=1)
    is_zero = np.all(np.isclose(seq_raw, 0.0, atol=1e-6), axis=1)
    missing_mask = has_nan | is_zero

    num_missing = int(np.sum(missing_mask))
    missing_ratio = num_missing / float(t_len)

    valid_indices = np.where(~missing_mask)[0]
    if len(valid_indices) < 5 or missing_ratio > max_missing_ratio:
        return seq_raw, num_missing, missing_ratio, False

    if num_missing == 0:
        return seq_raw.copy(), 0, 0.0, True

    seq_clean = np.zeros_like(seq_raw, dtype=np.float32)
    all_t = np.arange(t_len, dtype=np.float32)
    valid_t = valid_indices.astype(np.float32)

    for c in range(num_coords):
        valid_vals = seq_raw[valid_indices, c]
        seq_clean[:, c] = np.interp(all_t, valid_t, valid_vals)

    return seq_clean, num_missing, missing_ratio, True


def temporal_resample(seq_clean: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """Co giãn chuỗi thời gian về độ dài chuẩn target_len = 30 frames."""
    t_raw, num_coords = seq_clean.shape
    if t_raw == target_len:
        return seq_clean.copy()

    src_steps = np.linspace(0.0, 1.0, t_raw, dtype=np.float32)
    dst_steps = np.linspace(0.0, 1.0, target_len, dtype=np.float32)

    resampled = np.zeros((target_len, num_coords), dtype=np.float32)
    for c in range(num_coords):
        resampled[:, c] = np.interp(dst_steps, src_steps, seq_clean[:, c])

    return resampled


def extract_66d_dynamic_features(seq_30x63: np.ndarray) -> np.ndarray:
    """
    Biến đổi chuỗi toạ độ (30, 63) thành chuỗi đặc trưng động (30, 66):
    1. Hình thái 21 khớp chuẩn hoá theo gốc cổ tay P0 và kích thước lòng bàn tay s_t.
    2. Ba vị trí khớp 0 [0, 1, 2] được nhúng 3 đặc trưng ngón tay quan trọng:
       - Chiều [0]: Độ mở kẹp Cái-Trỏ chuẩn hoá d_{4,8}(t)
       - Chiều [1]: Vận tốc bung/khép ngón (Zoom Rate) = 5.0 * (d_{4,8}(t) - d_{4,8}(t-1))
       - Chiều [2]: Độ tương phản duỗi Trỏ vs Giữa = ||P_8|| - ||P_12||
    3. Ba chiều cuối [63, 64, 65]: Vận tốc dịch chuyển cổ tay (dx, dy, dz).
    """
    t_len = seq_30x63.shape[0]
    pts = seq_30x63.reshape(t_len, 21, 3)

    wrist = pts[:, 0, :]
    velocity = np.zeros((t_len, 3), dtype=np.float32)
    velocity[1:] = wrist[1:] - wrist[:-1]

    rel_pts = pts - wrist[:, np.newaxis, :]
    middle_mcp = pts[:, 9, :]
    palm_size = np.linalg.norm(middle_mcp - wrist, axis=1, keepdims=True) + 1e-6
    norm_pts = rel_pts / palm_size[:, :, np.newaxis]

    pinch_dist = np.linalg.norm(norm_pts[:, 8, :] - norm_pts[:, 4, :], axis=1)
    pinch_vel = np.zeros(t_len, dtype=np.float32)
    pinch_vel[1:] = (pinch_dist[1:] - pinch_dist[:-1]) * 5.0

    idx_ext = np.linalg.norm(norm_pts[:, 8, :], axis=1)
    mid_ext = np.linalg.norm(norm_pts[:, 12, :], axis=1)
    idx_mid_contrast = idx_ext - mid_ext

    norm_pts[:, 0, 0] = pinch_dist
    norm_pts[:, 0, 1] = pinch_vel
    norm_pts[:, 0, 2] = idx_mid_contrast

    shape_features = norm_pts.reshape(t_len, 63)
    features_66d = np.hstack([shape_features, velocity]).astype(np.float32)
    return features_66d


def extract_static_pose_features(pts_21x3: np.ndarray, aspect: float = 4.0 / 3.0) -> np.ndarray:
    """
    Trích xuất đặc trưng hình dáng tĩnh (1 frame đơn lẻ) bất biến xoay 360 độ trong mặt phẳng 2D,
    tách nhiễu độ sâu Z khỏi khoảng cách 2D và bổ sung góc thẳng khớp cục bộ cho StaticPoseNet.

    Đầu vào:
        pts_21x3: Mảng (21, 3) hoặc (63,) toạ độ MediaPipe chuẩn hoá [0, 1].
        aspect: Tỷ lệ khung hình (width / height) để khôi phục tỷ lệ pixel đẳng hướng trước khi xoay.
    Đầu ra:
        Mảng float32 shape (9, 21): 9 kênh đặc trưng hình học cho từng khớp 0..20
        [rx, ry, rz, d_wrist_2d, d_palm_2d, d_thumb_2d, d_index_2d, d_middle_2d, joint_cos_2d].
    """
    pts = np.asarray(pts_21x3, dtype=np.float32).reshape(21, 3).copy()
    pts[:, 0] *= float(aspect)

    wrist = pts[0:1, :]
    rel = pts - wrist
    palm_2d = float(np.linalg.norm(rel[9, :2]))
    palm_3d = float(np.linalg.norm(rel[9]))
    scale = max(palm_2d, 0.55 * palm_3d) + 1e-6

    xy_norm = rel[:, :2] / scale
    vx, vy = float(xy_norm[9, 0]), float(xy_norm[9, 1])
    if (vx * vx + vy * vy) > 1e-8:
        angle = -np.arctan2(vx, -vy)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        rx = xy_norm[:, 0] * cos_a - xy_norm[:, 1] * sin_a
        ry = xy_norm[:, 0] * sin_a + xy_norm[:, 1] * cos_a
    else:
        rx = xy_norm[:, 0]
        ry = xy_norm[:, 1]
    rz = np.clip(rel[:, 2] / scale, -2.5, 2.5).astype(np.float32)

    xy_c = np.stack([rx, ry], axis=1).astype(np.float32)
    d_wrist_2d = np.linalg.norm(xy_c, axis=1)
    d_palm_2d = np.linalg.norm(xy_c - xy_c[9:10, :], axis=1)
    d_thumb_2d = np.linalg.norm(xy_c - xy_c[4:5, :], axis=1)
    d_index_2d = np.linalg.norm(xy_c - xy_c[8:9, :], axis=1)
    d_middle_2d = np.linalg.norm(xy_c - xy_c[12:13, :], axis=1)

    # Kênh 8: Cosine góc thẳng dọc khớp xương 2D (bất biến co ngắn phối cảnh 3D)
    joint_cos = np.ones(21, dtype=np.float32)
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    palm_dir = xy_c[9]
    palm_len = float(np.linalg.norm(palm_dir)) + 1e-6
    for ch in chains:
        _, mcp, pip, dip, tip = ch
        v_palm_tip = xy_c[tip]
        joint_cos[mcp] = float(np.dot(palm_dir, v_palm_tip) / (palm_len * (float(np.linalg.norm(v_palm_tip)) + 1e-6)))
        for a, b, c in ((0, mcp, pip), (mcp, pip, dip), (pip, dip, tip)):
            v1 = xy_c[b] - xy_c[a]
            v2 = xy_c[c] - xy_c[b]
            n1 = float(np.linalg.norm(v1)) + 1e-6
            n2 = float(np.linalg.norm(v2)) + 1e-6
            if c == tip:
                joint_cos[dip] = float(np.dot(v1, v2) / (n1 * n2))
                # Độ thẳng toàn ngón từ MCP -> PIP so với PIP -> TIP đặt tại tip
                vm = xy_c[pip] - xy_c[mcp]
                vt = xy_c[tip] - xy_c[pip]
                joint_cos[tip] = float(np.dot(vm, vt) / ((float(np.linalg.norm(vm)) + 1e-6) * (float(np.linalg.norm(vt)) + 1e-6)))
            else:
                joint_cos[b] = float(np.dot(v1, v2) / (n1 * n2))

    features = np.stack(
        [rx, ry, rz, d_wrist_2d, d_palm_2d, d_thumb_2d, d_index_2d, d_middle_2d, joint_cos],
        axis=0,
    ).astype(np.float32)
    return features


