"""
CHUẨN HOÁ ẢNH NÉT VẼ KHÔNG GIAN (AIR-INK RASTERIZER & SYNTHETIC STROKE ENGINE)
================================================================================
Module dùng chung từ Bước 1C/Bước 2 đến Bước 7 Deploy (chỉ dùng NumPy + OpenCV):
1. `rasterize_strokes_to_64x64`: Biến danh sách nét vẽ 2D [(x, y), ...] thành ảnh
   xám chuẩn 64x64 (nền đen 0, nét trắng 255) căn giữa giữ nguyên tỷ lệ khung hình.
2. `generate_synthetic_air_strokes`: Sinh quỹ đạo nét vẽ giả lập run tay tự nhiên
   cho 9 lớp hình học (`line`, `circle`, `rectangle`, `triangle`, `arrow`,
   `checkmark`, `cross`, `star`, `freeform`).
"""

from __future__ import annotations
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _apply_hand_tremor(pts: np.ndarray, rng: np.random.Generator, amp: float = 0.025) -> np.ndarray:
    """Thêm dao động run tay tần số thấp dọc theo quỹ đạo nét vẽ."""
    if len(pts) < 3:
        return pts
    t = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
    f1 = rng.uniform(1.0, 2.8)
    p1 = rng.uniform(0.0, 2.0 * np.pi)
    f2 = rng.uniform(1.2, 3.2)
    p2 = rng.uniform(0.0, 2.0 * np.pi)
    jx = amp * np.sin(2.0 * np.pi * f1 * t + p1) + rng.normal(0.0, amp * 0.25, size=len(pts))
    jy = amp * np.cos(2.0 * np.pi * f2 * t + p2) + rng.normal(0.0, amp * 0.25, size=len(pts))
    out = pts.copy().astype(np.float32)
    out[:, 0] += jx.astype(np.float32)
    out[:, 1] += jy.astype(np.float32)
    return out


def _sample_segment(p0: Tuple[float, float], p1: Tuple[float, float], n_pts: int = 20) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)[:, None]
    a = np.array(p0, dtype=np.float32)[None, :]
    b = np.array(p1, dtype=np.float32)[None, :]
    return a + t * (b - a)


def rasterize_strokes_to_64x64(
    strokes: Sequence[np.ndarray],
    size: int = 64,
    thickness: int = 2,
    pad_ratio: float = 0.16,
    blur_sigma: float = 0.40,
) -> np.ndarray:
    """
    Rasterize cụm nét vẽ 2D thành ảnh uint8 shape (size, size) với nền đen (0) nét trắng (255).
    Giữ nguyên tỷ lệ khung hình (Aspect-Preserving Padding) để dấu `-` không biến thành ô vuông.
    """
    valid_strokes = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in strokes if len(s) >= 1]
    canvas = np.zeros((size, size), dtype=np.uint8)
    if not valid_strokes:
        return canvas

    all_pts = np.vstack(valid_strokes)
    x_min, y_min = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
    x_max, y_max = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    w = x_max - x_min
    h = y_max - y_min
    span = max(w, h)
    if span < 1e-4:
        return canvas

    usable = float(size) * max(0.20, 1.0 - 2.0 * float(pad_ratio))
    scale = usable / span
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * (size - 1)

    thick_px = max(1, int(thickness))
    dot_r = max(2, int(round(thick_px * 1.15)))

    for st in valid_strokes:
        px = np.round((st[:, 0] - cx) * scale + half).astype(np.int32)
        py = np.round((st[:, 1] - cy) * scale + half).astype(np.int32)
        px = np.clip(px, 1, size - 2)
        py = np.clip(py, 1, size - 2)
        pts_i = np.stack([px, py], axis=1)

        if len(pts_i) == 1:
            cv2.circle(canvas, (int(pts_i[0, 0]), int(pts_i[0, 1])), dot_r, 255, -1, cv2.LINE_AA)
        else:
            # Nếu nét rất ngắn (chấm tròn của ÷ hoặc ?), tô tròn rõ ràng
            st_span = float(np.hypot(st[:, 0].max() - st[:, 0].min(), st[:, 1].max() - st[:, 1].min()))
            if st_span < 0.08 * span:
                mx, my = int(round(pts_i[:, 0].mean())), int(round(pts_i[:, 1].mean()))
                cv2.circle(canvas, (mx, my), dot_r, 255, -1, cv2.LINE_AA)
            else:
                for i in range(len(pts_i) - 1):
                    cv2.line(
                        canvas,
                        (int(pts_i[i, 0]), int(pts_i[i, 1])),
                        (int(pts_i[i + 1, 0]), int(pts_i[i + 1, 1])),
                        255,
                        thickness=thick_px,
                        lineType=cv2.LINE_AA,
                    )

    if blur_sigma > 0.0:
        blurred = cv2.GaussianBlur(canvas, (3, 3), sigmaX=float(blur_sigma))
        peak = int(blurred.max())
        if peak > 0:
            canvas = np.clip(blurred.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)
    return canvas


def _rotate_strokes(strokes: List[np.ndarray], deg: float) -> List[np.ndarray]:
    if abs(deg) < 1e-3:
        return strokes
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    out = []
    for st in strokes:
        x = st[:, 0] * c - st[:, 1] * s
        y = st[:, 0] * s + st[:, 1] * c
        out.append(np.stack([x, y], axis=1).astype(np.float32))
    return out


def generate_synthetic_air_strokes(class_name: str, rng: np.random.Generator) -> List[np.ndarray]:
    """
    Sinh chuỗi nét vẽ 2D mô phỏng thao tác vẽ tay trên không trung cho 9 lớp hình học.
    """
    tremor = float(rng.uniform(0.012, 0.038))
    strokes: List[np.ndarray] = []

    if class_name == "line":
        mode = int(rng.integers(0, 3))
        if mode == 0:  # Gạch chân ngang
            ang = np.deg2rad(rng.uniform(-16.0, 16.0))
        elif mode == 1:  # Gạch dọc
            ang = np.deg2rad(90.0 + rng.uniform(-16.0, 16.0))
        else:  # Đường chéo
            ang = np.deg2rad(rng.choice([-45.0, 45.0]) + rng.uniform(-15.0, 15.0))
        d = rng.uniform(0.7, 1.0)
        p0 = (-d * np.cos(ang), -d * np.sin(ang))
        p1 = (d * np.cos(ang), d * np.sin(ang))
        if rng.random() < 0.5:
            p0, p1 = p1, p0
        pts = _sample_segment(p0, p1, n_pts=28)
        # Độ cong tự nhiên nhẹ của cánh tay khi gạch ngang
        curve = rng.uniform(-0.08, 0.08)
        t = np.linspace(-1.0, 1.0, len(pts))
        pts[:, 0] += -np.sin(ang) * curve * (1.0 - t * t)
        pts[:, 1] += np.cos(ang) * curve * (1.0 - t * t)
        strokes.append(_apply_hand_tremor(pts, rng, tremor * 0.7))

    elif class_name == "circle":
        n_pts = int(rng.integers(34, 52))
        start = rng.uniform(0.0, 2.0 * np.pi)
        span = rng.uniform(1.85 * np.pi, 2.18 * np.pi) * rng.choice([-1.0, 1.0])
        th = np.linspace(start, start + span, n_pts)
        rx = rng.uniform(0.65, 1.0)
        ry = rng.uniform(0.65, 1.0)
        pts = np.stack([rx * np.cos(th), ry * np.sin(th)], axis=1).astype(np.float32)
        strokes = _rotate_strokes([_apply_hand_tremor(pts, rng, tremor)], float(rng.uniform(-25.0, 25.0)))

    elif class_name == "rectangle":
        w = rng.uniform(0.65, 1.25)
        h = rng.uniform(0.50, 1.15)
        skew_x = rng.uniform(-0.08, 0.08)
        corners = [
            (-w + skew_x, -h),
            (w + rng.uniform(-0.06, 0.06), -h + rng.uniform(-0.06, 0.06)),
            (w - skew_x, h),
            (-w + rng.uniform(-0.06, 0.06), h + rng.uniform(-0.06, 0.06)),
        ]
        style = int(rng.integers(0, 3))
        if style == 0:  # 1 nét khép kín liên tục
            seq = []
            idx0 = int(rng.integers(0, 4))
            direction = int(rng.choice([-1, 1]))
            for k in range(4):
                a = corners[(idx0 + direction * k) % 4]
                b = corners[(idx0 + direction * (k + 1)) % 4]
                seg = _sample_segment(a, b, 12)
                seq.append(seg if k == 0 else seg[1:])
            pts = np.vstack(seq)
            if rng.random() < 0.45:  # Hở nhẹ góc cuối
                pts = pts[:-2]
            strokes.append(_apply_hand_tremor(pts, rng, tremor))
        elif style == 1:  # 2 nét chữ L + L ngược
            s1 = np.vstack([_sample_segment(corners[0], corners[1], 12), _sample_segment(corners[1], corners[2], 12)[1:]])
            s2 = np.vstack([_sample_segment(corners[0], corners[3], 12), _sample_segment(corners[3], corners[2], 12)[1:]])
            strokes.extend([_apply_hand_tremor(s1, rng, tremor), _apply_hand_tremor(s2, rng, tremor)])
        else:  # 4 nét rời có nhô góc nhẹ
            for k in range(4):
                a, b = corners[k], corners[(k + 1) % 4]
                seg = _sample_segment(a, b, 12)
                strokes.append(_apply_hand_tremor(seg, rng, tremor))
        strokes = _rotate_strokes(strokes, float(rng.uniform(-12.0, 12.0)))

    elif class_name == "triangle":
        top = (rng.uniform(-0.22, 0.22), rng.uniform(-0.95, -0.65))
        left = (rng.uniform(-1.0, -0.65), rng.uniform(0.60, 0.95))
        right = (rng.uniform(0.65, 1.0), rng.uniform(0.60, 0.95))
        verts = [top, right, left]
        if rng.random() < 0.65:
            seq = []
            idx0 = int(rng.integers(0, 3))
            d = int(rng.choice([-1, 1]))
            for k in range(3):
                a, b = verts[(idx0 + d * k) % 3], verts[(idx0 + d * (k + 1)) % 3]
                seg = _sample_segment(a, b, 14)
                seq.append(seg if k == 0 else seg[1:])
            strokes.append(_apply_hand_tremor(np.vstack(seq), rng, tremor))
        else:
            for k in range(3):
                strokes.append(_apply_hand_tremor(_sample_segment(verts[k], verts[(k + 1) % 3], 14), rng, tremor))
        strokes = _rotate_strokes(strokes, float(rng.uniform(-20.0, 20.0)))

    elif class_name == "arrow":
        # Mũi tên bất kỳ góc nào 0..360 độ (chủ yếu 4 hướng + chéo)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(0.75, 1.0)
        head_len = rng.uniform(0.50, 0.78) * length
        head_ang = np.deg2rad(rng.uniform(28.0, 44.0))
        tail = np.array([-length, 0.0], dtype=np.float32)
        tip = np.array([length, rng.uniform(-0.08, 0.08)], dtype=np.float32)
        w1 = np.array([tip[0] - head_len * np.cos(head_ang), tip[1] - head_len * np.sin(head_ang)], dtype=np.float32)
        w2 = np.array([tip[0] - head_len * np.cos(head_ang), tip[1] + head_len * np.sin(head_ang)], dtype=np.float32)

        if rng.random() < 0.55:  # 2 nét: thân + đầu chữ V
            s_shaft = _sample_segment(tuple(tail), tuple(tip), 20)
            s_head = np.vstack([_sample_segment(tuple(w1), tuple(tip), 10), _sample_segment(tuple(tip), tuple(w2), 10)[1:]])
            raw_s = [s_shaft, s_head]
        else:  # 1 nét vẽ liền mạch hoặc 3 nét
            if rng.random() < 0.5:
                s1 = np.vstack([_sample_segment(tuple(tail), tuple(tip), 18), _sample_segment(tuple(tip), tuple(w1), 10)[1:]])
                s2 = _sample_segment(tuple(tip), tuple(w2), 10)
                raw_s = [s1, s2]
            else:
                raw_s = [_sample_segment(tuple(tail), tuple(tip), 18), _sample_segment(tuple(w1), tuple(tip), 10), _sample_segment(tuple(w2), tuple(tip), 10)]
        strokes = _rotate_strokes([_apply_hand_tremor(s, rng, tremor * 0.85) for s in raw_s], float(np.rad2deg(ang)))

    elif class_name == "checkmark":
        p0 = (rng.uniform(-0.90, -0.55), rng.uniform(-0.05, 0.35))
        pv = (rng.uniform(-0.30, 0.05), rng.uniform(0.55, 0.90))
        p2 = (rng.uniform(0.55, 1.00), rng.uniform(-0.95, -0.50))
        if rng.random() < 0.80:
            pts = np.vstack([_sample_segment(p0, pv, 14), _sample_segment(pv, p2, 22)[1:]])
            strokes.append(_apply_hand_tremor(pts, rng, tremor * 0.85))
        else:
            strokes.append(_apply_hand_tremor(_sample_segment(p0, pv, 14), rng, tremor))
            strokes.append(_apply_hand_tremor(_sample_segment(pv, p2, 22), rng, tremor))
        strokes = _rotate_strokes(strokes, float(rng.uniform(-15.0, 15.0)))

    elif class_name == "cross":
        r = rng.uniform(0.75, 1.0)
        d1 = rng.uniform(-0.18, 0.18)
        d2 = rng.uniform(-0.18, 0.18)
        s1 = _sample_segment((-r + d1, -r), (r, r + d2), 18)
        s2 = _sample_segment((r, -r + d1), (-r + d2, r), 18)
        if rng.random() < 0.5:
            s1 = s1[::-1]
        strokes = _rotate_strokes(
            [_apply_hand_tremor(s1, rng, tremor), _apply_hand_tremor(s2, rng, tremor)],
            float(rng.uniform(-14.0, 14.0)),
        )

    elif class_name == "star":
        if rng.random() < 0.72:
            # Ngôi sao 5 cánh nét liền cổ điển
            angles = np.deg2rad(np.array([-90, 54, 198, -18, 126, -90], dtype=np.float32) + rng.uniform(-8, 8, size=6))
            radii = rng.uniform(0.80, 1.05, size=6)
            radii[-1] = radii[0]
            verts = [(float(radii[i] * np.cos(angles[i])), float(radii[i] * np.sin(angles[i]))) for i in range(6)]
            pts = np.vstack([_sample_segment(verts[i], verts[i + 1], 10)[1 if i > 0 else 0 :] for i in range(5)])
            strokes.append(_apply_hand_tremor(pts, rng, tremor * 0.85))
        else:
            # Ngôi sao 2 tam giác lồng (David star / teacher quick star)
            for flip in (1.0, -1.0):
                ang = np.deg2rad(np.array([-90, 30, 150]) * flip)
                vs = [(float(np.cos(a)), float(np.sin(a))) for a in ang]
                seq = np.vstack([_sample_segment(vs[i], vs[(i + 1) % 3], 12)[1 if i > 0 else 0 :] for i in range(3)])
                strokes.append(_apply_hand_tremor(seq, rng, tremor))
        strokes = _rotate_strokes(strokes, float(rng.uniform(-18.0, 18.0)))

    elif class_name == "freeform":
        style = int(rng.integers(0, 4))
        t = np.linspace(0.0, 1.0, int(rng.integers(36, 65)), dtype=np.float32)
        if style == 0:  # Sóng lượn nhiều nhịp (không phải 1 gạch thẳng)
            cycles = rng.uniform(1.8, 4.0)
            pts = np.stack([t * 2.0 - 1.0, 0.45 * np.sin(2.0 * np.pi * cycles * t + rng.uniform(0, 6))], axis=1)
            strokes.append(_apply_hand_tremor(pts, rng, tremor * 1.3))
        elif style == 1:  # Xoắn ốc / nguệch ngoạc tự do
            th = t * rng.uniform(3.2, 6.0) * np.pi
            rad = 0.15 + 0.85 * t
            pts = np.stack([rad * np.cos(th), rad * np.sin(th) * rng.uniform(0.6, 1.3)], axis=1)
            strokes.append(_apply_hand_tremor(pts, rng, tremor * 1.4))
        elif style == 2:  # Hình số 8 / vòng lặp kép
            th = t * 2.0 * np.pi
            pts = np.stack([np.sin(th), np.sin(2.0 * th) * rng.uniform(0.6, 1.1)], axis=1)
            strokes.append(_apply_hand_tremor(pts, rng, tremor * 1.2))
        else:  # Chữ viết nguệch đa nét ngẫu nhiên
            n_sub = int(rng.integers(2, 4))
            for _ in range(n_sub):
                anchor = rng.uniform(-0.8, 0.8, size=(int(rng.integers(4, 7)), 2)).astype(np.float32)
                seg_list = [_sample_segment(tuple(anchor[i]), tuple(anchor[i + 1]), 8) for i in range(len(anchor) - 1)]
                strokes.append(_apply_hand_tremor(np.vstack(seg_list), rng, tremor))
        strokes = _rotate_strokes(strokes, float(rng.uniform(-40.0, 40.0)))

    else:
        raise ValueError(f"Chưa hỗ trợ sinh nét giả lập cho lớp: {class_name}")

    return strokes
