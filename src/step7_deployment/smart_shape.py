"""
SMART SHAPE VECTORIZER & POWERPOINT COM VECTOR SHAPE ENGINE (STAGE 5+)
======================================================================
Module chuyển đổi kết quả nhận diện 9 lớp hình từ Model 3 (`AirInkResCNN`)
thành hình học sắc nét trên PowerPoint:
1. Ưu tiên chèn **Vector Shape thật của PowerPoint COM** (`AddShape`, `AddLine`,
   `BuildFreeform`) khi đang trình chiếu Slide Show (`F5`) — đường nét sắc gọn 100%,
   có gắn tag `__AI_SHAPE_...__` để tẩy 2 ngón (`point_2f`) xoá trực tiếp được.
2. Nâng cấp **Mực vẽ fallback (`replay_win32_strokes`)**: khớp hình chuẩn
   (PCA đường thẳng, tam giác theo 6 mẫu gần nhất, vòng tròn 72 điểm, ngôi sao
   10 đỉnh viền ngoài, mũi tên theo nét thân dài nhất) + nghỉ tại đỉnh góc nhọn
   (`corner dwell`) để PowerPoint không cắt vát góc.
3. Nâng cấp **Tẩy nét nháp (`erase_raw_strokes_on_ppt`)**: quét đa điểm dọc toàn bộ
   chiều dài từng nét (kể cả chấm ngắn) trước khi đặt hình sạch.
"""

from __future__ import annotations
from contextlib import contextmanager
import ctypes
import math
import os
import time
from typing import Iterator, List, Optional, Sequence, Set, Tuple
import uuid

import numpy as np

from common.win32_input import (
    ensure_powerpoint_foreground,
    is_powerpoint_slideshow_active,
    win32_drag_cursor,
    win32_mouse_down,
    win32_mouse_up,
    win32_send_ctrl_key,
)

# Office / PowerPoint COM enumeration constants (verified against Microsoft Learn)
_MSO_FALSE = 0
_MSO_TRUE = -1
_MSO_SHAPE_RECTANGLE = 1
_MSO_SHAPE_ISOSCELES_TRIANGLE = 7
_MSO_SHAPE_OVAL = 9
_MSO_SHAPE_5POINT_STAR = 92
_MSO_ARROWHEAD_TRIANGLE = 2
_MSO_ARROWHEAD_MEDIUM = 2
_MSO_SEGMENT_LINE = 0
_MSO_EDITING_AUTO = 0
_MSO_EDITING_CORNER = 1

PPT_AI_SHAPE_PREFIX = "__AI_SHAPE_"
_DEFAULT_PEN_COLOR_BGR = 0x0000E6  # Đỏ tươi chuẩn nét bút PowerPoint (BGR: R=230, G=0, B=0)


@contextmanager
def _high_res_win_timer(period_ms: int = 1) -> Iterator[None]:
    """Bật timer độ phân giải 1ms trên Windows để `time.sleep` mượt khi kéo/tẩy chuột."""
    winmm = None
    if os.name == "nt":
        try:
            winmm = ctypes.windll.winmm
            winmm.timeBeginPeriod(int(period_ms))
        except Exception:
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            try:
                winmm.timeEndPeriod(int(period_ms))
            except Exception:
                pass


def _clip_pt(x: float, y: float, screen_w: int, screen_h: int, margin: int = 8) -> Tuple[int, int]:
    ix = int(np.clip(round(x), margin, max(margin + 1, screen_w - margin)))
    iy = int(np.clip(round(y), margin, max(margin + 1, screen_h - margin)))
    return ix, iy


def _stroke_arc_length(st: np.ndarray) -> float:
    if len(st) < 2:
        return 0.0
    diffs = np.diff(st[:, :2], axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _interpolate_polyline(
    pts: Sequence[Tuple[float, float]], step_px: float = 8.0
) -> List[Tuple[float, float]]:
    """Nội suy đều theo bước `step_px` và giữ nguyên mọi đỉnh góc."""
    if len(pts) < 2:
        return list(pts)
    dense: List[Tuple[float, float]] = [pts[0]]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dist = math.hypot(x1 - x0, y1 - y0)
        n_seg = max(1, int(math.ceil(dist / max(step_px, 4.0))))
        for s in range(1, n_seg + 1):
            a = s / float(n_seg)
            dense.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
    return dense


def _snap_direction_angle(dx: float, dy: float, tol_deg: float = 12.0) -> Tuple[float, float]:
    """Nắn góc vector đơn vị về trục ngang, dọc hoặc chéo 45° nếu lệch trong ngưỡng `tol_deg`."""
    ang = math.degrees(math.atan2(dy, dx))
    targets = (-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0)
    best_t = min(targets, key=lambda t: abs(ang - t))
    if abs(ang - best_t) <= tol_deg:
        rad = math.radians(best_t)
        return math.cos(rad), math.sin(rad)
    norm = max(math.hypot(dx, dy), 1e-6)
    return dx / norm, dy / norm


def _fit_line_pca(
    valid: Sequence[np.ndarray], all_pts: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Khớp đoạn thẳng bằng PCA trên toàn cụm điểm và giữ đúng chiều vẽ."""
    mean_pt = all_pts.mean(axis=0)
    centered = all_pts - mean_pt
    if len(centered) >= 3:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u = eigvecs[:, int(np.argmax(eigvals))]
        ux, uy = float(u[0]), float(u[1])
    else:
        ux = float(valid[-1][-1, 0] - valid[0][0, 0])
        uy = float(valid[-1][-1, 1] - valid[0][0, 1])

    ux, uy = _snap_direction_angle(ux, uy, tol_deg=12.0)
    proj = centered[:, 0] * ux + centered[:, 1] * uy
    t_min, t_max = float(proj.min()), float(proj.max())
    if t_max - t_min < 12.0:
        t_min, t_max = -12.0, 12.0

    pa = (float(mean_pt[0] + t_min * ux), float(mean_pt[1] + t_min * uy))
    pb = (float(mean_pt[0] + t_max * ux), float(mean_pt[1] + t_max * uy))
    start_raw = valid[0][0]
    if math.hypot(start_raw[0] - pb[0], start_raw[1] - pb[1]) < math.hypot(
        start_raw[0] - pa[0], start_raw[1] - pa[1]
    ):
        pa, pb = pb, pa
    return pa, pb


def _sample_closed_poly_perimeter(
    verts: Sequence[Tuple[float, float]], n_per_edge: int = 12
) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        for s in range(n_per_edge):
            a = s / float(n_per_edge)
            pts.append((x0 + a * (x1 - x0), y0 + a * (y1 - y0)))
    return np.asarray(pts, dtype=np.float32)


def _fit_triangle_vertices(
    all_pts: np.ndarray, x0: float, y0: float, x1: float, y1: float
) -> List[Tuple[float, float]]:
    """Chọn mẫu tam giác hình học sạch khớp sát nhất với nét vẽ tay (cân, ngược, hoặc vuông 4 góc)."""
    cx = 0.5 * (x0 + x1)
    candidates = [
        [(cx, y0), (x1, y1), (x0, y1)],  # Tam giác cân đứng
        [(x0, y0), (x1, y0), (cx, y1)],  # Tam giác cân ngược
        [(x0, y0), (x0, y1), (x1, y1)],  # Tam giác vuông góc dưới-trái
        [(x1, y0), (x1, y1), (x0, y1)],  # Tam giác vuông góc dưới-phải
        [(x0, y0), (x1, y0), (x0, y1)],  # Tam giác vuông góc trên-trái
        [(x0, y0), (x1, y0), (x1, y1)],  # Tam giác vuông góc trên-phải
    ]
    if len(all_pts) > 80:
        idx = np.linspace(0, len(all_pts) - 1, 80, dtype=int)
        obs = all_pts[idx]
    else:
        obs = all_pts

    best_verts = candidates[0]
    best_err = float("inf")
    for verts in candidates:
        perim = _sample_closed_poly_perimeter(verts, n_per_edge=12)
        d2 = ((obs[:, None, :] - perim[None, :, :]) ** 2).sum(axis=2)
        err = float(d2.min(axis=1).mean())
        if err < best_err:
            best_err = err
            best_verts = verts
    return best_verts


def _infer_arrow_tail_tip(
    raw_strokes: Sequence[Sequence[Tuple[float, float]]]
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Ước lượng điểm đuôi (tail) và đỉnh mũi tên (tip) bất biến 360° & thứ tự vẽ."""
    arrs = [np.asarray(st, dtype=np.float32).reshape(-1, 2) for st in raw_strokes if len(st) >= 2]
    if len(arrs) >= 2:
        lengths = [_stroke_arc_length(a) for a in arrs]
        shaft_idx = int(np.argmax(lengths))
        shaft = arrs[shaft_idx]
        p_start = (float(shaft[0, 0]), float(shaft[0, 1]))
        p_end = (float(shaft[-1, 0]), float(shaft[-1, 1]))
        other_pts = np.vstack([a for i, a in enumerate(arrs) if i != shaft_idx])
        head_center = other_pts.mean(axis=0)
        d_start = math.hypot(p_start[0] - float(head_center[0]), p_start[1] - float(head_center[1]))
        d_end = math.hypot(p_end[0] - float(head_center[0]), p_end[1] - float(head_center[1]))
        tail, tip = (p_end, p_start) if d_start < d_end else (p_start, p_end)
    else:
        s0 = arrs[0]
        tail = (float(s0[0, 0]), float(s0[0, 1]))
        dists = np.hypot(s0[:, 0] - tail[0], s0[:, 1] - tail[1])
        tip_idx = int(np.argmax(dists))
        tip = (float(s0[tip_idx, 0]), float(s0[tip_idx, 1]))

    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    length = max(math.hypot(dx, dy), 24.0)
    ux, uy = _snap_direction_angle(dx, dy, tol_deg=10.0)
    tip_snapped = (tail[0] + length * ux, tail[1] + length * uy)
    return tail, tip_snapped


def _generate_star_outer_verts(cx: float, cy: float, outer_r: float) -> List[Tuple[float, float]]:
    """Sinh 10 đỉnh viền ngoài của ngôi sao 5 cánh (rỗng ruột, không nét đan chéo)."""
    inner_r = 0.382 * outer_r
    verts: List[Tuple[float, float]] = []
    for k in range(11):
        ang = math.radians(-90.0 + 36.0 * k)
        rad = outer_r if (k % 2 == 0) else inner_r
        verts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return verts


def generate_clean_shape_strokes(
    shape_name: str,
    raw_strokes: Sequence[Sequence[Tuple[float, float]]],
    screen_w: int,
    screen_h: int,
) -> List[List[Tuple[int, int]]]:
    """Sinh danh sách các nét pixel sạch cho hình học được Model 3 nhận diện."""
    valid = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in raw_strokes if len(s) >= 2]
    if not valid:
        return []

    all_pts = np.vstack(valid)
    x0, y0 = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
    x1, y1 = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw = max(x1 - x0, 24.0)
    bh = max(y1 - y0, 24.0)

    out_float: List[List[Tuple[float, float]]] = []

    if shape_name == "line":
        pa, pb = _fit_line_pca(valid, all_pts)
        out_float.append(_interpolate_polyline([pa, pb], step_px=8.0))

    elif shape_name == "circle":
        ratio = bw / max(bh, 1.0)
        if 0.78 <= ratio <= 1.28:
            rx = ry = 0.25 * (bw + bh)
        else:
            rx, ry = 0.5 * bw, 0.5 * bh
        n_pts = 72
        thetas = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
        ring = [(cx + rx * math.cos(float(t)), cy + ry * math.sin(float(t))) for t in thetas]
        ring.append(ring[0])
        out_float.append(ring)

    elif shape_name == "rectangle":
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        out_float.append(_interpolate_polyline(corners, step_px=8.0))

    elif shape_name == "triangle":
        tri = _fit_triangle_vertices(all_pts, x0, y0, x1, y1)
        verts = tri + [tri[0]]
        out_float.append(_interpolate_polyline(verts, step_px=8.0))

    elif shape_name == "arrow":
        tail, tip = _infer_arrow_tail_tip(valid)
        dx, dy = tip[0] - tail[0], tip[1] - tail[1]
        shaft_len = max(math.hypot(dx, dy), 30.0)
        ux, uy = dx / shaft_len, dy / shaft_len
        head_len = float(np.clip(0.24 * shaft_len, 18.0, 44.0))
        ang_w = math.radians(28.0)
        cos_w, sin_w = math.cos(ang_w), math.sin(ang_w)
        bx, by = -ux, -uy
        w1 = (tip[0] + head_len * (bx * cos_w - by * sin_w), tip[1] + head_len * (bx * sin_w + by * cos_w))
        w2 = (tip[0] + head_len * (bx * cos_w + by * sin_w), tip[1] + head_len * (-bx * sin_w + by * cos_w))
        out_float.append(_interpolate_polyline([tail, tip], step_px=8.0))
        out_float.append(_interpolate_polyline([w1, tip, w2], step_px=7.0))

    elif shape_name == "checkmark":
        p0 = (x0 + 0.08 * bw, y0 + 0.52 * bh)
        pv = (x0 + 0.36 * bw, y1 - 0.06 * bh)
        p2 = (x1 - 0.05 * bw, y0 + 0.08 * bh)
        out_float.append(_interpolate_polyline([p0, pv, p2], step_px=8.0))

    elif shape_name == "cross":
        out_float.append(_interpolate_polyline([(x0, y0), (x1, y1)], step_px=8.0))
        out_float.append(_interpolate_polyline([(x1, y0), (x0, y1)], step_px=8.0))

    elif shape_name == "star":
        rad = 0.5 * min(bw, bh)
        verts = _generate_star_outer_verts(cx, cy, rad)
        out_float.append(_interpolate_polyline(verts, step_px=8.0))

    return [[_clip_pt(px, py, screen_w, screen_h) for px, py in st] for st in out_float]


def _get_corner_indices(
    st: Sequence[Tuple[int, int]], angle_deg_thresh: float = 32.0
) -> Set[int]:
    """Xác định các đỉnh góc nhọn dọc theo polyline để tạm dừng nhỏ khi vẽ chuột."""
    corners: Set[int] = set()
    if len(st) < 3:
        return corners
    cos_thresh = math.cos(math.radians(angle_deg_thresh))
    for i in range(1, len(st) - 1):
        v1x, v1y = st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]
        v2x, v2y = st[i + 1][0] - st[i][0], st[i + 1][1] - st[i][1]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        cos_ang = (v1x * v2x + v1y * v2y) / (n1 * n2)
        if cos_ang <= cos_thresh:
            corners.add(i)
    return corners


def send_ctrl_z_batch(count: int) -> None:
    """Gửi tổ hợp Ctrl+Z `count` lần để xoá nét viết tay thô trên trình soạn thảo thông thường."""
    n_undo = max(0, int(count))
    if n_undo == 0:
        return
    ensure_powerpoint_foreground()
    for _ in range(n_undo):
        win32_send_ctrl_key("Z")
        time.sleep(0.03)


def _sample_eraser_points(st: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Lấy 1-5 điểm rải đều theo chiều dài nét để cục tẩy Ctrl+E quét sạch không bỏ sót."""
    arr = np.asarray(st, dtype=np.float32).reshape(-1, 2)
    if len(arr) == 0:
        return []
    if len(arr) == 1:
        return [(float(arr[0, 0]), float(arr[0, 1]))]
    arc = _stroke_arc_length(arr)
    if arc < 18.0:
        mid = arr[len(arr) // 2]
        return [(float(mid[0]), float(mid[1]))]
    n_samples = 5 if arc > 140.0 else 3
    ratios = np.linspace(0.12, 0.88, n_samples)
    pts: List[Tuple[float, float]] = []
    for r in ratios:
        idx = int(np.clip(round(float(r) * (len(arr) - 1)), 0, len(arr) - 1))
        pts.append((float(arr[idx, 0]), float(arr[idx, 1])))
    return pts


def erase_raw_strokes_on_ppt(
    raw_strokes: Sequence[Sequence[Tuple[float, float]]],
    screen_w: int,
    screen_h: int,
    restore_pen: bool = True,
) -> str:
    """
    Xoá sạch toàn bộ các nét viết tay thô vừa vẽ (gồm cả chấm ngắn 1-2 điểm):
    - Trong PowerPoint Slide Show (`F5` / `screenClass`): bật Tẩy nét (`Ctrl+E`),
      quét `±8px` tại 1-5 điểm rải đều theo chiều dài mỗi nét rồi trả về Bút (`Ctrl+P`).
    - Ở cửa sổ chỉnh sửa thông thường: gửi `Ctrl+Z` theo số nét.
    """
    valid = [s for s in raw_strokes if len(s) >= 1]
    if not valid:
        return "none"

    if is_powerpoint_slideshow_active():
        ensure_powerpoint_foreground()
        time.sleep(0.03)
        win32_send_ctrl_key("E")
        time.sleep(0.11)
        with _high_res_win_timer(1):
            for st in valid:
                for sx, sy in _sample_eraser_points(st):
                    mx, my = _clip_pt(sx, sy, screen_w, screen_h, margin=12)
                    p0x, p0y = _clip_pt(mx - 8, my - 8, screen_w, screen_h, margin=8)
                    p1x, p1y = _clip_pt(mx + 8, my + 8, screen_w, screen_h, margin=8)
                    win32_mouse_down(p0x, p0y, screen_w, screen_h)
                    win32_drag_cursor(
                        p0x, p0y, p1x, p1y, screen_w, screen_h, steps=6, step_delay_sec=0.0015
                    )
                    time.sleep(0.010)
                    win32_mouse_up()
                    time.sleep(0.025)
        if restore_pen:
            win32_send_ctrl_key("P")
            time.sleep(0.10)
        return "slideshow_eraser"

    send_ctrl_z_batch(len(valid))
    time.sleep(0.04)
    return "ctrl_z"


def replay_win32_strokes(
    strokes_px: Sequence[Sequence[Tuple[int, int]]],
    screen_w: int,
    screen_h: int,
) -> None:
    """Tái vẽ các nét vector sạch lên PowerPoint bằng Win32 Mouse Drag có nghỉ tại góc nhọn."""
    if not strokes_px:
        return
    ensure_powerpoint_foreground()
    with _high_res_win_timer(1):
        for st in strokes_px:
            if len(st) < 2:
                continue
            corners = _get_corner_indices(st, angle_deg_thresh=32.0)
            x0, y0 = st[0]
            win32_mouse_down(x0, y0, screen_w, screen_h)
            time.sleep(0.025)
            prev_x, prev_y = x0, y0
            for idx in range(1, len(st)):
                nx, ny = st[idx]
                win32_drag_cursor(
                    prev_x,
                    prev_y,
                    nx,
                    ny,
                    screen_w,
                    screen_h,
                    steps=4,
                    step_delay_sec=0.0015,
                )
                time.sleep(0.022 if idx in corners else 0.003)
                prev_x, prev_y = nx, ny
            time.sleep(0.020)
            win32_mouse_up()
            time.sleep(0.035)


# ==============================================================================
# POWERPOINT COM VECTOR SHAPE ENGINE (SLIDE SHOW F5)
# ==============================================================================

def _get_slideshow_com_context(screen_w: int, screen_h: int):
    """Lấy `(ppt, slide, slide_w_pt, slide_h_pt, scale, offset_x, offset_y)` nếu F5 Slide Show đang chạy."""
    import win32com.client  # type: ignore

    ppt = win32com.client.GetActiveObject("PowerPoint.Application")
    if int(ppt.SlideShowWindows.Count) < 1:
        return None
    slide = ppt.SlideShowWindows(1).View.Slide
    pres = ppt.ActivePresentation
    slide_w_pt = float(pres.PageSetup.SlideWidth)
    slide_h_pt = float(pres.PageSetup.SlideHeight)
    scale = min(
        float(screen_w) / max(slide_w_pt, 1.0),
        float(screen_h) / max(slide_h_pt, 1.0),
    )
    offset_x = 0.5 * (float(screen_w) - slide_w_pt * scale)
    offset_y = 0.5 * (float(screen_h) - slide_h_pt * scale)
    return ppt, slide, slide_w_pt, slide_h_pt, scale, offset_x, offset_y


def _px_to_slide_pt(
    px: float,
    py: float,
    slide_w_pt: float,
    slide_h_pt: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> Tuple[float, float]:
    sx = float(np.clip((px - offset_x) / max(scale, 1e-6), 4.0, max(12.0, slide_w_pt - 4.0)))
    sy = float(np.clip((py - offset_y) / max(scale, 1e-6), 4.0, max(12.0, slide_h_pt - 4.0)))
    return sx, sy


def _refresh_slideshow_view(slide) -> None:
    """Ép PowerPoint Slide Show vẽ lại ngay lập tức mà không reset hiệu ứng hay mực Ink."""
    try:
        tmp = slide.Shapes.AddTextbox(1, -200.0, -200.0, 1.0, 1.0)
        tmp.Delete()
    except Exception:
        pass


def _style_ai_shape(
    shp,
    group_tag: str,
    color_bgr: int = _DEFAULT_PEN_COLOR_BGR,
    line_weight: float = 3.25,
    has_fill: bool = True,
) -> None:
    shp.Name = str(group_tag)
    if has_fill:
        try:
            shp.Fill.Visible = _MSO_FALSE
        except Exception:
            pass
    shp.Line.Visible = _MSO_TRUE
    shp.Line.ForeColor.RGB = int(color_bgr)
    shp.Line.Weight = float(line_weight)


def insert_ppt_com_shape(
    shape_name: str,
    raw_strokes: Sequence[Sequence[Tuple[float, float]]],
    screen_w: int,
    screen_h: int,
    group_tag: Optional[str] = None,
    color_bgr: int = _DEFAULT_PEN_COLOR_BGR,
    line_weight: float = 3.25,
) -> Tuple[bool, Optional[str], str]:
    """
    Chèn Vector Shape sắc nét trực tiếp vào Slide đang trình chiếu (`SlideShowWindows(1)`)
    qua PowerPoint COM. Tự động trả về `(False, None, lý do)` nếu không ở Slide Show F5 hoặc thiếu COM.
    """
    valid = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in raw_strokes if len(s) >= 2]
    if not valid:
        return False, None, "Khong co net ve"

    try:
        ctx = _get_slideshow_com_context(screen_w, screen_h)
        if ctx is None:
            return False, None, "PowerPoint chua mo Slide Show (F5)"
        _ppt, slide, _slide_w_pt, _slide_h_pt, scale, offset_x, offset_y = ctx
    except Exception as e:
        return False, None, f"COM Fallback ({e})"

    all_pts = np.vstack(valid)
    x0, y0 = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
    x1, y1 = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw = max(x1 - x0, 24.0)
    bh = max(y1 - y0, 24.0)

    def to_pt(px: float, py: float) -> Tuple[float, float]:
        return _px_to_slide_pt(px, py, slide_w_pt, slide_h_pt, scale, offset_x, offset_y)

    tag = group_tag or f"{PPT_AI_SHAPE_PREFIX}{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    created: List[object] = []

    try:
        if shape_name == "line":
            pa, pb = _fit_line_pca(valid, all_pts)
            p1x, p1y = to_pt(pa[0], pa[1])
            p2x, p2y = to_pt(pb[0], pb[1])
            shp = slide.Shapes.AddLine(p1x, p1y, p2x, p2y)
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=False)
            created.append(shp)

        elif shape_name == "arrow":
            tail, tip = _infer_arrow_tail_tip(valid)
            p1x, p1y = to_pt(tail[0], tail[1])
            p2x, p2y = to_pt(tip[0], tip[1])
            shp = slide.Shapes.AddLine(p1x, p1y, p2x, p2y)
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=False)
            shp.Line.EndArrowheadStyle = _MSO_ARROWHEAD_TRIANGLE
            shp.Line.EndArrowheadWidth = _MSO_ARROWHEAD_MEDIUM
            shp.Line.EndArrowheadLength = _MSO_ARROWHEAD_MEDIUM
            created.append(shp)

        elif shape_name == "circle":
            if 0.78 <= (bw / max(bh, 1.0)) <= 1.28:
                d = 0.5 * (bw + bh)
                lx0, ty0, rx1, by1 = cx - 0.5 * d, cy - 0.5 * d, cx + 0.5 * d, cy + 0.5 * d
            else:
                lx0, ty0, rx1, by1 = x0, y0, x1, y1
            l_pt, t_pt = to_pt(lx0, ty0)
            r_pt, b_pt = to_pt(rx1, by1)
            shp = slide.Shapes.AddShape(
                _MSO_SHAPE_OVAL, l_pt, t_pt, max(12.0, r_pt - l_pt), max(12.0, b_pt - t_pt)
            )
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=True)
            created.append(shp)

        elif shape_name == "rectangle":
            l_pt, t_pt = to_pt(x0, y0)
            r_pt, b_pt = to_pt(x1, y1)
            shp = slide.Shapes.AddShape(
                _MSO_SHAPE_RECTANGLE, l_pt, t_pt, max(12.0, r_pt - l_pt), max(12.0, b_pt - t_pt)
            )
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=True)
            created.append(shp)

        elif shape_name == "triangle":
            tri = _fit_triangle_vertices(all_pts, x0, y0, x1, y1)
            pts_pt = [to_pt(vx, vy) for vx, vy in tri]
            fb = slide.Shapes.BuildFreeform(_MSO_EDITING_CORNER, pts_pt[0][0], pts_pt[0][1])
            fb.AddNodes(_MSO_SEGMENT_LINE, _MSO_EDITING_AUTO, pts_pt[1][0], pts_pt[1][1])
            fb.AddNodes(_MSO_SEGMENT_LINE, _MSO_EDITING_AUTO, pts_pt[2][0], pts_pt[2][1])
            fb.AddNodes(_MSO_SEGMENT_LINE, _MSO_EDITING_AUTO, pts_pt[0][0], pts_pt[0][1])
            shp = fb.ConvertToShape()
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=True)
            created.append(shp)

        elif shape_name == "star":
            side = min(bw, bh)
            l_pt, t_pt = to_pt(cx - 0.5 * side, cy - 0.5 * side)
            r_pt, b_pt = to_pt(cx + 0.5 * side, cy + 0.5 * side)
            s_pt = max(16.0, min(r_pt - l_pt, b_pt - t_pt))
            shp = slide.Shapes.AddShape(_MSO_SHAPE_5POINT_STAR, l_pt, t_pt, s_pt, s_pt)
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=True)
            created.append(shp)

        elif shape_name == "checkmark":
            p0 = to_pt(x0 + 0.08 * bw, y0 + 0.52 * bh)
            pv = to_pt(x0 + 0.36 * bw, y1 - 0.06 * bh)
            p2 = to_pt(x1 - 0.05 * bw, y0 + 0.08 * bh)
            fb = slide.Shapes.BuildFreeform(_MSO_EDITING_CORNER, p0[0], p0[1])
            fb.AddNodes(_MSO_SEGMENT_LINE, _MSO_EDITING_AUTO, pv[0], pv[1])
            fb.AddNodes(_MSO_SEGMENT_LINE, _MSO_EDITING_AUTO, p2[0], p2[1])
            shp = fb.ConvertToShape()
            _style_ai_shape(shp, tag, color_bgr, line_weight, has_fill=False)
            created.append(shp)

        elif shape_name == "cross":
            tl = to_pt(x0, y0)
            br = to_pt(x1, y1)
            tr = to_pt(x1, y0)
            bl = to_pt(x0, y1)
            s1 = slide.Shapes.AddLine(tl[0], tl[1], br[0], br[1])
            s2 = slide.Shapes.AddLine(tr[0], tr[1], bl[0], bl[1])
            _style_ai_shape(s1, tag, color_bgr, line_weight, has_fill=False)
            _style_ai_shape(s2, tag, color_bgr, line_weight, has_fill=False)
            created.extend([s1, s2])

        else:
            return False, None, f"Khong ho tro COM shape: {shape_name}"

        _refresh_slideshow_view(slide)
        return True, tag, f"COM Shape '{shape_name}' ({tag})"
    except Exception as e:
        for shp in created:
            try:
                shp.Delete()
            except Exception:
                pass
        return False, None, f"COM Fallback ({e})"


def delete_ppt_com_shapes_by_tag(tag: Optional[str]) -> int:
    """Xoá các Vector Shape AI theo `tag` trên Slide đang trình chiếu (dùng khi sửa Top-3)."""
    if not tag:
        return 0
    try:
        ctx = _get_slideshow_com_context(1920, 1080)
        if ctx is None:
            return 0
        _ppt, slide, *_rest = ctx
        deleted = 0
        for idx in range(int(slide.Shapes.Count), 0, -1):
            shp = slide.Shapes(idx)
            if str(shp.Name) == str(tag):
                shp.Delete()
                deleted += 1
        if deleted > 0:
            _refresh_slideshow_view(slide)
        return deleted
    except Exception:
        return 0


def erase_ppt_com_shapes_at_cursor(
    scr_x: int,
    scr_y: int,
    screen_w: int,
    screen_h: int,
    radius_px: float = 30.0,
) -> int:
    """Xoá các Vector Shape AI (`__AI_SHAPE_...`) khi cục tẩy 2 ngón (`point_2f`) quét qua."""
    try:
        ctx = _get_slideshow_com_context(screen_w, screen_h)
        if ctx is None:
            return 0
        _ppt, slide, _slide_w_pt, _slide_h_pt, scale, offset_x, offset_y = ctx
        pt_x = (float(scr_x) - offset_x) / max(scale, 1e-6)
        pt_y = (float(scr_y) - offset_y) / max(scale, 1e-6)
        pad_pt = float(radius_px) / max(scale, 1e-6)

        hit_tags: Set[str] = set()
        n_shapes = int(slide.Shapes.Count)
        for idx in range(1, n_shapes + 1):
            shp = slide.Shapes(idx)
            name = str(shp.Name)
            if not name.startswith(PPT_AI_SHAPE_PREFIX):
                continue
            left = float(shp.Left)
            top = float(shp.Top)
            width = max(float(shp.Width), 6.0)
            height = max(float(shp.Height), 6.0)
            if (left - pad_pt) <= pt_x <= (left + width + pad_pt) and (
                top - pad_pt
            ) <= pt_y <= (top + height + pad_pt):
                hit_tags.add(name)

        if not hit_tags:
            return 0

        deleted = 0
        for idx in range(int(slide.Shapes.Count), 0, -1):
            shp = slide.Shapes(idx)
            if str(shp.Name) in hit_tags:
                shp.Delete()
                deleted += 1
        if deleted > 0:
            _refresh_slideshow_view(slide)
        return deleted
    except Exception:
        return 0
