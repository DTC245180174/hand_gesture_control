"""
MODULE VẼ BẢNG ĐIỀU KHIỂN THỊ GIÁC (HUD RENDERER)
=================================================
Hiển thị trực tiếp xác suất 8 lớp từ mô hình học sâu, trạng thái điều khiển,
vùng active box và thông số FPS trên khung hình OpenCV.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from common.config import ACTION_DISPLAY, CLASS_NAMES
from step7_deployment.controller import MARGIN_X, MARGIN_Y


def _screen_to_hud_pt(
    px: Tuple[float, float], screen_w: int, screen_h: int, hud_w: int, hud_h: int
) -> Tuple[int, int]:
    nx = float(px[0]) / max(float(screen_w - 1), 1.0)
    ny = float(px[1]) / max(float(screen_h - 1), 1.0)
    ix = MARGIN_X + nx * (1.0 - 2.0 * MARGIN_X)
    iy = MARGIN_Y + ny * (1.0 - 2.0 * MARGIN_Y)
    return int(np.clip(ix * hud_w, 0, hud_w - 1)), int(np.clip(iy * hud_h, 0, hud_h - 1))


def _draw_air_ink_hud(canvas: np.ndarray, controller: object, w: int, h: int) -> None:
    """Vẽ nét Air-Ink trực tiếp trên HUD và ô xem trước 64x64 của Model 3."""
    screen_w = int(getattr(controller, "screen_w", 1920))
    screen_h = int(getattr(controller, "screen_h", 1080))
    strokes = list(getattr(controller, "ink_strokes", []))
    cur_st = list(getattr(controller, "ink_current_stroke", []))

    for st in strokes:
        if len(st) >= 2:
            pts = np.array([_screen_to_hud_pt(p, screen_w, screen_h, w, h) for p in st], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (255, 220, 40), 3, cv2.LINE_AA)
    if len(cur_st) >= 2:
        pts = np.array([_screen_to_hud_pt(p, screen_w, screen_h, w, h) for p in cur_st], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 180), 3, cv2.LINE_AA)

    if getattr(controller, "is_unified_laser_mode", lambda: False)():
        sub = str(getattr(controller, "unified_substate", "laser"))
        mode_title = "LASER + AI NAN HINH"
        mode_col = (0, 230, 255) if sub == "pen_armed" else (80, 120, 255)
        sub_hint = (
            f"Strokes: {len(strokes) + (1 if cur_st else 0)} | Dung ~1s -> Vector Shape"
            if sub == "pen_armed"
            else "Laser san sang | Giu yen 0.7s de bat But AI"
        )
    else:
        mode_title, mode_col = ("AI NAN HINH (9 Lop)", (0, 230, 255))
        sub_hint = f"Strokes: {len(strokes) + (1 if cur_st else 0)} | Dung ~1s de nan hinh"

    # Khung Top-Right: Preview 64x64 + Top-3 Model 3
    px0, py0 = max(350, w - 315), 95
    px1, py1 = w - 12, 265
    if px1 - px0 >= 250:
        panel = canvas.copy()
        cv2.rectangle(panel, (px0, py0), (px1, py1), (22, 26, 36), -1)
        cv2.addWeighted(panel, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), mode_col, 1, cv2.LINE_AA)

        cv2.putText(canvas, mode_title, (px0 + 10, py0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, sub_hint, (px0 + 10, py0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)

        img64 = getattr(controller, "last_ink_img64", None)
        thumb_sz = 92
        tx0, ty0 = px0 + 12, py0 + 54
        if img64 is not None and img64.shape == (64, 64):
            thumb = cv2.resize(img64, (thumb_sz, thumb_sz), interpolation=cv2.INTER_NEAREST)
            canvas[ty0 : ty0 + thumb_sz, tx0 : tx0 + thumb_sz] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        else:
            cv2.rectangle(canvas, (tx0, ty0), (tx0 + thumb_sz, ty0 + thumb_sz), (10, 10, 14), -1)
            cv2.putText(canvas, "64x64", (tx0 + 22, ty0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 130), 1)
        cv2.rectangle(canvas, (tx0, ty0), (tx0 + thumb_sz, ty0 + thumb_sz), (130, 150, 180), 1)

        pred = str(getattr(controller, "last_ink_pred", ""))
        p_conf = float(getattr(controller, "last_ink_conf", 0.0))
        acc = bool(getattr(controller, "last_ink_accepted", False))
        rx0 = tx0 + thumb_sz + 14
        if pred:
            p_col = (50, 240, 110) if acc else (0, 190, 255)
            cv2.putText(canvas, f"Pred: {pred}", (rx0, ty0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, p_col, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Conf: {p_conf*100:.1f}%", (rx0, ty0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
            top3 = list(getattr(controller, "last_top3_candidates", []))
            for idx, (cand, cprob) in enumerate(top3[:3], start=1):
                cv2.putText(
                    canvas,
                    f"[{idx}] {cand}: {cprob*100:.0f}%",
                    (rx0, ty0 + 66 + (idx - 1) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (195, 220, 245) if idx == 1 else (160, 175, 190),
                    1,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(canvas, "Viet 1 net hoac", (rx0, ty0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)
            cv2.putText(canvas, "nhieu net -> dung", (rx0, ty0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)
            cv2.putText(canvas, "1.05s de AI chot", (rx0, ty0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)

def draw_hud_overlay(
    frame: np.ndarray,
    state: Dict[str, object],
    controller: object,
    fps: float,
    index_tip_px: Optional[Tuple[int, int]] = None,
    in_deadzone: bool = False,
) -> np.ndarray:
    """Vẽ bảng điều khiển HUD hiển thị phân phối xác suất và trạng thái của Model."""
    h, w, _ = frame.shape
    canvas = frame.copy()

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (20, 20, 20), -1)
    cv2.rectangle(overlay, (10, 95), (335, 410), (25, 25, 25), -1)

    # Vẽ vùng nghỉ hạ tay 15% mép dưới nếu tính năng bottom_deadzone đang bật
    if getattr(controller, "bottom_deadzone", False):
        dz_y = int(h * 0.85)
        dz_color = (0, 140, 255) if in_deadzone else (45, 45, 55)
        cv2.rectangle(overlay, (0, dz_y), (w, h), dz_color, -1)

    cv2.addWeighted(overlay, 0.74, canvas, 0.26, 0, canvas)

    if getattr(controller, "bottom_deadzone", False):
        dz_y = int(h * 0.85)
        line_col = (0, 180, 255) if in_deadzone else (130, 130, 150)
        cv2.line(canvas, (0, dz_y), (w, dz_y), line_col, 2, cv2.LINE_AA)
        dz_text = (
            "VUNG NGHI HA TAY (DEAD-ZONE 15%) -> DANG NGHI (KHONG NHAN LENH)"
            if in_deadzone
            else "VUNG NGHI HA TAY 15% (Dua co tay len tren vach nay de dieu khien)"
        )
        cv2.putText(canvas, dz_text, (16, dz_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, line_col, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (3, 3), (w - 4, h - 4), (60, 200, 60), 2)

    confirmed_cls = int(state["confirmed_class"])
    raw_cls = int(state["raw_class"])
    conf_val = float(state["confidence"])
    peak_t = int(state.get("peak_attn_frame", 0))

    if getattr(controller, "is_ai_ink_active", lambda: False)():
        _draw_air_ink_hud(canvas, controller, w, h)

    if index_tip_px is not None and confirmed_cls == 3:
        is_unified = getattr(controller, "is_unified_laser_mode", lambda: False)()
        unified_sub = str(getattr(controller, "unified_substate", "laser"))
        if is_unified and unified_sub == "laser":
            d_prog = float(getattr(controller, "unified_dwell_progress", 0.0))
            cv2.circle(canvas, index_tip_px, 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 6, (80, 80, 255), -1, cv2.LINE_AA)
            if d_prog > 0.02:
                cv2.ellipse(
                    canvas,
                    index_tip_px,
                    (20, 20),
                    -90,
                    0,
                    int(round(360.0 * d_prog)),
                    (0, 165, 255),
                    3,
                    cv2.LINE_AA,
                )
                l_tag = "MO BUT..."
            else:
                l_tag = "LASER (Giu yen -> But)"
            cv2.putText(
                canvas,
                l_tag,
                (index_tip_px[0] + 24, index_tip_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255) if d_prog > 0.02 else (60, 80, 255),
                2,
                cv2.LINE_AA,
            )
        elif getattr(controller, "is_ai_ink_active", lambda: False)():
            pen_dn = bool(getattr(controller, "ink_pen_down", False))
            prog = float(getattr(controller, "ink_commit_progress", 0.0))
            ring_col = (0, 255, 180) if pen_dn else (0, 210, 255)
            cv2.circle(canvas, index_tip_px, 14, ring_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if prog > 0.02:
                cv2.ellipse(
                    canvas,
                    index_tip_px,
                    (20, 20),
                    -90,
                    0,
                    int(round(360.0 * prog)),
                    (0, 140, 255),
                    3,
                    cv2.LINE_AA,
                )
            tag = "PEN DOWN" if pen_dn else ("CHOT AI..." if prog > 0.02 else ("BUT AI SAN SANG" if is_unified else "PEN UP"))
            cv2.putText(
                canvas, tag, (index_tip_px[0] + 28, index_tip_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ring_col, 2, cv2.LINE_AA
            )
        elif is_unified and unified_sub == "pen_armed":
            drawing_now = bool(getattr(controller, "mouse_down_active", False))
            f_prog = float(getattr(controller, "unified_pen_finish_progress", 0.0))
            cv2.circle(canvas, index_tip_px, 14, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 4, (0, 255, 255), -1, cv2.LINE_AA)
            if f_prog > 0.02:
                cv2.ellipse(
                    canvas,
                    index_tip_px,
                    (20, 20),
                    -90,
                    0,
                    int(round(360.0 * f_prog)),
                    (0, 130, 255),
                    3,
                    cv2.LINE_AA,
                )
            p_tag = "DANG VE..." if drawing_now else "BUT TAY SAN SANG"
            cv2.putText(
                canvas, p_tag, (index_tip_px[0] + 24, index_tip_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 2, cv2.LINE_AA
            )
        elif getattr(controller, "pen_draw_click", True):
            cv2.circle(canvas, index_tip_px, 12, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            # Chế độ Tia Laser Đỏ (Ctrl+L)
            cv2.circle(canvas, index_tip_px, 14, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, index_tip_px, 6, (80, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas, "LASER (Ctrl+L)", (index_tip_px[0] + 18, index_tip_px[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 80, 255), 2, cv2.LINE_AA
            )
    elif index_tip_px is not None and confirmed_cls == 4:
        cv2.circle(canvas, index_tip_px, 22, (220, 80, 220), 2, cv2.LINE_AA)
        cv2.circle(canvas, index_tip_px, 6, (255, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "ERASER", (index_tip_px[0] + 26, index_tip_px[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 120, 255), 2, cv2.LINE_AA
        )
    elif index_tip_px is not None and (confirmed_cls == 7 or getattr(controller, "last_triggered_class", 0) == 7):
        cv2.circle(canvas, index_tip_px, 20, (0, 235, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, index_tip_px, 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "LEFT CLICK!", (index_tip_px[0] + 24, index_tip_px[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 235, 255), 2, cv2.LINE_AA
        )

    if not controller.live_ppt:
        mode_text = "PAUSED (DRY-RUN)"
        mode_color = (0, 180, 255)
    elif getattr(controller, "ppt_only_lock", False) and not getattr(controller, "ppt_window_active", True):
        mode_text = "STANDBY (Mo PowerPoint de dieu khien)"
        mode_color = (0, 215, 255)
    else:
        mode_text = "LIVE PPT"
        mode_color = (50, 230, 50)

    cv2.putText(
        canvas,
        f"[{mode_text}] Model: {controller.model_info} | FPS: {fps:.1f} | {state['status']}",
        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, mode_color, 2
    )

    disp_label, disp_color = ACTION_DISPLAY[confirmed_cls]
    cv2.putText(
        canvas,
        f"Model Pred: [{raw_cls}] {CLASS_NAMES[raw_cls]} ({conf_val*100:.1f}%, Attn t={peak_t}) -> Confirmed: {disp_label}",
        (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, disp_color, 2
    )

    cv2.putText(
        canvas,
        f"Last Triggered Command: {controller.last_triggered_action}",
        (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1
    )

    cv2.putText(
        canvas, "TRAINED MODEL PROBS", (20, 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1
    )

    probs = state["probs"]
    for cid, cname in enumerate(CLASS_NAMES):
        y_pos = 146 + cid * 30
        p_val = float(probs[cid]) if len(probs) > cid else 0.0
        bar_w = int(p_val * 160)

        cv2.putText(canvas, f"[{cid}] {cname:<11}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1)
        cv2.rectangle(canvas, (135, y_pos - 12), (295, y_pos + 4), (60, 60, 60), -1)
        bar_color = (50, 220, 50) if p_val >= controller.conf_threshold else (100, 160, 240)
        if bar_w > 0:
            cv2.rectangle(canvas, (135, y_pos - 12), (135 + bar_w, y_pos + 4), bar_color, -1)
        cv2.putText(canvas, f"{p_val*100:>3.0f}%", (298, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        th_x = 135 + int(controller.conf_threshold * 160)
        cv2.line(canvas, (th_x, y_pos - 14), (th_x, y_pos + 6), (0, 80, 255), 1)

    cv2.putText(
        canvas,
        "[Q/F9] Quit | [P/F8] PiP | [SPACE] Pause | [D] Doi 1 Ngon | [M/F7] AI <-> But | [1-3] Doi Top-3 | [C] Xoa",
        (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1
    )
    return canvas

