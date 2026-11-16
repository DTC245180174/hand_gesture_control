"""
MODULE QUẢN LÝ DUAL DEEP LEARNING & ĐIỀU KHIỂN CỬ CHỈ THỜI GIAN THỰC (OPTION B)
================================================================================
Kiến trúc 2 tầng thuần Học Sâu huấn luyện từ đầu (không luật if-else đè Model):
1. Model 1 (`StaticPoseNet`, 1 frame): Chuyên trách nhận diện tức thì tư thế tĩnh
   `point_1f` (Bút/Laser) và `point_2f` (Cục tẩy) bất biến góc xoay & tốc độ lia tay,
   tự động nhả bút mượt mà ngay khi hạ ngón tay (`other_pose`).
2. Model 2 (`HybridConvBiGRUAttention`, 30 frames): Chuyên trách các cử chỉ động
   `swipe_left`, `swipe_right`, `zoom_in`, `zoom_out`, `click` kèm cổng an toàn
   Confidence Threshold, Majority Voting và Cooldown chống dội khi thu tay về.
"""

from __future__ import annotations
import ctypes
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from common.config import (
    ACTION_DISPLAY,
    AIR_INK_CLASSES,
    AIR_INK_IMG_SIZE,
    AIR_INK_SHAPE_IDS,
    CLASS_NAMES,
    DISCRETE_CLASSES,
    SEQ_LEN,
)
from common.win32_input import (
    CursorRingOverlay,
    OneEuroFilter2D,
    ensure_powerpoint_foreground,
    find_powerpoint_window,
    play_action_beep,
    play_dwell_beep,
    win32_drag_cursor,
    win32_left_click,
    win32_mouse_down,
    win32_mouse_up,
    win32_move_cursor,
    win32_send_ctrl_key,
    win32_send_ppt_slide_key,
)
from step2_clean_diagnose.feature_extractor import (
    extract_66d_dynamic_features,
    extract_static_pose_features,
    temporal_resample,
)
from step2_clean_diagnose.ink_rasterizer import rasterize_strokes_to_64x64
from step7_deployment.smart_shape import (
    delete_ppt_com_shapes_by_tag,
    erase_ppt_com_shapes_at_cursor,
    erase_raw_strokes_on_ppt,
    generate_clean_shape_strokes,
    insert_ppt_com_shape,
    replay_win32_strokes,
)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
except Exception:
    pyautogui = None

_USER32 = ctypes.windll.user32 if os.name == "nt" else None

# Vùng hoạt động Active Box nhẹ nhàng hơn [0.10..0.90] để dễ chạm mép Slide mà không mỏi tay
MARGIN_X, MARGIN_Y = 0.10, 0.10


_VALID_POINT_1F_MODES = ("laser_ai", "laser_pen", "ai_shapes", "pen", "laser")
_AI_INK_MODES = ("laser_ai", "ai_shapes")
_UNIFIED_LASER_MODES = ("laser_ai", "laser_pen")


class RealTimeGestureController:
    """Bộ điều khiển cử chỉ thời gian thực Triple Deep Learning (StaticPose + DynamicSeq + AirInkResCNN)."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        model_info: str = "HybridConvBiGRUAttention",
        conf_threshold: float = 0.85,
        vote_window: int = 5,
        vote_required: int = 3,
        cooldown_sec: float = 2.0,
        live_ppt: bool = True,
        pen_draw_click: bool = True,
        ppt_only_lock: bool = False,
        pointer_smoothing: float = 0.60,
        audio_beep: bool = True,
        bottom_deadzone: bool = True,
        pose_model: Optional[torch.nn.Module] = None,
        air_ink_model: Optional[torch.nn.Module] = None,
        point_1f_mode: Optional[str] = None,
        use_ppt_com_shapes: bool = True,
        enable_cursor_overlay: bool = True,
    ) -> None:
        self.model = model
        self.pose_model = pose_model
        self.air_ink_model = air_ink_model
        self.device = device
        self.model_info = model_info
        self.conf_threshold = conf_threshold
        self.vote_window = vote_window
        self.vote_required = vote_required
        self.cooldown_sec = cooldown_sec
        self.live_ppt = live_ppt
        self.ppt_only_lock = ppt_only_lock
        self.ppt_window_active = True
        self.pointer_smoothing = float(np.clip(pointer_smoothing, 0.10, 0.95))
        self.audio_beep = audio_beep
        self.bottom_deadzone = bottom_deadzone

        if point_1f_mode is not None:
            init_mode = point_1f_mode.lower().strip()
        elif not pen_draw_click:
            init_mode = "laser"
        elif self.air_ink_model is not None:
            init_mode = "laser_ai"
        else:
            init_mode = "laser_pen"

        if init_mode not in _VALID_POINT_1F_MODES:
            init_mode = (
                "laser_ai"
                if (pen_draw_click and self.air_ink_model is not None)
                else ("laser_pen" if pen_draw_click else "laser")
            )
        if init_mode in _AI_INK_MODES and self.air_ink_model is None:
            init_mode = "laser_pen" if init_mode == "laser_ai" else "pen"
        self.point_1f_mode: str = init_mode
        self.last_ai_mode: str = "laser_ai"
        self.pen_draw_click = self.point_1f_mode != "laser"

        self.landmark_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        self.vote_buffer: Deque[int] = deque(maxlen=vote_window)

        self.missing_streak = 0
        self.last_valid_pts = np.zeros(63, dtype=np.float32)
        self.last_trigger_time = -10.0
        self.last_triggered_action = "Chua co lenh"
        self.last_triggered_class = 0
        self.dynamic_guard_until = -10.0

        # Trạng thái Hysteresis thuần xác suất cho Model 1 (StaticPoseNet)
        self.active_pose_cls = 0          # 0 (none), 3 (point_1f), 4 (point_2f)
        self.pose_enter_cand = 0
        self.pose_enter_streak = 0
        self.pose_switch_cand = 0
        self.pose_switch_streak = 0
        self.pose_exit_streak = 0
        self.pen_paused = False

        # Trạng thái phân đoạn nét AI Nắn Hình (Model 3: Delayed Pen-Down -> Dwell Pen-Up -> Commit)
        self.use_ppt_com_shapes = bool(use_ppt_com_shapes)
        self.ink_strokes: List[List[Tuple[float, float]]] = []
        self.ink_ppt_drawn_flags: List[bool] = []
        self.ink_current_stroke: List[Tuple[float, float]] = []
        self.ink_anchor_px: Optional[Tuple[float, float]] = None
        self.ink_last_move_t: float = 0.0
        self.ink_pen_down: bool = False
        self.ink_cur_ppt_drawn: bool = False
        self.ink_stroke_arc_len: float = 0.0
        self.ink_dwell_pen_up_sec: float = 0.42
        self.ink_commit_sec: float = 1.05
        self.ink_commit_progress: float = 0.0
        self.last_ink_img64: Optional[np.ndarray] = None
        self.last_ink_pred: str = ""
        self.last_ink_conf: float = 0.0
        self.last_ink_accepted: bool = False
        self.last_ink_kind: str = ""  # "shape" | "freeform" | ""
        self.last_top3_candidates: List[Tuple[str, float]] = []
        self.last_top3_meta: List[Tuple[int, str, str, bool]] = []
        self._last_ppt_ink_strokes: List[List[Tuple[float, float]]] = []
        self._last_shape_raw_strokes: List[List[Tuple[float, float]]] = []
        self._last_ppt_com_tag: Optional[str] = None
        self._last_com_erase_check_t: float = -10.0

        # Trạng thái tích hợp Laser <-> Bút bằng Dwell (Phương án C: giữ yên ~0.72s để mở bút)
        self.enable_cursor_overlay: bool = bool(enable_cursor_overlay)
        self._cursor_overlay: Optional[CursorRingOverlay] = (
            CursorRingOverlay() if (self.enable_cursor_overlay and self.live_ppt) else None
        )
        self.unified_substate: str = "laser"  # "laser" | "pen_armed"
        self.unified_dwell_anchor_px: Optional[Tuple[float, float]] = None
        self.unified_dwell_start_t: float = 0.0
        self.unified_dwell_progress: float = 0.0
        self.unified_dwell_grace_sec: float = 0.22
        self.unified_dwell_arm_sec: float = 0.72
        self.unified_pen_idle_timeout_sec: float = 2.60
        self.unified_laser_pen_finish_sec: float = 0.65
        self.unified_dwell_guard_until: float = -10.0
        self.unified_pen_armed_t: float = 0.0
        self.unified_pen_finish_progress: float = 0.0
        self.unified_pen_stroke_pts: List[Tuple[float, float]] = []
        self.unified_pen_arc_len: float = 0.0

        self.smoothed_cursor: Optional[Tuple[float, float]] = None
        self.last_drag_px: Optional[Tuple[int, int]] = None
        self.active_tool_mode: Optional[str] = None  # "pen" (Ctrl+P), "laser" (Ctrl+L), "eraser" (Ctrl+E)
        self.tool_ready_time: float = 0.0
        min_cf = max(0.35, 2.20 - 1.85 * self.pointer_smoothing)
        beta_val = max(0.005, 0.030 * (1.0 - 0.65 * self.pointer_smoothing))
        self.cursor_filter = OneEuroFilter2D(min_cutoff=min_cf, beta=beta_val, d_cutoff=1.0)
        self.mouse_down_active = False

        if _USER32 is not None:
            self.screen_w = int(_USER32.GetSystemMetrics(0))
            self.screen_h = int(_USER32.GetSystemMetrics(1))
        elif pyautogui is not None:
            try:
                self.screen_w, self.screen_h = pyautogui.size()
            except Exception:
                self.screen_w, self.screen_h = (1920, 1080)
        else:
            self.screen_w, self.screen_h = (1920, 1080)

    def is_ai_ink_active(self) -> bool:
        return self.air_ink_model is not None and self.point_1f_mode in _AI_INK_MODES

    def is_unified_laser_mode(self) -> bool:
        return self.point_1f_mode in _UNIFIED_LASER_MODES

    def _update_cursor_overlay(
        self,
        scr_x: int,
        scr_y: int,
        mode: str = "hidden",
        progress: float = 0.0,
        visible: bool = False,
    ) -> None:
        if self._cursor_overlay is None:
            return
        if not visible or not self.can_send_to_ppt():
            self._cursor_overlay.hide()
            return
        self._cursor_overlay.update(scr_x, scr_y, mode=mode, progress=progress, visible=True)

    def _reset_unified_substate(self) -> None:
        self.unified_substate = "laser"
        self.unified_dwell_anchor_px = None
        self.unified_dwell_start_t = 0.0
        self.unified_dwell_progress = 0.0
        self.unified_pen_armed_t = 0.0
        self.unified_pen_finish_progress = 0.0
        self.unified_pen_stroke_pts.clear()
        self.unified_pen_arc_len = 0.0
        if self._cursor_overlay is not None:
            self._cursor_overlay.hide()

    def _enter_unified_laser_state(
        self, can_ppt: bool, now_t: float, guard_sec: float = 0.65
    ) -> None:
        """Đưa chế độ tích hợp về rọi tia Laser (`Ctrl+L`) an toàn sau khi vẽ xong."""
        if self.mouse_down_active and can_ppt:
            try:
                win32_mouse_up()
            except Exception:
                pass
        self.mouse_down_active = False
        self.ink_pen_down = False
        self.last_drag_px = None
        self.unified_substate = "laser"
        self.unified_dwell_anchor_px = None
        self.unified_dwell_start_t = now_t
        self.unified_dwell_progress = 0.0
        self.unified_pen_armed_t = 0.0
        self.unified_pen_finish_progress = 0.0
        self.unified_pen_stroke_pts.clear()
        self.unified_pen_arc_len = 0.0
        self.unified_dwell_guard_until = now_t + float(guard_sec)
        if can_ppt and self.active_tool_mode != "laser":
            win32_send_ctrl_key("L")
            self.active_tool_mode = "laser"
            self.tool_ready_time = now_t + 0.10
        if self._cursor_overlay is not None:
            self._cursor_overlay.hide()

    def close(self) -> None:
        """Giải phóng nút chuột, trả con trỏ PowerPoint và đóng cửa sổ vòng tròn Laser."""
        self._release_pen_if_needed(reset_tool=True, send_arrow=False)
        if self._cursor_overlay is not None:
            self._cursor_overlay.close()
            self._cursor_overlay = None

    def _reset_ink_cluster(self) -> None:
        self.ink_strokes.clear()
        self.ink_ppt_drawn_flags.clear()
        self.ink_current_stroke.clear()
        self.ink_anchor_px = None
        self.ink_last_move_t = 0.0
        self.ink_pen_down = False
        self.ink_cur_ppt_drawn = False
        self.ink_stroke_arc_len = 0.0
        self.ink_commit_progress = 0.0

    def _render_shape_onto_ppt(
        self,
        shape_name: str,
        raw_strokes: List[List[Tuple[float, float]]],
    ) -> str:
        """Ưu tiên chèn PowerPoint COM Vector Shape nếu đang trình chiếu F5; fallback sang Ink Replay."""
        if self.use_ppt_com_shapes:
            ok, shape_tag, _msg = insert_ppt_com_shape(
                shape_name, raw_strokes, self.screen_w, self.screen_h
            )
            if ok and shape_tag:
                self._last_ppt_com_tag = shape_tag
                self._last_ppt_ink_strokes.clear()
                return "com_shape"

        clean_px = generate_clean_shape_strokes(
            shape_name, raw_strokes, self.screen_w, self.screen_h
        )
        replay_win32_strokes(clean_px, self.screen_w, self.screen_h)
        self._last_ppt_com_tag = None
        self._last_ppt_ink_strokes = (
            [[(float(px), float(py)) for px, py in st] for st in clean_px] if clean_px else []
        )
        return "ink_replay"

    def set_point_1f_mode(self, mode: str) -> str:
        m = mode.lower().strip()
        if m not in _VALID_POINT_1F_MODES:
            m = "laser_ai" if self.air_ink_model is not None else "laser_pen"
        if m in _AI_INK_MODES and self.air_ink_model is None:
            print("[!] Chua co best_air_ink_model.pt — chuyen sang che do Pen khong nan hinh.")
            m = "laser_pen" if m == "laser_ai" else "pen"
        if m != self.point_1f_mode:
            self._reset_ink_cluster()
            self._reset_unified_substate()
            self._release_pen_if_needed(reset_tool=True, send_arrow=False)
            self.point_1f_mode = m
            self.pen_draw_click = m != "laser"
        return self.point_1f_mode

    def cycle_point_1f_mode(self) -> str:
        """Phím [D]: Xoay vòng Laser+AI -> Laser+Bút tay -> Chỉ AI Hình -> Chỉ Bút -> Chỉ Laser."""
        order = (
            ("laser_ai", "laser_pen", "ai_shapes", "pen", "laser")
            if self.air_ink_model is not None
            else ("laser_pen", "pen", "laser")
        )
        if self.point_1f_mode in order:
            nxt = order[(order.index(self.point_1f_mode) + 1) % len(order)]
        else:
            nxt = order[0]
        return self.set_point_1f_mode(nxt)

    def cycle_ai_submode(self) -> str:
        """Phím [M]/[F7]: Bật / Tắt nhanh AI Nắn Hình trong chế độ hiện tại."""
        if self.air_ink_model is None:
            print("[!] Chua nap duoc Model 3 (best_air_ink_model.pt).")
            return self.point_1f_mode
        if self.point_1f_mode == "laser_ai":
            return self.set_point_1f_mode("laser_pen")
        if self.point_1f_mode == "laser_pen":
            return self.set_point_1f_mode("laser_ai")
        if self.point_1f_mode == "ai_shapes":
            return self.set_point_1f_mode("pen")
        if self.point_1f_mode == "pen":
            return self.set_point_1f_mode("ai_shapes")
        return self.set_point_1f_mode("laser_ai")

    def select_top_candidate(self, rank_1based: int) -> None:
        """Phím [1/2/3]: Chọn lại ứng viên hình học trong Top-3 và vẽ lại nếu cần."""
        if 1 <= rank_1based <= len(self.last_top3_candidates):
            chosen_shape, prob = self.last_top3_candidates[rank_1based - 1]
            meta = (
                self.last_top3_meta[rank_1based - 1]
                if rank_1based - 1 < len(self.last_top3_meta)
                else None
            )
            chosen_raw = meta[1] if meta is not None else chosen_shape
            can_ppt = self.can_send_to_ppt()
            raw_src = self._last_shape_raw_strokes or self._last_ppt_ink_strokes

            if can_ppt:
                if self._last_ppt_com_tag:
                    delete_ppt_com_shapes_by_tag(self._last_ppt_com_tag)
                    self._last_ppt_com_tag = None
                elif self._last_ppt_ink_strokes:
                    erase_raw_strokes_on_ppt(
                        self._last_ppt_ink_strokes, self.screen_w, self.screen_h, restore_pen=True
                    )
                    self._last_ppt_ink_strokes.clear()

            if chosen_raw != "freeform":
                self.last_ink_accepted = True
                self.last_ink_kind = "shape"
                if can_ppt and raw_src:
                    self.active_tool_mode = "pen"
                    self._render_shape_onto_ppt(chosen_raw, raw_src)
            else:
                self.last_ink_accepted = False
                self.last_ink_kind = "freeform"
                if can_ppt and raw_src:
                    self.active_tool_mode = "pen"
                    clipped = [
                        [(int(round(px)), int(round(py))) for px, py in st]
                        for st in raw_src
                        if len(st) >= 2
                    ]
                    replay_win32_strokes(clipped, self.screen_w, self.screen_h)
                    self._last_ppt_ink_strokes = [list(s) for s in raw_src]

            self.last_ink_pred = chosen_shape
            self.last_ink_conf = float(prob)
            self.last_triggered_action = f"SUA TOP-{rank_1based} -> {chosen_shape.upper()} ({float(prob)*100:.0f}%)"
            if can_ppt and self.is_unified_laser_mode() and self.active_pose_cls == 3:
                self._enter_unified_laser_state(can_ppt=True, now_t=time.perf_counter(), guard_sec=0.55)

    def clear_ink_preview(self) -> None:
        """Xoá bộ đệm nét vẽ đang chờ và ô xem trước 64x64 trên HUD."""
        self._reset_ink_cluster()
        self._reset_unified_substate()
        self.last_top3_candidates.clear()
        self.last_top3_meta.clear()
        self._last_ppt_ink_strokes.clear()
        self._last_shape_raw_strokes.clear()
        self._last_ppt_com_tag = None
        self.last_ink_pred = ""
        self.last_ink_conf = 0.0
        self.last_ink_accepted = False
        self.last_ink_kind = ""
        self.last_ink_img64 = None

    def can_send_to_ppt(self) -> bool:
        """Kiểm tra hệ thống có đang được phép gửi phím/chuột sang PowerPoint hay không."""
        if not self.live_ppt:
            return False
        if self.ppt_only_lock and not (self.ppt_window_active or find_powerpoint_window() != 0):
            return False
        return True

    def _fill_buffer_steady(self) -> None:
        """Lấp đầy cửa sổ trượt bằng tư thế đứng yên hiện tại để xoá vận tốc tồn dư."""
        if len(self.landmark_buffer) == 0:
            return
        steady_pt = self.landmark_buffer[-1].copy()
        self.landmark_buffer.clear()
        for _ in range(SEQ_LEN):
            self.landmark_buffer.append(steady_pt.copy())

    def reset_all_buffers(self) -> None:
        """Xoá sạch cửa sổ trượt khi mất dấu bàn tay hoặc đưa tay xuống vùng nghỉ."""
        if self.is_ai_ink_active() and (self.ink_strokes or len(self.ink_current_stroke) >= 2):
            self._commit_air_ink_cluster(can_ppt=self.can_send_to_ppt(), return_to_laser=False)
        else:
            self._reset_ink_cluster()
        self._reset_unified_substate()
        was_tool = self.active_tool_mode is not None
        self.landmark_buffer.clear()
        self.vote_buffer.clear()
        self.cursor_filter.reset()
        self.smoothed_cursor = None
        self.active_pose_cls = 0
        self.pose_enter_cand = 0
        self.pose_enter_streak = 0
        self.pose_switch_cand = 0
        self.pose_switch_streak = 0
        self.pose_exit_streak = 0
        self.pen_paused = False
        self._release_pen_if_needed(reset_tool=True, send_arrow=was_tool)

    def _clear_motion_after_trigger(self) -> None:
        """Đặt lại trạng thái tĩnh trong buffer sau mỗi lần kích hoạt cử chỉ rời rạc."""
        self.vote_buffer.clear()
        self._fill_buffer_steady()
        self._release_pen_if_needed(reset_tool=False, send_arrow=False)

    def _release_pen_if_needed(self, reset_tool: bool = False, send_arrow: bool = False) -> None:
        """Nhả chuột trái an toàn và tuỳ chọn trả PowerPoint về con trỏ mũi tên (Ctrl+A)."""
        if self.mouse_down_active:
            if self.live_ppt:
                try:
                    win32_mouse_up()
                    print("[PEN] Nha chuot trai (ket thuc net ve)")
                except Exception:
                    pass
        self.mouse_down_active = False
        self.ink_pen_down = False
        self.last_drag_px = None
        if self.last_triggered_class in (3, 4):
            self.last_triggered_class = 0
        if reset_tool:
            had_tool = self.active_tool_mode is not None
            self.active_tool_mode = None
            self.tool_ready_time = 0.0
            self.pen_paused = False
            if self._cursor_overlay is not None:
                self._cursor_overlay.hide()
            if send_arrow and had_tool and self.can_send_to_ppt():
                win32_send_ctrl_key("A")
                print("[PEN] Tu dong thoat cong cu ve (Ctrl+A)")

    @torch.no_grad()
    def _infer_air_ink_cluster(
        self, strokes: List[List[Tuple[float, float]]]
    ) -> Optional[Dict[str, object]]:
        """Rasterize cụm nét thành ảnh 64x64 và suy luận Model 3 theo mặt nạ `mode_mask`."""
        if self.air_ink_model is None or not strokes:
            return None
        raw_arrs = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in strokes if len(s) >= 2]
        if not raw_arrs:
            return None

        # Lọc bỏ hạt nhiễu siêu nhỏ (<14px) khi đã có nét hình chính (>=28px) để tránh làm hỏng dự đoán
        spans = [max(float(a[:, 0].max() - a[:, 0].min()), float(a[:, 1].max() - a[:, 1].min())) for a in raw_arrs]
        min_real_span = max(28.0, 0.025 * float(self.screen_h))
        min_speck_span = max(14.0, 0.012 * float(self.screen_h))
        if any(sp >= min_real_span for sp in spans):
            filtered_pairs = [(s, a) for s, a, sp in zip(strokes, raw_arrs, spans) if len(s) >= 2 and sp >= min_speck_span]
            if filtered_pairs:
                clean_strokes = [p[0] for p in filtered_pairs]
                valid = [p[1] for p in filtered_pairs]
            else:
                clean_strokes = [s for s in strokes if len(s) >= 2]
                valid = raw_arrs
        else:
            clean_strokes = [s for s in strokes if len(s) >= 2]
            valid = raw_arrs

        all_pts = np.vstack(valid)
        x0, y0 = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
        x1, y1 = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
        min_span = max(24.0, 0.022 * float(self.screen_h))
        if max(x1 - x0, y1 - y0) < min_span:
            return None

        img64 = rasterize_strokes_to_64x64(
            valid, size=AIR_INK_IMG_SIZE, thickness=2, pad_ratio=0.16, blur_sigma=0.40
        )
        if int(img64.max()) < 20:
            return None
        self.last_ink_img64 = img64

        valid_ids = AIR_INK_SHAPE_IDS
        conf_gate = 0.72

        xt = torch.from_numpy((img64.astype(np.float32) * (1.0 / 255.0))[None, None, :, :]).to(
            self.device
        )
        logits = self.air_ink_model(xt)[0].cpu().numpy()
        masked = np.full_like(logits, -1e9)
        idx_list = list(valid_ids)
        masked[idx_list] = logits[idx_list]
        shifted = masked - masked.max()
        exp_l = np.exp(shifted)
        probs = exp_l / np.maximum(exp_l.sum(), 1e-12)

        order = np.argsort(-probs)
        top3: List[Tuple[str, float]] = []
        top3_meta: List[Tuple[int, str, str, bool]] = []
        for gid in order[:3]:
            igid = int(gid)
            if float(probs[igid]) <= 0.0 or igid not in valid_ids:
                continue
            raw_n = AIR_INK_CLASSES[igid]
            top3.append((raw_n, float(probs[igid])))
            top3_meta.append((igid, raw_n, raw_n, True))

        top1_gid = int(order[0])
        top1_raw = AIR_INK_CLASSES[top1_gid]
        top1_conf = float(probs[top1_gid])
        accepted = (top1_raw != "freeform") and (top1_conf >= conf_gate)

        return {
            "raw_name": top1_raw,
            "disp_name": top1_raw,
            "is_shape": True,
            "confidence": top1_conf,
            "conf_gate": conf_gate,
            "accepted": accepted,
            "bbox_px": (x0, y0, x1, y1),
            "valid_strokes": clean_strokes,
            "top3": top3,
            "top3_meta": top3_meta,
        }

    def _commit_air_ink_cluster(self, can_ppt: bool, return_to_laser: bool = False) -> None:
        """Chốt cụm nét vừa vẽ, suy luận Model 3 (chỉ 9 lớp hình) và nắn hình lên PowerPoint."""
        if self.mouse_down_active and can_ppt:
            try:
                win32_mouse_up()
            except Exception:
                pass
        self.mouse_down_active = False
        self.ink_pen_down = False
        self.last_drag_px = None

        all_strokes: List[List[Tuple[float, float]]] = [
            list(s) for s in self.ink_strokes if len(s) >= 2
        ]
        drawn_flags: List[bool] = [
            bool(self.ink_ppt_drawn_flags[i]) if i < len(self.ink_ppt_drawn_flags) else True
            for i, s in enumerate(self.ink_strokes)
            if len(s) >= 2
        ]
        if len(self.ink_current_stroke) >= 2:
            all_strokes.append(list(self.ink_current_stroke))
            drawn_flags.append(bool(self.ink_cur_ppt_drawn))

        self._reset_ink_cluster()
        if not all_strokes:
            if return_to_laser and self.is_unified_laser_mode():
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=time.perf_counter(), guard_sec=0.55)
            return

        drawn_strokes = [s for s, flag in zip(all_strokes, drawn_flags) if flag]
        res = self._infer_air_ink_cluster(all_strokes)
        if res is None:
            if return_to_laser and self.is_unified_laser_mode():
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=time.perf_counter(), guard_sec=0.55)
            return

        raw_name = str(res["raw_name"])
        disp_name = str(res["disp_name"])
        conf = float(res["confidence"])
        accepted = bool(res.get("accepted", False))
        clean_strokes = [list(s) for s in res.get("valid_strokes", all_strokes)]  # type: ignore[arg-type]
        self.last_top3_candidates = list(res["top3"])
        self.last_top3_meta = list(res.get("top3_meta", []))
        self._last_shape_raw_strokes = [list(s) for s in clean_strokes]
        self._last_ppt_ink_strokes = [list(s) for s in drawn_strokes]
        self._last_ppt_com_tag = None
        self.last_ink_pred = disp_name
        self.last_ink_conf = conf

        if accepted:
            self.last_ink_accepted = True
            self.last_ink_kind = "shape"
            if can_ppt:
                erase_mode = erase_raw_strokes_on_ppt(
                    drawn_strokes, self.screen_w, self.screen_h, restore_pen=True
                )
                self.active_tool_mode = "pen"
                draw_mode = self._render_shape_onto_ppt(raw_name, clean_strokes)
            else:
                erase_mode = "dry_run"
                draw_mode = "dry_run"
            self.last_triggered_action = f"AI SHAPE: {disp_name.upper()} ({conf*100:.0f}%)"
            print(
                f"[AIR-INK SHAPE] {disp_name} ({conf*100:.1f}%) "
                f"tu {len(clean_strokes)} net [erase={erase_mode}, draw={draw_mode}]"
            )
            if self.audio_beep:
                play_action_beep(7)
        else:
            self.last_ink_accepted = False
            self.last_ink_kind = "freeform" if raw_name == "freeform" else ""
            self.last_triggered_action = f"AI SHAPE: GIU NET GOC ({disp_name}, {conf*100:.0f}%)"
            print(f"[AIR-INK SHAPE] Giu nguyen net goc ({raw_name}, {conf*100:.1f}%)")

        if return_to_laser and self.is_unified_laser_mode():
            self._enter_unified_laser_state(can_ppt=can_ppt, now_t=time.perf_counter(), guard_sec=0.65)

    @torch.no_grad()
    def _infer_static_pose(
        self, pts_21x3: np.ndarray, aspect: float
    ) -> Tuple[np.ndarray, int, float]:
        """Suy luận Model 1 (`StaticPoseNet`, 1 frame) trả về [p_other, p_1f, p_2f]."""
        feat_6x21 = extract_static_pose_features(pts_21x3, aspect=aspect)
        x_t = torch.from_numpy(feat_6x21[np.newaxis, ...]).to(self.device)
        logits = self.pose_model(x_t)
        probs_3 = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(probs_3.argmax())
        mapped_cls = 3 if pred_idx == 1 else (4 if pred_idx == 2 else 0)
        return probs_3, mapped_cls, float(probs_3[pred_idx])

    @torch.no_grad()
    def _infer_with_trained_model(
        self, seq_30x63: np.ndarray
    ) -> Tuple[np.ndarray, int, float, int]:
        """Suy luận Model 2 (`HybridConvBiGRUAttention`) trên cửa sổ (1, 30, 66)."""
        feat_full30 = extract_66d_dynamic_features(seq_30x63)
        x_tensor = torch.from_numpy(feat_full30[np.newaxis, ...]).to(self.device)

        if hasattr(self.model, "attention"):
            logits, attn_weights = self.model(x_tensor, return_attn=True)
            peak_attn_frame = int(attn_weights[0].argmax().item()) + 1
        else:
            logits = self.model(x_tensor)
            peak_attn_frame = 15

        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        raw_class = int(probs.argmax())
        confidence = float(probs[raw_class])
        return probs, raw_class, confidence, peak_attn_frame

    @staticmethod
    def _map_pose_probs_to_hud(probs_3: np.ndarray) -> np.ndarray:
        """Ánh xạ xác suất 3 lớp của StaticPoseNet sang thanh hiển thị 8 lớp cho HUD."""
        hud_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        hud_probs[0] = float(probs_3[0])
        hud_probs[3] = float(probs_3[1])
        hud_probs[4] = float(probs_3[2])
        return hud_probs

    def push_frame_landmarks(
        self,
        pts_63: Optional[np.ndarray],
        current_time: float,
        aspect: float = 4.0 / 3.0,
    ) -> Dict[str, object]:
        """Điều phối Dual Deep Learning: StaticPoseNet (1 frame) + HybridConvBiGRUAttention (30 frames)."""
        cooldown_rem = max(0.0, self.cooldown_sec - (current_time - self.last_trigger_time))

        if pts_63 is None or np.allclose(pts_63, 0.0):
            self.missing_streak += 1
            if self.missing_streak <= 5 and len(self.landmark_buffer) > 0:
                self.landmark_buffer.append(self.last_valid_pts.copy())
            else:
                self.reset_all_buffers()
                return {
                    "status": "NO_HAND",
                    "raw_class": 0, "gated_class": 0, "confirmed_class": 0,
                    "confidence": 0.0, "peak_attn_frame": 0,
                    "probs": np.zeros(len(CLASS_NAMES), dtype=np.float32),
                    "cooldown_remaining": cooldown_rem, "triggered_event": None,
                }
        else:
            self.missing_streak = 0
            self.last_valid_pts = pts_63.astype(np.float32)
            self.landmark_buffer.append(self.last_valid_pts.copy())

        if cooldown_rem > 0.0:
            if len(self.landmark_buffer) == SEQ_LEN and cooldown_rem <= 0.25:
                self._fill_buffer_steady()
            return {
                "status": f"COOLDOWN ({cooldown_rem:.1f}s)",
                "raw_class": 0, "gated_class": 0, "confirmed_class": 0,
                "confidence": 0.0, "peak_attn_frame": 0,
                "probs": np.zeros(len(CLASS_NAMES), dtype=np.float32),
                "cooldown_remaining": cooldown_rem, "triggered_event": None,
            }

        # ════════════════════════════════════════════════════════════════════════
        # TẦNG 1: MODEL 1 (STATICPOSENET - 1 FRAME) -> TRẠNG THÁI CHỈ TRỎ 1F / 2F
        # ════════════════════════════════════════════════════════════════════════
        if self.pose_model is not None:
            pts_21x3 = self.last_valid_pts.reshape(21, 3)
            probs_3, pose_raw_cls, pose_conf = self._infer_static_pose(pts_21x3, aspect=aspect)
            enter_th = min(self.conf_threshold, 0.70)
            keep_th = 0.45

            if self.active_pose_cls in (3, 4):
                other_cls = 4 if self.active_pose_cls == 3 else 3
                other_idx = 2 if other_cls == 4 else 1

                # Chuyển mượt 1 ngón <-> 2 ngón nếu Model 1 tự tin >= enter_th trong 2 frames
                if float(probs_3[other_idx]) >= enter_th:
                    if self.pose_switch_cand == other_cls:
                        self.pose_switch_streak += 1
                    else:
                        self.pose_switch_cand = other_cls
                        self.pose_switch_streak = 1
                    if self.pose_switch_streak >= 2:
                        self._reset_ink_cluster()
                        self._reset_unified_substate()
                        self._release_pen_if_needed(reset_tool=False, send_arrow=False)
                        self.active_pose_cls = other_cls
                        self.pose_switch_cand = 0
                        self.pose_switch_streak = 0
                        self.pose_exit_streak = 0
                else:
                    self.pose_switch_cand = 0
                    self.pose_switch_streak = 0

                curr_prob = float(probs_3[1 if self.active_pose_cls == 3 else 2])
                if curr_prob >= keep_th:
                    self.pose_exit_streak = 0
                    self.pen_paused = False
                else:
                    self.pose_exit_streak += 1
                    # Tạm nhấc bút ngay trong vài frame chuyển tiếp khi hạ ngón để tránh kéo nét mực thừa
                    if float(probs_3[0]) >= 0.72:
                        self.pen_paused = True

                    if self.pose_exit_streak >= 4:
                        if self.active_pose_cls == 3 and self.is_ai_ink_active():
                            self._commit_air_ink_cluster(can_ppt=self.can_send_to_ppt(), return_to_laser=False)
                        else:
                            self._reset_ink_cluster()
                        self._reset_unified_substate()
                        self.active_pose_cls = 0
                        self.pose_exit_streak = 0
                        self.pose_enter_cand = 0
                        self.pose_enter_streak = 0
                        self._release_pen_if_needed(reset_tool=True, send_arrow=True)
                        self.vote_buffer.clear()
                        self._fill_buffer_steady()
                        self.dynamic_guard_until = current_time + 0.40
                        self.last_triggered_class = 0
                        self.last_triggered_action = "THOAT CONG CU VE (Ha ngon tay - Ctrl+A)"
                        return {
                            "status": "MODEL_READY (POSE->SEQ)",
                            "raw_class": 0, "gated_class": 0, "confirmed_class": 0,
                            "confidence": float(probs_3[0]), "peak_attn_frame": 1,
                            "probs": self._map_pose_probs_to_hud(probs_3),
                            "cooldown_remaining": 0.0,
                            "triggered_event": "exit_tool_ctrl_a",
                        }

                if self.active_pose_cls in (3, 4):
                    act_cls = self.active_pose_cls
                    act_prob = float(probs_3[1 if act_cls == 3 else 2])
                    self.last_triggered_class = act_cls
                    ev = "point_1f (Spatial Pen)" if act_cls == 3 else "point_2f (Spatial Eraser)"
                    self.last_triggered_action = (
                        "POINT_1F (1 Ngon - Viet)" if act_cls == 3 else "POINT_2F (2 Ngon - Xoa muc)"
                    )
                    return {
                        "status": "MODEL_READY (POSE_NET)",
                        "raw_class": act_cls,
                        "gated_class": act_cls,
                        "confirmed_class": act_cls,
                        "confidence": act_prob,
                        "peak_attn_frame": 1,
                        "probs": self._map_pose_probs_to_hud(probs_3),
                        "cooldown_remaining": 0.0,
                        "triggered_event": ev,
                    }
            else:
                # Chưa vào chế độ chỉ trỏ -> xét điều kiện vào point_1f / point_2f (2 frames liên tiếp >= enter_th)
                if pose_raw_cls in (3, 4) and pose_conf >= enter_th:
                    if self.pose_enter_cand == pose_raw_cls:
                        self.pose_enter_streak += 1
                    else:
                        self.pose_enter_cand = pose_raw_cls
                        self.pose_enter_streak = 1

                    if self.pose_enter_streak >= 2:
                        self.active_pose_cls = pose_raw_cls
                        self.pose_enter_cand = 0
                        self.pose_enter_streak = 0
                        self.pose_exit_streak = 0
                        self.pen_paused = False
                        self.vote_buffer.clear()
                        self.last_triggered_class = pose_raw_cls
                        ev = "point_1f (Spatial Pen)" if pose_raw_cls == 3 else "point_2f (Spatial Eraser)"
                        self.last_triggered_action = (
                            "POINT_1F (1 Ngon - Viet)" if pose_raw_cls == 3 else "POINT_2F (2 Ngon - Xoa muc)"
                        )
                        return {
                            "status": "MODEL_READY (POSE_NET)",
                            "raw_class": pose_raw_cls,
                            "gated_class": pose_raw_cls,
                            "confirmed_class": pose_raw_cls,
                            "confidence": pose_conf,
                            "peak_attn_frame": 1,
                            "probs": self._map_pose_probs_to_hud(probs_3),
                            "cooldown_remaining": 0.0,
                            "triggered_event": ev,
                        }
                else:
                    self.pose_enter_cand = 0
                    self.pose_enter_streak = 0

        # ════════════════════════════════════════════════════════════════════════
        # TẦNG 2: MODEL 2 (HYBRIDCONVBiGRUATTENTION - 30 FRAMES) -> VUỐT/ZOOM/CLICK
        # ════════════════════════════════════════════════════════════════════════
        if current_time < self.dynamic_guard_until:
            self._fill_buffer_steady()
            return {
                "status": "MODEL_READY (EXIT_GUARD)",
                "raw_class": 0, "gated_class": 0, "confirmed_class": 0,
                "confidence": 1.0, "peak_attn_frame": 15,
                "probs": np.eye(len(CLASS_NAMES), dtype=np.float32)[0],
                "cooldown_remaining": 0.0, "triggered_event": None,
            }

        if len(self.landmark_buffer) < SEQ_LEN:
            return {
                "status": f"WARMUP ({len(self.landmark_buffer)}/{SEQ_LEN})",
                "raw_class": 0, "gated_class": 0, "confirmed_class": 0,
                "confidence": 0.0, "peak_attn_frame": 0,
                "probs": np.zeros(len(CLASS_NAMES), dtype=np.float32),
                "cooldown_remaining": 0.0, "triggered_event": None,
            }

        seq_30x63 = np.array(self.landmark_buffer, dtype=np.float32)
        probs, raw_class, confidence, peak_attn_frame = self._infer_with_trained_model(seq_30x63)

        # Kiểm tra cửa sổ ngắn 16 frames (chỉ cho vuốt nhanh 1, 2 và phải tự đạt >= conf_threshold từ Model)
        if not (raw_class in (1, 2) and confidence >= self.conf_threshold):
            fast_seq = temporal_resample(seq_30x63[-16:], SEQ_LEN)
            f_probs, f_cls, f_conf, f_attn = self._infer_with_trained_model(fast_seq)
            if f_cls in (1, 2) and f_conf >= self.conf_threshold:
                probs, raw_class, confidence, peak_attn_frame = f_probs, f_cls, f_conf, f_attn

        if self.pose_model is not None:
            # Khi đã có Model 1 (StaticPoseNet), Model 2 chỉ kích hoạt 5 lệnh động DISCRETE_CLASSES
            gated_class = raw_class if (raw_class in DISCRETE_CLASSES and confidence >= self.conf_threshold) else 0
        else:
            gated_class = raw_class if confidence >= self.conf_threshold else 0

        self.vote_buffer.append(gated_class)

        confirmed_class = 0
        if gated_class != 0:
            most_common_cls, count = Counter(self.vote_buffer).most_common(1)[0]
            if most_common_cls == gated_class:
                req_votes = min(2, self.vote_required) if gated_class in (1, 2) else self.vote_required
                if count >= req_votes:
                    confirmed_class = gated_class

        triggered_event = None
        if confirmed_class in DISCRETE_CLASSES:
            self._release_pen_if_needed(reset_tool=False, send_arrow=False)
            triggered_event = self._execute_discrete_action(confirmed_class, current_time)
        elif confirmed_class in (3, 4):
            self.pen_paused = False
            triggered_event = "point_1f (Spatial Pen)" if confirmed_class == 3 else "point_2f (Spatial Eraser)"
            self.last_triggered_class = confirmed_class
        else:
            self._release_pen_if_needed(reset_tool=False, send_arrow=False)

        return {
            "status": "MODEL_READY (SEQ_NET)",
            "raw_class": raw_class,
            "gated_class": gated_class,
            "confirmed_class": confirmed_class,
            "confidence": confidence,
            "peak_attn_frame": peak_attn_frame,
            "probs": probs,
            "cooldown_remaining": 0.0,
            "triggered_event": triggered_event,
        }

    def _execute_discrete_action(self, cls_id: int, current_time: float) -> Optional[str]:
        """Gửi phím phần cứng hoặc Click chuột điều khiển PowerPoint tương ứng với nhãn do Model quyết định."""
        action_name = CLASS_NAMES[cls_id]
        if not self.can_send_to_ppt():
            self.last_triggered_action = f"{action_name.upper()} [CHUA MO POWERPOINT]"
            self.vote_buffer.clear()
            return None

        if cls_id == 1:
            win32_send_ppt_slide_key("next")
            print("[SLIDE] Next Slide (VK_NEXT / PageDown)")
        elif cls_id == 2:
            win32_send_ppt_slide_key("prev")
            print("[SLIDE] Previous Slide (VK_PRIOR / PageUp)")
        elif cls_id == 7:
            ensure_powerpoint_foreground()
            if self.smoothed_cursor is not None:
                cx = int(np.clip(self.smoothed_cursor[0], 5, self.screen_w - 5))
                cy = int(np.clip(self.smoothed_cursor[1], 5, self.screen_h - 5))
            elif len(self.landmark_buffer) > 0:
                pts0 = self.landmark_buffer[0].reshape(21, 3)
                cx = int(np.clip((1.0 - float(pts0[8, 0])) * self.screen_w, 5, self.screen_w - 5))
                cy = int(np.clip(float(pts0[8, 1]) * self.screen_h, 5, self.screen_h - 5))
            else:
                cx, cy = self.screen_w // 2, self.screen_h // 2
            win32_left_click(cx, cy, self.screen_w, self.screen_h)
            print(f"[CLICK] Click chuot trai tai ({cx}, {cy})")
        elif pyautogui is not None:
            ensure_powerpoint_foreground()
            if cls_id == 5:
                pyautogui.hotkey("ctrl", "=")
            elif cls_id == 6:
                pyautogui.hotkey("ctrl", "-")

        if self.audio_beep:
            play_action_beep(cls_id)

        # Tuân thủ Cooldown >= 1.05s cho thao tác vuốt để nhịp thu tay về không kích hoạt trượt ngược slide
        if cls_id in (1, 2):
            eff_cd = min(self.cooldown_sec, 1.05) if self.cooldown_sec >= 1.05 else self.cooldown_sec
            self.last_trigger_time = current_time - max(0.0, self.cooldown_sec - eff_cd)
        else:
            self.last_trigger_time = current_time

        self.last_triggered_class = cls_id
        self.last_triggered_action = f"{action_name.upper()} ({ACTION_DISPLAY[cls_id][0]})"
        self.active_tool_mode = None
        self._clear_motion_after_trigger()
        return action_name

    def _should_split_spatial_cluster(self, px: float, py: float) -> bool:
        """Kiểm tra xem nét mới ở `(px, py)` có nằm tách biệt hẳn khỏi cụm hình cũ hay không."""
        valid = [np.asarray(s, dtype=np.float32).reshape(-1, 2) for s in self.ink_strokes if len(s) >= 2]
        if not valid:
            return False
        pts = np.vstack(valid)
        bx0, by0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        bx1, by1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        prev_span = max(bx1 - bx0, by1 - by0)
        if prev_span < 40.0:
            return False
        dx = max(0.0, bx0 - px, px - bx1)
        dy = max(0.0, by0 - py, py - by1)
        return math.hypot(dx, dy) > max(120.0, 0.90 * prev_span)

    def _finalize_current_ink_stroke(self, can_ppt: bool) -> None:
        """Nhả chuột an toàn và lưu nét đang vẽ nếu đủ dài (bỏ qua chấm rung <16px)."""
        if self.mouse_down_active and can_ppt:
            try:
                win32_mouse_up()
            except Exception:
                pass
        self.mouse_down_active = False
        self.ink_pen_down = False
        min_real_stroke_px = max(16.0, 0.015 * float(self.screen_h))
        if len(self.ink_current_stroke) >= 2 and (
            self.ink_cur_ppt_drawn or self.ink_stroke_arc_len >= min_real_stroke_px
        ):
            self.ink_strokes.append(list(self.ink_current_stroke))
            self.ink_ppt_drawn_flags.append(bool(self.ink_cur_ppt_drawn))
        self.ink_current_stroke.clear()
        self.ink_cur_ppt_drawn = False
        self.ink_stroke_arc_len = 0.0

    def _update_ai_ink_pointer(
        self, scr_x: int, scr_y: int, prev_x: int, prev_y: int, now_t: float
    ) -> None:
        """
        Phân đoạn nét thông minh trong `point_1f` (AI Nắn Hình):
        - Delayed Pen-Down: Chờ nét đi đủ >= 16px mới nhấn giữ chuột PowerPoint (tránh chấm đỏ rác).
        - Tách cụm theo khoảng cách / nhảy cóc frame để không gộp vệt thừa thành freeform.
        - Dừng 0.42s -> Dwell Pen-Up; Dừng 1.05s (hoặc hạ ngón) -> Model 3 + COM Shape / Ink Replay.
        """
        can_ppt = self.can_send_to_ppt()
        if can_ppt and self.active_tool_mode != "pen":
            if self.mouse_down_active:
                win32_mouse_up()
                self.mouse_down_active = False
            win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            win32_send_ctrl_key("P")
            self.active_tool_mode = "pen"
            self.tool_ready_time = now_t + 0.14
            self.ink_anchor_px = (float(scr_x), float(scr_y))
            self.ink_last_move_t = now_t
            self.ink_cur_ppt_drawn = False
            self.ink_stroke_arc_len = 0.0
            self.last_drag_px = (scr_x, scr_y)
            self.last_triggered_action = f"POINT_1F [{self.point_1f_mode.upper()}: MO BUT Ctrl+P]"
            return

        if self.ink_anchor_px is None:
            self.ink_anchor_px = (float(scr_x), float(scr_y))
            self.ink_last_move_t = now_t

        if self.pen_paused:
            self._finalize_current_ink_stroke(can_ppt=can_ppt)
            if can_ppt:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            self.ink_anchor_px = (float(scr_x), float(scr_y))
            self.last_drag_px = (scr_x, scr_y)
            self._update_cursor_overlay(scr_x, scr_y, visible=False)
            return

        move_thresh_px = max(7.0, 0.0055 * float(self.screen_h))
        min_real_stroke_px = max(16.0, 0.015 * float(self.screen_h))
        teleport_thresh_px = max(180.0, 0.22 * float(min(self.screen_w, self.screen_h)))
        dist = math.hypot(scr_x - self.ink_anchor_px[0], scr_y - self.ink_anchor_px[1])

        if not self.ink_pen_down:
            if dist > teleport_thresh_px:
                if self._should_split_spatial_cluster(float(scr_x), float(scr_y)):
                    self._commit_air_ink_cluster(
                        can_ppt=can_ppt, return_to_laser=self.is_unified_laser_mode()
                    )
                    if self.is_unified_laser_mode():
                        return
                self.ink_anchor_px = (float(scr_x), float(scr_y))
                self.ink_last_move_t = now_t
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                self._update_cursor_overlay(
                    scr_x, scr_y, mode="pen_ready", progress=1.0, visible=self.is_unified_laser_mode()
                )
                return

            if dist >= move_thresh_px * 1.35:
                if self._should_split_spatial_cluster(float(scr_x), float(scr_y)):
                    self._commit_air_ink_cluster(
                        can_ppt=can_ppt, return_to_laser=self.is_unified_laser_mode()
                    )
                    if self.is_unified_laser_mode():
                        return
                    self.ink_anchor_px = (float(scr_x), float(scr_y))
                    self.ink_last_move_t = now_t
                    if can_ppt:
                        win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                    self.last_drag_px = (scr_x, scr_y)
                    return
                self.ink_pen_down = True
                self.ink_cur_ppt_drawn = False
                self.ink_current_stroke = [self.ink_anchor_px, (float(scr_x), float(scr_y))]
                self.ink_stroke_arc_len = dist
                self.ink_anchor_px = (float(scr_x), float(scr_y))
                self.ink_last_move_t = now_t
                self.ink_commit_progress = 0.0
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
            else:
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                if self.ink_strokes:
                    idle_dur = now_t - self.ink_last_move_t
                    span_commit = max(0.20, self.ink_commit_sec - self.ink_dwell_pen_up_sec)
                    self.ink_commit_progress = float(
                        np.clip((idle_dur - self.ink_dwell_pen_up_sec) / span_commit, 0.0, 1.0)
                    )
                    if idle_dur >= self.ink_commit_sec:
                        self._commit_air_ink_cluster(
                            can_ppt=can_ppt, return_to_laser=self.is_unified_laser_mode()
                        )
                        return
                    if self.ink_commit_progress > 0.02:
                        self._update_cursor_overlay(
                            scr_x, scr_y, mode="ai_commit", progress=self.ink_commit_progress, visible=True
                        )
                    else:
                        self._update_cursor_overlay(
                            scr_x, scr_y, mode="pen_ready", progress=1.0, visible=self.is_unified_laser_mode()
                        )
                else:
                    self.ink_commit_progress = 0.0
                    self._update_cursor_overlay(
                        scr_x, scr_y, mode="pen_ready", progress=1.0, visible=self.is_unified_laser_mode()
                    )
        else:
            self.ink_commit_progress = 0.0
            self._update_cursor_overlay(scr_x, scr_y, visible=False)
            if dist > teleport_thresh_px:
                self._finalize_current_ink_stroke(can_ppt=can_ppt)
                self.ink_anchor_px = (float(scr_x), float(scr_y))
                self.ink_last_move_t = now_t
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                return

            if dist >= move_thresh_px * 0.50:
                self.ink_current_stroke.append((float(scr_x), float(scr_y)))
                self.ink_stroke_arc_len += dist
                self.ink_anchor_px = (float(scr_x), float(scr_y))
                self.ink_last_move_t = now_t
                if can_ppt:
                    if (
                        not self.ink_cur_ppt_drawn
                        and now_t >= self.tool_ready_time
                        and len(self.ink_current_stroke) >= 3
                        and self.ink_stroke_arc_len >= min_real_stroke_px
                    ):
                        p0x = int(round(self.ink_current_stroke[0][0]))
                        p0y = int(round(self.ink_current_stroke[0][1]))
                        win32_mouse_down(p0x, p0y, self.screen_w, self.screen_h)
                        px_prev, py_prev = p0x, p0y
                        for bx, by in self.ink_current_stroke[1:]:
                            ibx, iby = int(round(bx)), int(round(by))
                            win32_drag_cursor(
                                px_prev, py_prev, ibx, iby, self.screen_w, self.screen_h, steps=3
                            )
                            px_prev, py_prev = ibx, iby
                        self.ink_cur_ppt_drawn = True
                        self.mouse_down_active = True
                        self.last_drag_px = (px_prev, py_prev)
                    elif self.ink_cur_ppt_drawn:
                        p_x, p_y = self.last_drag_px if self.last_drag_px is not None else (prev_x, prev_y)
                        steps = max(1, min(12, int(math.hypot(scr_x - p_x, scr_y - p_y) // 6)))
                        win32_drag_cursor(p_x, p_y, scr_x, scr_y, self.screen_w, self.screen_h, steps=steps)
                        self.last_drag_px = (scr_x, scr_y)
                    else:
                        win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                        self.last_drag_px = (scr_x, scr_y)
            elif (now_t - self.ink_last_move_t) >= self.ink_dwell_pen_up_sec:
                self._finalize_current_ink_stroke(can_ppt=can_ppt)
                self.last_drag_px = (scr_x, scr_y)
                self.ink_anchor_px = (float(scr_x), float(scr_y))

    def _update_unified_laser_pen_pointer(
        self, scr_x: int, scr_y: int, prev_x: int, prev_y: int, now_t: float
    ) -> None:
        """
        Phương án C (Tích hợp Laser + Bút không cần train lại Model 1):
        - Mặc định giơ 1 ngón: rọi Tia Laser (`Ctrl+L`) thoải mái khắp slide.
        - Giữ yên ~0.72s (có vòng tròn nạp tiến độ trực tiếp trên slide): tự mở Bút (`Ctrl+P`).
        - Trong `laser_ai`: vẽ nét/hình -> dừng 1.05s AI chốt COM Shape -> tự quay lại Laser (`Ctrl+L`).
        - Trong `laser_pen`: vẽ tự do -> dừng 0.65s nhả bút -> tự quay lại Laser (`Ctrl+L`).
        """
        can_ppt = self.can_send_to_ppt()
        if self.pen_paused:
            if self.unified_substate == "pen_armed":
                if self.point_1f_mode == "laser_ai":
                    self._finalize_current_ink_stroke(can_ppt=can_ppt)
                elif self.mouse_down_active and can_ppt:
                    try:
                        win32_mouse_up()
                    except Exception:
                        pass
                    self.mouse_down_active = False
            if can_ppt:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            self._update_cursor_overlay(scr_x, scr_y, visible=False)
            self.last_drag_px = (scr_x, scr_y)
            return

        dwell_radius_px = max(14.0, 0.013 * float(self.screen_h))
        min_freehand_px = max(12.0, 0.010 * float(self.screen_h))
        move_thresh_px = max(7.0, 0.0055 * float(self.screen_h))
        teleport_thresh_px = max(180.0, 0.22 * float(min(self.screen_w, self.screen_h)))

        if self.unified_substate == "laser":
            if can_ppt and self.active_tool_mode != "laser":
                if self.mouse_down_active:
                    try:
                        win32_mouse_up()
                    except Exception:
                        pass
                    self.mouse_down_active = False
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                win32_send_ctrl_key("L")
                self.active_tool_mode = "laser"
                self.tool_ready_time = now_t + 0.10
                self.unified_dwell_anchor_px = (float(scr_x), float(scr_y))
                self.unified_dwell_start_t = max(now_t, self.unified_dwell_guard_until)
                self.unified_dwell_progress = 0.0
                self.last_drag_px = (scr_x, scr_y)
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
                self.last_triggered_action = "POINT_1F [LASER -> Giu yen 0.7s de bat But]"
                return

            if can_ppt:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            self.last_drag_px = (scr_x, scr_y)

            if self.unified_dwell_anchor_px is None or now_t < self.unified_dwell_guard_until:
                self.unified_dwell_anchor_px = (float(scr_x), float(scr_y))
                self.unified_dwell_start_t = now_t
                self.unified_dwell_progress = 0.0
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
                return

            dist_anchor = math.hypot(
                scr_x - self.unified_dwell_anchor_px[0],
                scr_y - self.unified_dwell_anchor_px[1],
            )
            if dist_anchor > dwell_radius_px:
                self.unified_dwell_anchor_px = (float(scr_x), float(scr_y))
                self.unified_dwell_start_t = now_t
                self.unified_dwell_progress = 0.0
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
                return

            dwell_dur = now_t - self.unified_dwell_start_t
            if dwell_dur < self.unified_dwell_grace_sec:
                self.unified_dwell_progress = 0.0
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
                return

            span_arm = max(0.15, self.unified_dwell_arm_sec - self.unified_dwell_grace_sec)
            self.unified_dwell_progress = float(
                np.clip((dwell_dur - self.unified_dwell_grace_sec) / span_arm, 0.0, 1.0)
            )
            if dwell_dur < self.unified_dwell_arm_sec:
                self._update_cursor_overlay(
                    scr_x, scr_y, mode="dwell_arm", progress=self.unified_dwell_progress, visible=True
                )
                return

            # Giữ yên đủ ngưỡng -> Mở Bút (Ctrl+P)
            self.unified_substate = "pen_armed"
            self.unified_dwell_progress = 0.0
            self.unified_pen_armed_t = now_t
            self.unified_pen_finish_progress = 0.0
            self.unified_pen_stroke_pts.clear()
            self.unified_pen_arc_len = 0.0
            ax, ay = self.unified_dwell_anchor_px
            self.ink_anchor_px = (float(ax), float(ay))
            self.ink_last_move_t = now_t
            self.ink_pen_down = False
            self.ink_cur_ppt_drawn = False
            self.ink_stroke_arc_len = 0.0
            self.ink_current_stroke.clear()

            if can_ppt:
                win32_send_ctrl_key("P")
                self.active_tool_mode = "pen"
                self.tool_ready_time = now_t + 0.12
                if self.audio_beep:
                    play_dwell_beep("pen_on")
            else:
                self.active_tool_mode = "pen"
                self.tool_ready_time = now_t + 0.02
            self._update_cursor_overlay(scr_x, scr_y, mode="pen_ready", progress=1.0, visible=True)
            mode_tag = "AI NAN HINH" if self.point_1f_mode == "laser_ai" else "BUT VE TAY"
            self.last_triggered_action = f"POINT_1F [DA BAT {mode_tag} (Ctrl+P)]"
            return

        # Trạng thái `pen_armed`
        if self.point_1f_mode == "laser_ai":
            if (
                not self.ink_strokes
                and not self.ink_pen_down
                and not self.ink_cur_ppt_drawn
                and (now_t - self.unified_pen_armed_t) >= self.unified_pen_idle_timeout_sec
            ):
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=now_t, guard_sec=0.55)
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_triggered_action = "POINT_1F [HUY BUT -> QUAY VE LASER]"
                return
            self._update_ai_ink_pointer(scr_x, scr_y, prev_x, prev_y, now_t)
            return

        # Nhánh `laser_pen`: Laser + Bút vẽ tay tự do (không nắn hình AI)
        if self.ink_anchor_px is None:
            self.ink_anchor_px = (float(scr_x), float(scr_y))
            self.ink_last_move_t = now_t

        dist = math.hypot(scr_x - self.ink_anchor_px[0], scr_y - self.ink_anchor_px[1])
        if not self.mouse_down_active:
            if dist > teleport_thresh_px:
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=now_t, guard_sec=0.55)
                if can_ppt:
                    win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_triggered_action = "POINT_1F [HUY BUT -> QUAY VE LASER]"
                return

            if dist >= move_thresh_px * 0.50:
                if not self.unified_pen_stroke_pts:
                    self.unified_pen_stroke_pts.append(self.ink_anchor_px)
                self.unified_pen_stroke_pts.append((float(scr_x), float(scr_y)))
                self.unified_pen_arc_len += dist
                self.ink_anchor_px = (float(scr_x), float(scr_y))
                self.ink_last_move_t = now_t

            if can_ppt:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            self.last_drag_px = (scr_x, scr_y)

            if (
                now_t >= self.tool_ready_time
                and len(self.unified_pen_stroke_pts) >= 2
                and self.unified_pen_arc_len >= min_freehand_px
            ):
                if can_ppt:
                    p0x = int(round(self.unified_pen_stroke_pts[0][0]))
                    p0y = int(round(self.unified_pen_stroke_pts[0][1]))
                    win32_mouse_down(p0x, p0y, self.screen_w, self.screen_h)
                    px_prev, py_prev = p0x, p0y
                    for bx, by in self.unified_pen_stroke_pts[1:]:
                        ibx, iby = int(round(bx)), int(round(by))
                        win32_drag_cursor(
                            px_prev, py_prev, ibx, iby, self.screen_w, self.screen_h, steps=3
                        )
                        px_prev, py_prev = ibx, iby
                    self.last_drag_px = (px_prev, py_prev)
                self.mouse_down_active = True
                self.ink_pen_down = True
                self.unified_pen_finish_progress = 0.0
                self._update_cursor_overlay(scr_x, scr_y, visible=False)
                self.last_triggered_action = "POINT_1F [DANG VE BUT TAY -> Dung 0.65s ve Laser]"
            elif (
                self.unified_pen_arc_len < min_freehand_px
                and (now_t - self.unified_pen_armed_t) >= self.unified_pen_idle_timeout_sec
            ):
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=now_t, guard_sec=0.55)
                self.last_triggered_action = "POINT_1F [HUY BUT -> QUAY VE LASER]"
            else:
                self._update_cursor_overlay(scr_x, scr_y, mode="pen_ready", progress=1.0, visible=True)
            return

        # Đang nhấn giữ chuột vẽ tay tự do trong `laser_pen`
        if dist > teleport_thresh_px:
            self._enter_unified_laser_state(can_ppt=can_ppt, now_t=now_t, guard_sec=0.55)
            if can_ppt:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
            self.last_triggered_action = "POINT_1F [XONG NET -> QUAY VE LASER]"
            return

        if dist >= move_thresh_px * 0.45:
            if can_ppt:
                p_x, p_y = self.last_drag_px if self.last_drag_px is not None else (prev_x, prev_y)
                steps = max(1, min(12, int(math.hypot(scr_x - p_x, scr_y - p_y) // 6)))
                win32_drag_cursor(p_x, p_y, scr_x, scr_y, self.screen_w, self.screen_h, steps=steps)
            self.last_drag_px = (scr_x, scr_y)
            self.ink_anchor_px = (float(scr_x), float(scr_y))
            self.ink_last_move_t = now_t
            self.unified_pen_finish_progress = 0.0
            self._update_cursor_overlay(scr_x, scr_y, visible=False)
        else:
            idle_dur = now_t - self.ink_last_move_t
            fin_grace = 0.25
            span_fin = max(0.15, self.unified_laser_pen_finish_sec - fin_grace)
            self.unified_pen_finish_progress = (
                float(np.clip((idle_dur - fin_grace) / span_fin, 0.0, 1.0))
                if idle_dur >= fin_grace
                else 0.0
            )
            if idle_dur >= self.unified_laser_pen_finish_sec:
                self._enter_unified_laser_state(can_ppt=can_ppt, now_t=now_t, guard_sec=0.65)
                if can_ppt and self.audio_beep:
                    play_dwell_beep("laser_on")
                self.last_triggered_action = "POINT_1F [XONG NET -> TU QUAY VE LASER]"
            elif self.unified_pen_finish_progress > 0.02:
                self._update_cursor_overlay(
                    scr_x,
                    scr_y,
                    mode="pen_finish",
                    progress=self.unified_pen_finish_progress,
                    visible=True,
                )
            else:
                self._update_cursor_overlay(scr_x, scr_y, visible=False)

    def update_spatial_pointer(
        self, pointer_xy_mirror: Tuple[float, float], active_mode: int = 0
    ) -> Tuple[int, int]:
        """Điều khiển con trỏ chuột không gian mượt mà bằng OneEuroFilter2D & vẽ Win32 2 bước."""
        ix, iy = pointer_xy_mirror
        norm_x = float(np.clip((ix - MARGIN_X) / max(1e-4, 1.0 - 2.0 * MARGIN_X), 0.0, 1.0))
        norm_y = float(np.clip((iy - MARGIN_Y) / max(1e-4, 1.0 - 2.0 * MARGIN_Y), 0.0, 1.0))

        target_x = norm_x * (self.screen_w - 1)
        target_y = norm_y * (self.screen_h - 1)
        now_t = time.perf_counter()

        prev_x, prev_y = (
            (int(self.smoothed_cursor[0]), int(self.smoothed_cursor[1]))
            if self.smoothed_cursor is not None
            else (int(target_x), int(target_y))
        )
        sx, sy = self.cursor_filter((target_x, target_y), now_t)
        self.smoothed_cursor = (sx, sy)

        scr_x = int(np.clip(sx, 5, self.screen_w - 5))
        scr_y = int(np.clip(sy, 5, self.screen_h - 5))

        if active_mode == 3:
            if self.is_unified_laser_mode():
                self._update_unified_laser_pen_pointer(scr_x, scr_y, prev_x, prev_y, now_t)
                return scr_x, scr_y
            if self.is_ai_ink_active():
                self._update_ai_ink_pointer(scr_x, scr_y, prev_x, prev_y, now_t)
                return scr_x, scr_y

        self._update_cursor_overlay(scr_x, scr_y, visible=False)

        if active_mode in (3, 4) and self.can_send_to_ppt():
            if active_mode == 3:
                if self.pen_draw_click:
                    want_tool = "pen"
                    hotkey = "P"
                    step1_txt = "MO BUT VE (Ctrl+P)"
                    step2_txt = "DANG VE MUC DO (Ha ngon de thoat)"
                else:
                    want_tool = "laser"
                    hotkey = "L"
                    step1_txt = "MO TIA LASER DO (Ctrl+L)"
                    step2_txt = "DANG ROI LASER DO (Ha ngon de thoat)"
                label = "POINT_1F"
            else:
                want_tool = "eraser"
                hotkey = "E"
                step1_txt = "MO CUC TAY (Ctrl+E)"
                step2_txt = "DANG XOA MUC (Ha ngon de thoat)"
                label = "POINT_2F"

            if self.active_tool_mode != want_tool:
                if self.mouse_down_active:
                    win32_mouse_up()
                    self.mouse_down_active = False
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                win32_send_ctrl_key(hotkey)
                self.active_tool_mode = want_tool
                self.tool_ready_time = now_t + 0.12
                self.last_drag_px = (scr_x, scr_y)
                self.last_triggered_action = f"{label} [BUOC 1: {step1_txt}]"
                print(f"[PEN] BUOC 1: Gui Ctrl+{hotkey} mo {want_tool}")
                return scr_x, scr_y

            if active_mode == 3 and not self.pen_draw_click:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_triggered_action = f"{label} [{step2_txt}]"
                return scr_x, scr_y

            if self.pen_paused:
                if self.mouse_down_active:
                    win32_mouse_up()
                    self.mouse_down_active = False
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                self.last_triggered_action = f"{label} [TAM NHAC BUT]"
                return scr_x, scr_y

            if now_t < self.tool_ready_time:
                win32_move_cursor(scr_x, scr_y, self.screen_w, self.screen_h)
                self.last_drag_px = (scr_x, scr_y)
                self.last_triggered_action = f"{label} [BUOC 1: {step1_txt}]"
            elif not self.mouse_down_active:
                win32_mouse_down(scr_x, scr_y, self.screen_w, self.screen_h)
                self.mouse_down_active = True
                self.last_drag_px = (scr_x, scr_y)
                self.last_triggered_action = f"{label} [{step2_txt}]"
                print(f"[PEN] BUOC 2: Nhan giu chuot tai ({scr_x}, {scr_y})")
            else:
                p_x, p_y = self.last_drag_px if self.last_drag_px is not None else (prev_x, prev_y)
                steps = max(1, min(12, int(math.hypot(scr_x - p_x, scr_y - p_y) // 6)))
                win32_drag_cursor(p_x, p_y, scr_x, scr_y, self.screen_w, self.screen_h, steps=steps)
                self.last_drag_px = (scr_x, scr_y)
                self.last_triggered_action = f"{label} [{step2_txt}]"
                if (
                    active_mode == 4
                    and self.use_ppt_com_shapes
                    and (now_t - self._last_com_erase_check_t) >= 0.12
                ):
                    self._last_com_erase_check_t = now_t
                    n_del = erase_ppt_com_shapes_at_cursor(
                        scr_x, scr_y, self.screen_w, self.screen_h, radius_px=30.0
                    )
                    if n_del > 0:
                        self.last_triggered_action = f"POINT_2F [XOA {n_del} HINH AI]"

        return scr_x, scr_y
