"""
BƯỚC 7: TRIỂN KHAI ĐIỀU KHIỂN POWERPOINT THỜI GIAN THỰC (REAL-TIME DEPLOYMENT)
=============================================================================
Điểm chạy chính của hệ thống:
1. Đọc luồng video từ Webcam cá nhân qua MediaPipe Hands.
2. Suy luận 100% qua mô hình học sâu `best_gesture_model.pt`.
3. Tương tác trực tiếp với Microsoft PowerPoint Slide Show:
   - Next / Prev Slide (Right / Left Arrow)
   - Bút vẽ không gian liền nét (Ctrl+P + Hold Mouse)
   - Xoá mực 2 ngón (Ctrl+E + Hold Mouse)
   - Zoom In / Zoom Out (Ctrl + '=' / Ctrl + '-')
   - Cooldown 2.0s chống loạn
   - Cửa sổ PiP nổi xuyên chuột (Click-Through) trên slide.
"""

from __future__ import annotations
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch

from common.config import MODELS_DIR, NUM_AIR_INK_CLASSES
from common.mediapipe_tracker import MediaPipeHandTracker, ThreadedCamera
from common.win32_input import (
    is_powerpoint_foreground,
    maintain_opencv_pip_overlay,
    set_opencv_window_pip_overlay,
)
from step4_model_selection.models import build_air_ink_model, build_model, build_pose_model
from step7_deployment.controller import RealTimeGestureController
from step7_deployment.hud import draw_hud_overlay

_USER32 = ctypes.windll.user32 if os.name == "nt" else None


def run_live_webcam_deployment(
    model: torch.nn.Module,
    device: torch.device,
    model_info: str,
    checkpoint_path: Path,
    camera_id: int = 0,
    conf_threshold: float = 0.85,
    cooldown_sec: float = 2.0,
    start_paused: bool = False,
    pen_draw_click: bool = True,
    start_pip: bool = False,
    pip_corner: str = "bottom_right",
    ppt_only_lock: bool = False,
    bottom_deadzone: bool = True,
    pointer_smoothing: float = 0.60,
    audio_beep: bool = True,
    pose_model: Optional[torch.nn.Module] = None,
    air_ink_model: Optional[torch.nn.Module] = None,
    point_1f_mode: Optional[str] = None,
) -> None:
    tracker = MediaPipeHandTracker(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
    )

    controller = RealTimeGestureController(
        model=model,
        device=device,
        model_info=model_info,
        conf_threshold=conf_threshold,
        vote_window=5,
        vote_required=3,
        cooldown_sec=cooldown_sec,
        live_ppt=not start_paused,
        pen_draw_click=pen_draw_click,
        ppt_only_lock=ppt_only_lock,
        pointer_smoothing=pointer_smoothing,
        audio_beep=audio_beep,
        bottom_deadzone=bottom_deadzone,
        pose_model=pose_model,
        air_ink_model=air_ink_model,
        point_1f_mode=point_1f_mode,
    )

    cap = ThreadedCamera(camera_id=camera_id, width=1280, height=720, fps=30)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"Không thể mở Webcam (camera_id={camera_id}).")

    window_name = "PowerPoint Hand Control - Step 7 (Press Q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if start_pip:
        is_fullscreen = False
        is_pip_overlay = True
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True
        is_pip_overlay = False
    pip_initialized = False
    last_f8_down = False
    last_f7_down = False
    prev_frame_time = time.perf_counter()
    fps = 30.0

    print("=" * 84)
    print("BƯỚC 7: HỆ THỐNG ĐIỀU KHIỂN POWERPOINT THỜI GIAN THỰC (100% TRAINED MODEL)")
    print(f"  - Trọng số Model    : {checkpoint_path}")
    print(f"  - Cấu hình Model    : {model_info}")
    print(f"  - Cooldown Debounce : {cooldown_sec}s | Smoothing: {int(pointer_smoothing*100)}%")
    print(f"  - PowerPoint Lock   : {'ON' if ppt_only_lock else 'OFF'} | Dead-Zone 15%: {'ON' if bottom_deadzone else 'OFF'}")
    print("  - Nhấn [Q], [ESC] hoặc phím toàn cục [F9] để dừng khẩn cấp (Kill-switch)")
    print("=" * 84)

    try:
        empty_streak = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                empty_streak += 1
                if empty_streak > 40:
                    print("[!] Mất tín hiệu khung hình từ Webcam, dừng vòng lặp an toàn.")
                    break
                if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                    break
                continue
            empty_streak = 0

            now = time.perf_counter()
            dt = max(now - prev_frame_time, 1e-4)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_frame_time = now

            # Kiểm tra trạng thái cửa sổ PowerPoint
            ppt_active = is_powerpoint_foreground(own_window_prefix="PowerPoint Hand Control")
            controller.ppt_window_active = ppt_active

            # Tự động chuyển PiP khi người dùng bật PowerPoint
            if not is_pip_overlay and ppt_active:
                is_pip_overlay = True
                is_fullscreen = False
                pip_initialized = True
                set_opencv_window_pip_overlay(window_name, True, controller.screen_w, controller.screen_h, corner=pip_corner)
                print(f"[AUTO PIP OVERLAY] Phát hiện PowerPoint -> Đã ghim camera góc ({pip_corner}) nổi xuyên chuột 100%!")

            frame_mirror = cv2.flip(frame, 1)
            h0, w0 = frame_mirror.shape[:2]
            if w0 < 1280:
                target_h = max(1, int(round(h0 * (1280.0 / max(1, w0)))))
                frame_mirror = cv2.resize(frame_mirror, (1280, target_h), interpolation=cv2.INTER_LINEAR)
            h, w, _ = frame_mirror.shape
            frame_aspect = float(w) / max(float(h), 1.0)

            rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
            lms_21x3 = tracker.extract_landmarks(rgb)

            pts_63: Optional[np.ndarray] = None
            index_tip_px: Optional[Tuple[int, int]] = None
            in_deadzone = False

            if lms_21x3 is not None:
                # Kiểm tra Vùng nghỉ hạ tay 15% mép dưới (cổ tay khớp 0 có y > 0.85)
                if bottom_deadzone and float(lms_21x3[0, 1]) > 0.85:
                    in_deadzone = True
                    pts_63 = None
                else:
                    model_lms = lms_21x3.copy()
                    model_lms[:, 0] = 1.0 - model_lms[:, 0]
                    pts_63 = model_lms.reshape(63).astype(np.float32)

            state = controller.push_frame_landmarks(pts_63, now, aspect=frame_aspect)
            confirmed_cls = int(state["confirmed_class"])
            if lms_21x3 is not None:
                MediaPipeHandTracker.draw_skeleton(frame_mirror, lms_21x3)

            if lms_21x3 is not None and not in_deadzone:
                raw_cls = int(state["raw_class"])
                if confirmed_cls == 4:
                    px_norm = float(0.5 * (lms_21x3[8, 0] + lms_21x3[12, 0]))
                    py_norm = float(0.5 * (lms_21x3[8, 1] + lms_21x3[12, 1]))
                else:
                    px_norm = float(lms_21x3[8, 0])
                    py_norm = float(lms_21x3[8, 1])

                index_tip_px = (int(px_norm * w), int(py_norm * h))
                # Khi đang thực hiện chớp mở tay 2 lần (OP2 - lớp 7), giữ nguyên vị trí tâm chuột đã ngắm
                # để chuyển động xoè/khép ngón không làm lệch con trỏ khỏi nút bấm trên Slide
                if raw_cls != 7 and confirmed_cls != 7:
                    controller.update_spatial_pointer((px_norm, py_norm), active_mode=confirmed_cls)

            hud_frame = draw_hud_overlay(
                frame_mirror, state, controller, fps, index_tip_px=index_tip_px, in_deadzone=in_deadzone
            )
            cv2.imshow(window_name, hud_frame)
            if is_pip_overlay and not pip_initialized:
                pip_initialized = True
                set_opencv_window_pip_overlay(window_name, True, controller.screen_w, controller.screen_h, corner=pip_corner)
            elif is_pip_overlay:
                maintain_opencv_pip_overlay(window_name, controller.screen_w, controller.screen_h, corner=pip_corner)

            f9_down = bool(_USER32 is not None and (_USER32.GetAsyncKeyState(0x78) & 0x8000))
            f8_down = bool(_USER32 is not None and (_USER32.GetAsyncKeyState(0x77) & 0x8000))
            f7_down = bool(_USER32 is not None and (_USER32.GetAsyncKeyState(0x76) & 0x8000))
            toggle_pip_req = f8_down and not last_f8_down
            cycle_ai_req = f7_down and not last_f7_down
            last_f8_down = f8_down
            last_f7_down = f7_down

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or f9_down:
                print("[KILL-SWITCH] Đã dừng chương trình an toàn.")
                break
            elif key == ord("p") or toggle_pip_req:
                is_pip_overlay = not is_pip_overlay
                if is_pip_overlay:
                    pip_initialized = True
                    is_fullscreen = False
                set_opencv_window_pip_overlay(
                    window_name, is_pip_overlay, controller.screen_w, controller.screen_h, corner=pip_corner
                )
                if not is_pip_overlay:
                    is_fullscreen = True
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            elif key == ord("f"):
                is_pip_overlay = False
                set_opencv_window_pip_overlay(window_name, False, controller.screen_w, controller.screen_h, corner=pip_corner)
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                if not is_fullscreen:
                    cv2.resizeWindow(window_name, 1280, 720)
            elif key == ord(" "):
                controller.live_ppt = not controller.live_ppt
                controller._reset_ink_cluster()
                controller._release_pen_if_needed(reset_tool=True)
                print(f"[MODE] Gửi phím PowerPoint: {'LIVE' if controller.live_ppt else 'PAUSED'}")
            elif key == ord("d"):
                new_mode = controller.cycle_point_1f_mode()
                print(f"[1F MODE] Chế độ point_1f: {new_mode}")
            elif key == ord("m") or cycle_ai_req:
                new_mode = controller.cycle_ai_submode()
                print(f"[AI MODE] Chế độ 1 ngón trỏ: {new_mode}")
            elif key in (ord("1"), ord("2"), ord("3")):
                controller.select_top_candidate(key - ord("0"))
            elif key == ord("c"):
                controller.clear_ink_preview()
    finally:
        controller.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def run_deployment_pipeline(
    camera_id: int = 0,
    conf_threshold: float = 0.85,
    cooldown_sec: float = 2.0,
    start_paused: bool = False,
    pen_draw_click: bool = True,
    start_pip: bool = False,
    pip_corner: str = "bottom_right",
    ppt_only_lock: bool = False,
    bottom_deadzone: bool = True,
    pointer_smoothing: float = 0.60,
    audio_beep: bool = True,
    point_1f_mode: Optional[str] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = MODELS_DIR / "best_gesture_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy trọng số tại {checkpoint_path}. Hãy chạy Step 5 trước.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch_name = ckpt.get("arch_name", "hybrid")
    best_epoch = ckpt.get("best_epoch", 0)
    best_val_acc = ckpt.get("best_metrics", {}).get("val_acc", 0.0)
    model_info = f"{arch_name.upper()} (Ep{best_epoch}, ValAcc={best_val_acc}%)"

    model = build_model(arch_name, input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    pose_model = None
    pose_ckpt_path = checkpoint_path.parent / "best_pose_model.pt"
    if not pose_ckpt_path.exists():
        pose_ckpt_path = MODELS_DIR / "best_pose_model.pt"
    if pose_ckpt_path.exists():
        p_ckpt = torch.load(pose_ckpt_path, map_location=device, weights_only=False)
        pose_model = build_pose_model(
            in_channels=p_ckpt["in_channels"], num_classes=p_ckpt["num_classes"]
        )
        pose_model.load_state_dict(p_ckpt["model_state_dict"])
        pose_model.to(device)
        pose_model.eval()
        p_ep = p_ckpt.get("best_epoch", 0)
        p_acc = p_ckpt.get("best_metrics", {}).get("val_acc", 0.0)
        model_info = f"{model_info} + PoseNet(Ep{p_ep},{p_acc}%)"

    air_ink_model = None
    air_ink_ckpt_path = checkpoint_path.parent / "best_air_ink_model.pt"
    if not air_ink_ckpt_path.exists():
        air_ink_ckpt_path = MODELS_DIR / "best_air_ink_model.pt"
    if air_ink_ckpt_path.exists():
        a_ckpt = torch.load(air_ink_ckpt_path, map_location=device, weights_only=False)
        air_ink_model = build_air_ink_model(
            in_channels=int(a_ckpt.get("in_channels", 1)),
            num_classes=int(a_ckpt.get("num_classes", NUM_AIR_INK_CLASSES)),
            dropout=float(a_ckpt.get("dropout", 0.25)),
        )
        air_ink_model.load_state_dict(a_ckpt["model_state_dict"])
        air_ink_model.to(device)
        air_ink_model.eval()
        a_ep = a_ckpt.get("best_epoch", 0)
        a_acc = a_ckpt.get("best_metrics", {}).get("val_acc", 0.0)
        model_info = f"{model_info} + AirInk(Ep{a_ep},{a_acc}%)"

    run_live_webcam_deployment(
        model=model,
        device=device,
        model_info=model_info,
        checkpoint_path=checkpoint_path,
        camera_id=camera_id,
        conf_threshold=conf_threshold,
        cooldown_sec=cooldown_sec,
        start_paused=start_paused,
        pen_draw_click=pen_draw_click,
        start_pip=start_pip,
        pip_corner=pip_corner,
        ppt_only_lock=ppt_only_lock,
        bottom_deadzone=bottom_deadzone,
        pointer_smoothing=pointer_smoothing,
        audio_beep=audio_beep,
        pose_model=pose_model,
        air_ink_model=air_ink_model,
        point_1f_mode=point_1f_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bước 7: Triển khai điều khiển PowerPoint thời gian thực")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số Webcam (mặc định: 0)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Ngưỡng xác suất tối thiểu từ Model (mặc định: 0.85)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Thời gian Cooldown sau mỗi lệnh (giây, mặc định: 2.0)")
    parser.add_argument("--paused", action="store_true", help="Khởi động ở chế độ PAUSED")
    parser.add_argument(
        "--point-1f-mode",
        type=str,
        default="laser_ai",
        choices=["laser_ai", "laser_pen", "ai_shapes", "pen", "laser"],
        help="Chế độ khi giơ 1 ngón trỏ: laser_ai | laser_pen | ai_shapes | pen | laser",
    )
    args = parser.parse_args()

    run_deployment_pipeline(
        camera_id=args.camera,
        conf_threshold=args.threshold,
        cooldown_sec=args.cooldown,
        start_paused=args.paused,
        pen_draw_click=(args.point_1f_mode != "laser"),
        point_1f_mode=args.point_1f_mode,
    )


if __name__ == "__main__":
    main()

