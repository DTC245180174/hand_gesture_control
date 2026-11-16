"""
BƯỚC 1C (AIR-INK): CÔNG CỤ THU THẬP MẪU HÌNH VẼ KHÔNG GIAN QUA WEBCAM (9 LỚP HÌNH)
====================================================================================
Sử dụng chính Model 1 (StaticPoseNet) ở chế độ 1 ngón trỏ (`point_1f`) + bộ lọc
OneEuroFilter2D + Dwell Pen-Up (0.40s tách nét) để thu thập mẫu hình vẽ thật
trên không trung, rasterize về ảnh 64x64 chuẩn và lưu vào:
`data/air_ink/raw_webcam/<class_name>/airink_<session>_<timestamp>.npz`.
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common.config import (
    AIR_INK_IMG_SIZE,
    AIR_INK_RAW_DIR,
    AIR_INK_SHAPE_CLASSES,
    MODELS_DIR,
)
from common.mediapipe_tracker import MediaPipeHandTracker, ThreadedCamera
from common.win32_input import OneEuroFilter2D
from step2_clean_diagnose.feature_extractor import extract_static_pose_features
from step2_clean_diagnose.ink_rasterizer import rasterize_strokes_to_64x64
from step4_model_selection.models import build_pose_model


def count_air_ink_samples() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cname in AIR_INK_SHAPE_CLASSES:
        cdir = AIR_INK_RAW_DIR / cname
        counts[cname] = len(list(cdir.glob("*.npz"))) if cdir.exists() else 0
    return counts


def run_air_ink_collector(camera_id: int = 0) -> None:
    for cname in AIR_INK_SHAPE_CLASSES:
        (AIR_INK_RAW_DIR / cname).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_model = None
    pose_path = MODELS_DIR / "best_pose_model.pt"
    if pose_path.exists():
        p_ckpt = torch.load(pose_path, map_location=device, weights_only=False)
        pose_model = build_pose_model(
            in_channels=p_ckpt["in_channels"], num_classes=p_ckpt["num_classes"]
        ).to(device)
        pose_model.load_state_dict(p_ckpt["model_state_dict"])
        pose_model.eval()
    else:
        print("[!] Không tìm thấy best_pose_model.pt — tạm giữ chế độ vẽ mỗi khi giơ tay.")

    tracker = MediaPipeHandTracker(min_detection_confidence=0.55, min_tracking_confidence=0.50)
    cap = ThreadedCamera(camera_id=camera_id, width=1280, height=720, fps=30)
    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(f"Không thể mở Webcam (camera_id={camera_id}).")

    session_id = f"sess_{int(time.time())}"
    class_idx = 0
    counts = count_air_ink_samples()
    filter_2d = OneEuroFilter2D(min_cutoff=1.1, beta=0.02, d_cutoff=1.0)

    strokes: List[List[Tuple[float, float]]] = []
    cur_stroke: List[Tuple[float, float]] = []
    pen_down = False
    last_anchor_pt: Optional[Tuple[float, float]] = None
    last_move_t = time.perf_counter()
    saved_history: List[Path] = []
    status_msg = "Gio 1 ngon tro va di chuyen de viet | Dung nhe 0.40s de tach net | [SPACE] Luu mau"

    dwell_pen_up_sec = 0.40
    move_thresh_norm = 0.0045
    win_name = "Air-Ink Collector - Step 1C ([N/P] Doi lop, [SPACE] Luu, [C] Xoa, [Q] Thoat)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    try:
        empty_streak = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                empty_streak += 1
                if empty_streak > 40:
                    break
                if cv2.waitKey(15) & 0xFF in (ord("q"), 27):
                    break
                continue
            empty_streak = 0

            now = time.perf_counter()
            frame_m = cv2.flip(frame, 1)
            h, w, _ = frame_m.shape
            aspect = float(w) / max(float(h), 1.0)
            rgb = cv2.cvtColor(frame_m, cv2.COLOR_BGR2RGB)
            lms = tracker.extract_landmarks(rgb)

            is_point_1f = False
            tip_px: Optional[Tuple[int, int]] = None
            if lms is not None:
                MediaPipeHandTracker.draw_skeleton(frame_m, lms)
                if pose_model is not None:
                    model_lms = lms.copy()
                    model_lms[:, 0] = 1.0 - model_lms[:, 0]
                    feat = extract_static_pose_features(model_lms, aspect=aspect)
                    with torch.no_grad():
                        pr = F.softmax(pose_model(torch.from_numpy(feat[None]).to(device)), dim=-1).cpu().numpy()[0]
                    is_point_1f = int(pr.argmax()) == 1 and float(pr[1]) >= 0.65
                else:
                    is_point_1f = True

                sx, sy = filter_2d((float(lms[8, 0]), float(lms[8, 1])), now)
                sx = float(np.clip(sx, 0.0, 1.0))
                sy = float(np.clip(sy, 0.0, 1.0))
                tip_px = (int(sx * w), int(sy * h))

                if is_point_1f:
                    if last_anchor_pt is None:
                        last_anchor_pt = (sx, sy)
                        last_move_t = now
                    dist = math.hypot(sx - last_anchor_pt[0], sy - last_anchor_pt[1])
                    if dist >= move_thresh_norm:
                        last_anchor_pt = (sx, sy)
                        last_move_t = now
                        if not pen_down:
                            pen_down = True
                            cur_stroke = [(sx, sy)]
                        else:
                            cur_stroke.append((sx, sy))
                    elif pen_down and (now - last_move_t) >= dwell_pen_up_sec:
                        if len(cur_stroke) >= 2:
                            strokes.append(cur_stroke)
                        cur_stroke = []
                        pen_down = False
                else:
                    if pen_down and len(cur_stroke) >= 2:
                        strokes.append(cur_stroke)
                    cur_stroke = []
                    pen_down = False
                    last_anchor_pt = None
            else:
                if pen_down and len(cur_stroke) >= 2:
                    strokes.append(cur_stroke)
                cur_stroke = []
                pen_down = False
                last_anchor_pt = None
                filter_2d.reset()

            # Vẽ các nét hiện tại lên màn hình
            all_render = [np.array(s, dtype=np.float32) for s in strokes if len(s) >= 2]
            if len(cur_stroke) >= 2:
                all_render.append(np.array(cur_stroke, dtype=np.float32))

            for st in all_render:
                pts_px = np.column_stack([st[:, 0] * w, st[:, 1] * h]).astype(np.int32)
                for i in range(len(pts_px) - 1):
                    cv2.line(frame_m, tuple(pts_px[i]), tuple(pts_px[i + 1]), (0, 235, 255), 4, cv2.LINE_AA)

            if tip_px is not None:
                col = (0, 255, 120) if is_point_1f else (140, 140, 140)
                cv2.circle(frame_m, tip_px, 10, col, 2, cv2.LINE_AA)

            # Xem trước ảnh rasterized 64x64 ở góc phải
            img64 = rasterize_strokes_to_64x64(all_render, size=AIR_INK_IMG_SIZE, thickness=2)
            preview = cv2.resize(img64, (160, 160), interpolation=cv2.INTER_NEAREST)
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            px0, py0 = w - 180, 95
            frame_m[py0 : py0 + 160, px0 : px0 + 160] = preview_bgr
            cv2.rectangle(frame_m, (px0, py0), (px0 + 160, py0 + 160), (0, 220, 255), 2)
            cv2.putText(frame_m, "64x64 INPUT", (px0 + 25, py0 + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

            cname = AIR_INK_SHAPE_CLASSES[class_idx]
            cv2.rectangle(frame_m, (0, 0), (w, 82), (18, 22, 32), -1)
            cv2.putText(
                frame_m,
                f"LOP [{class_idx + 1}/{len(AIR_INK_SHAPE_CLASSES)}]: {cname} | Da thu: {counts[cname]} mau | Session: {session_id}",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 240, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame_m,
                f"{status_msg} | [N/P] Doi lop | [C] Xoa net | [X] Xoa mau cuoi | [Q] Thoat",
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA,
            )

            cv2.imshow(win_name, frame_m)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                class_idx = (class_idx + 1) % len(AIR_INK_SHAPE_CLASSES)
            elif key == ord("p"):
                class_idx = (class_idx - 1) % len(AIR_INK_SHAPE_CLASSES)
            elif key == ord("c"):
                strokes.clear()
                cur_stroke.clear()
                pen_down = False
                status_msg = "Da xoa bang ve."
            elif key == ord("x") and saved_history:
                last_p = saved_history.pop()
                if last_p.exists():
                    last_p.unlink()
                    counts[last_p.parent.name] = max(0, counts.get(last_p.parent.name, 1) - 1)
                    status_msg = f"Da xoa mau vua luu: {last_p.name}"
            elif key == ord(" "):
                if len(all_render) == 0 or int(img64.max()) == 0:
                    status_msg = "[!] Chua co net ve de luu."
                else:
                    ts_ms = int(time.time() * 1000)
                    out_path = AIR_INK_RAW_DIR / cname / f"airink_{session_id}_{ts_ms}.npz"
                    np.savez_compressed(
                        out_path,
                        img64=img64.astype(np.uint8),
                        strokes=np.array(all_render, dtype=object),
                        class_name=np.array(cname),
                        session_id=np.array(session_id),
                    )
                    counts[cname] = counts.get(cname, 0) + 1
                    saved_history.append(out_path)
                    strokes.clear()
                    cur_stroke.clear()
                    pen_down = False
                    status_msg = f"DA LUU [{cname}] -> Tong {counts[cname]} mau!"
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_air_ink_collector()
