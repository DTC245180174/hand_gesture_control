"""
BƯỚC 1 (PHẦN B): CÔNG CỤ THU THẬP MẪU WEBCAM TƯƠNG TÁC (HYBRID DATA COLLECTOR)
=============================================================================
Thu thập nhanh các mẫu cử chỉ thực tế từ Webcam cá nhân cho 8 lớp [0..7].
Tự động lưu vào `data/raw_sequences/<class_name>/webcam_<class>_<timestamp>.npy`.
"""

from __future__ import annotations
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from common.config import CLASS_NAMES, GESTURE_GUIDES, RAW_SEQ_DIR
from common.mediapipe_tracker import MediaPipeHandTracker


def count_webcam_samples(raw_seq_dir: str = str(RAW_SEQ_DIR)) -> Dict[int, int]:
    counts = {}
    for cid, cname in enumerate(CLASS_NAMES):
        cdir = os.path.join(raw_seq_dir, cname)
        files = glob.glob(os.path.join(cdir, "webcam_*.npy"))
        counts[cid] = len(files)
    return counts


def run_webcam_collector(target_frames: int = 30, camera_id: int = 0) -> None:
    raw_seq_dir = str(RAW_SEQ_DIR)
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(raw_seq_dir, cname), exist_ok=True)

    tracker = MediaPipeHandTracker(
        min_detection_confidence=0.60,
        min_tracking_confidence=0.50,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        tracker.close()
        print(f"[!] Không thể mở Webcam (camera_id={camera_id}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    win_name = "Thu Thap Mau Webcam - Step 1B (Nhan SPACE de ghi, Q/ESC de thoat)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    active_class_id = 0
    is_recording = False
    is_counting_down = False
    countdown_start_time = 0.0
    countdown_duration = 0.7
    recorded_frames: List[np.ndarray] = []
    saved_history: List[str] = []
    counts = count_webcam_samples(raw_seq_dir)
    status_msg = "San sang. Chon phim [0..7] va nhan [SPACE] de bat dau thu."

    print("=" * 78)
    print("BƯỚC 1B: CÔNG CỤ THU THẬP CỬ CHỈ WEBCAM CÁ NHÂN (HYBRID COLLECTOR - 8 LỚP)")
    print("  - Nhấn [0..7] để chọn lớp cử chỉ | [SPACE] đếm ngược 0.7s & ghi 30 frames")
    print("  - Nhấn [X] để xoá mẫu vừa ghi   | [F] Fullscreen/Window | [Q/ESC] Thoát")
    print("=" * 78)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            now = time.perf_counter()
            frame_mirror = cv2.flip(frame, 1)
            h0, w0 = frame_mirror.shape[:2]
            if w0 < 1280:
                target_h = max(1, int(round(h0 * (1280.0 / max(1, w0)))))
                frame_mirror = cv2.resize(frame_mirror, (1280, target_h), interpolation=cv2.INTER_LINEAR)
            h, w, _ = frame_mirror.shape

            rgb = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2RGB)
            lms_21x3 = tracker.extract_landmarks(rgb)

            pts_63 = np.zeros(63, dtype=np.float32)
            if lms_21x3 is not None:
                MediaPipeHandTracker.draw_skeleton(frame_mirror, lms_21x3)
                # Đồng bộ hệ quy chiếu trục X với dữ liệu IPN HandS và Step 7 Deployment
                model_lms = lms_21x3.copy()
                model_lms[:, 0] = 1.0 - model_lms[:, 0]
                pts_63 = model_lms.reshape(63).astype(np.float32)

            cname = CLASS_NAMES[active_class_id]

            # Vẽ khung nền HUD
            overlay = frame_mirror.copy()
            cv2.rectangle(overlay, (0, 0), (w, 105), (20, 20, 20), -1)
            cv2.rectangle(overlay, (12, 115), (340, 395), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame_mirror, 0.25, 0, frame_mirror)

            # Xử lý đếm ngược (0.7s)
            if is_counting_down:
                elapsed = now - countdown_start_time
                rem = max(0.0, countdown_duration - elapsed)
                status_msg = f"CHUAN BI VI TRI TAY [{cname.upper()}]... {rem:.1f}s"
                if elapsed >= countdown_duration:
                    is_counting_down = False
                    is_recording = True
                    recorded_frames = []

            # Ghi nhận chuỗi 30 frames
            if is_recording:
                recorded_frames.append(pts_63.copy())
                progress = len(recorded_frames)
                status_msg = f">>> DANG GHI [{cname.upper()}]: {progress}/{target_frames} frames <<<"
                bar_w = int((w - 40) * (progress / float(target_frames)))
                cv2.rectangle(frame_mirror, (20, 84), (20 + bar_w, 98), (0, 0, 255), -1)

                if progress >= target_frames:
                    is_recording = False
                    seq_array = np.array(recorded_frames, dtype=np.float32)
                    valid_frames = int(np.sum(np.linalg.norm(seq_array, axis=1) > 1e-5))
                    if valid_frames >= 20:
                        ts = int(time.time() * 1000)
                        fname = f"webcam_{cname}_{ts}.npy"
                        fpath = os.path.join(raw_seq_dir, cname, fname)
                        np.save(fpath, seq_array)
                        saved_history.append(fpath)
                        counts = count_webcam_samples(raw_seq_dir)
                        status_msg = f"DA LUU #{counts[active_class_id]}: {fname} (Nhan [X] neu muon xoa)"
                        print(f"[+] Đã lưu mẫu #{counts[active_class_id]} cho [{active_class_id}] {cname} -> {fname}")
                    else:
                        status_msg = f"HUY MAU: Mat dau tay ({valid_frames}/30 frames hop le)"
                        print(f"[!] Huỷ mẫu vì mất dấu bàn tay ({valid_frames}/30 frames hợp lệ).")
                    recorded_frames = []

            cv2.putText(
                frame_mirror,
                f"LOP [{active_class_id}] {cname.upper()} (Da thu: {counts[active_class_id]}) | {status_msg}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 0, 255) if is_recording else ((0, 215, 255) if is_counting_down else (50, 240, 50)),
                2,
            )
            cv2.putText(
                frame_mirror,
                f"HUONG DAN: {GESTURE_GUIDES[active_class_id]}",
                (20, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (225, 225, 225),
                1,
            )

            # Bảng thống kê số mẫu Webcam từng lớp bên trái
            cv2.putText(
                frame_mirror,
                "SO MAU WEBCAM CA NHAN HIEN CO:",
                (22, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
            )
            for cid, cls_str in enumerate(CLASS_NAMES):
                y_pos = 168 + cid * 26
                prefix = ">>" if cid == active_class_id else "  "
                col = (50, 255, 50) if cid == active_class_id else (210, 210, 210)
                cv2.putText(
                    frame_mirror,
                    f"{prefix} [{cid}] {cls_str:<12}: {counts[cid]:>3d} mau",
                    (22, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    col,
                    2 if cid == active_class_id else 1,
                )

            cv2.putText(
                frame_mirror,
                "[0..7] Chon lop | [SPACE] Ghi 1 mau (0.7s) | [X] Xoa mau vua ghi | [F] Fullscreen | [Q/ESC] Thoat",
                (20, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 235, 235),
                1,
            )

            cv2.imshow(win_name, frame_mirror)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("0") <= key <= ord("7") and not is_recording and not is_counting_down:
                active_class_id = key - ord("0")
                status_msg = f"Da chon lop: [{active_class_id}] {CLASS_NAMES[active_class_id]}"
            elif key == ord(" ") and not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start_time = now
            elif key in (ord("x"), ord("X")) and not is_recording and not is_counting_down:
                if saved_history:
                    last_file = saved_history.pop()
                    if os.path.exists(last_file):
                        os.remove(last_file)
                        counts = count_webcam_samples(raw_seq_dir)
                        status_msg = f"Da xoa mau vua thu: {os.path.basename(last_file)}"
                        print(f"[-] Đã xoá mẫu vừa thu: {os.path.basename(last_file)}")
            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
                if not is_fullscreen:
                    cv2.resizeWindow(win_name, 1280, 720)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_collector()

