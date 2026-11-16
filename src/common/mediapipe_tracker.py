"""
MODULE TRÍCH XUẤT KHUNG XƯƠNG BÀN TAY (MEDIAPIPE HAND TRACKER)
==============================================================
Chỉ có duy nhất 1 nhiệm vụ: Phát hiện bàn tay và trích xuất toạ độ 21 khớp 3D (x, y, z).
Tương thích 100% cả `mp.solutions.hands` cổ điển lẫn `mp.tasks.vision.HandLandmarker` (Python 3.14).
Tuyệt đối KHÔNG chứa bộ phân loại cử chỉ tự động (tuân thủ nguyên tắc mô hình sequence tự huấn luyện).
"""

from __future__ import annotations
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from common.config import HAND_CONNECTIONS, HAND_LANDMARKER_TASK_PATH, HAND_LANDMARKER_TASK_URL, MODELS_DIR

INFER_W = 640


class ThreadedCamera:
    """
    Bộ đọc Camera đa luồng (mặc định 30 FPS khớp dữ liệu IPN HandS, DirectShow MJPG, BUFFERSIZE=1)
    giúp vòng lặp suy luận chính không bao giờ bị nghẽn I/O.
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> None:
        if os.name == "nt":
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_id)
        else:
            self.cap = cv2.VideoCapture(camera_id)

        self.opened = self.cap.isOpened()
        if self.opened:
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.frame: Optional[np.ndarray] = None
        self.fid: int = 0
        self.read_id: int = -1
        self.lock = threading.Lock()
        self.running = self.opened
        if self.opened:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while self.running:
            ret, f = self.cap.read()
            if ret and f is not None:
                with self.lock:
                    self.frame = f
                    self.fid += 1
            else:
                time.sleep(0.005)

    def isOpened(self) -> bool:
        return self.opened and self.running

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Lần đọc đầu tiên chờ tối đa 3.5 giây để phần cứng Camera DirectShow khởi động xong
        max_wait = 3.5 if self.read_id == -1 else 0.04
        deadline = time.perf_counter() + max_wait
        while self.running:
            with self.lock:
                if self.frame is not None and (self.fid != self.read_id or time.perf_counter() >= deadline):
                    self.read_id = self.fid
                    return True, self.frame.copy()
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.005)
        with self.lock:
            if self.frame is not None:
                self.read_id = self.fid
                return True, self.frame.copy()
            return False, None

    def release(self) -> None:
        self.running = False
        if self.opened:
            try:
                self.cap.release()
            except Exception:
                pass


class MediaPipeHandTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
    ) -> None:
        self.backend = "unknown"
        self.hands_legacy = None
        self.landmarker_task = None
        self._last_ts_ms = 0
        self.smooth_pts: Optional[np.ndarray] = None

        # Thử nghiệm Backend 1: MediaPipe Solutions (Chuẩn MediaPipe < 0.10.14)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.hands_legacy = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    model_complexity=1,
                )
                self.backend = "solutions"
                return
        except Exception:
            pass

        # Backend 2: MediaPipe Tasks API (Hỗ trợ chính thức Python 3.14+)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._ensure_task_model_downloaded()
            base_options = mp_python.BaseOptions(
                model_asset_path=str(HAND_LANDMARKER_TASK_PATH)
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.landmarker_task = vision.HandLandmarker.create_from_options(options)
            self.mp = mp
            self.backend = "tasks"
            return
        except Exception as e:
            raise RuntimeError(
                f"Không thể khởi tạo MediaPipe Hand Tracker bằng cả 2 backend solutions và tasks: {e}"
            )

    @staticmethod
    def _ensure_task_model_downloaded() -> None:
        if HAND_LANDMARKER_TASK_PATH.exists() and HAND_LANDMARKER_TASK_PATH.stat().st_size > 1_000_000:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[MEDIAPIPE TASKS] Đang tải mô hình 21 khớp -> {HAND_LANDMARKER_TASK_PATH} ...")
        urllib.request.urlretrieve(HAND_LANDMARKER_TASK_URL, str(HAND_LANDMARKER_TASK_PATH))
        print("[MEDIAPIPE TASKS] Đã tải xong hand_landmarker.task!")

    def _apply_adaptive_smoothing(self, raw_pts: np.ndarray) -> np.ndarray:
        """Lọc nhiễu thích ứng theo vận tốc cho 21 tọa độ khớp tay (kế thừa từ SIC_AI_Handpoint)."""
        if self.smooth_pts is None:
            self.smooth_pts = raw_pts.copy()
        else:
            delta = np.linalg.norm(raw_pts[:, :2] - self.smooth_pts[:, :2], axis=1, keepdims=True)
            alpha = np.clip(0.62 - delta * 18.0, 0.12, 0.62)
            self.smooth_pts = self.smooth_pts * alpha + raw_pts * (1.0 - alpha)
        return self.smooth_pts.copy()

    def extract_landmarks(self, rgb_frame: np.ndarray, smooth: bool = True) -> Optional[np.ndarray]:
        """
        Trích xuất mảng toạ độ chuẩn hoá (21, 3) chứa [x, y, z] của 21 khớp bàn tay.
        Giữ nguyên tỷ lệ khung hình gốc (aspect ratio) khi thu nhỏ chiều rộng về <= 640px
        để bàn tay không bao giờ bị méo tỷ lệ trên camera 4:3 hoặc 16:9.
        Trả về None nếu khung hình không phát hiện thấy bàn tay.
        """
        h_in, w_in = rgb_frame.shape[:2]
        if w_in > INFER_W:
            infer_h = max(1, int(round(h_in * (float(INFER_W) / float(w_in)))))
            infer_rgb = cv2.resize(rgb_frame, (INFER_W, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_rgb = rgb_frame

        if self.backend == "solutions":
            infer_rgb.flags.writeable = False
            results = self.hands_legacy.process(infer_rgb)
            infer_rgb.flags.writeable = True
            if not results.multi_hand_landmarks:
                self.smooth_pts = None
                return None
            hand_lms = results.multi_hand_landmarks[0].landmark
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
            return self._apply_adaptive_smoothing(raw_pts) if smooth else raw_pts

        # Backend Tasks API
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(infer_rgb),
        )
        now_ms = int(time.perf_counter() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms

        result = self.landmarker_task.detect_for_video(mp_image, now_ms)
        if not result.hand_landmarks:
            self.smooth_pts = None
            return None
        hand_lms = result.hand_landmarks[0]
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
        return self._apply_adaptive_smoothing(raw_pts) if smooth else raw_pts

    @staticmethod
    def draw_skeleton(
        frame_bgr: np.ndarray,
        lms_21x3: np.ndarray,
        finger_states: Optional[np.ndarray] = None,
    ) -> None:
        """Vẽ khung xương 21 khớp bàn tay sắc nét kiểu SIC_AI_Handpoint (hiển thị rõ ngón đang mở/gập)."""
        h, w, _ = frame_bgr.shape
        pts_px = [
            (int(np.clip(lms_21x3[i, 0] * w, 0, w - 1)), int(np.clip(lms_21x3[i, 1] * h, 0, h - 1)))
            for i in range(21)
        ]
        for u, v in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts_px[u], pts_px[v], (0, 200, 80), 2, cv2.LINE_AA)

        tip_lookup = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}
        mcp_set = {0, 5, 9, 13, 17}
        for i, (px, py) in enumerate(pts_px):
            if i in tip_lookup:
                is_up = bool(finger_states[tip_lookup[i]]) if finger_states is not None else True
                fill = (0, 255, 255) if is_up else (0, 140, 255)
                r = 7 if is_up else 5
                cv2.circle(frame_bgr, (px, py), r, fill, -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (px, py), r + 1, (255, 255, 255), 1, cv2.LINE_AA)
            elif i in mcp_set:
                cv2.circle(frame_bgr, (px, py), 5, (255, 160, 0), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

    def close(self) -> None:
        if self.hands_legacy is not None:
            self.hands_legacy.close()
        if self.landmarker_task is not None:
            self.landmarker_task.close()

