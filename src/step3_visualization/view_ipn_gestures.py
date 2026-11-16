"""
TRÌNH PHÁT VIDEO MÔ PHỎNG CỬ CHỈ GỐC TỪ BỘ DỮ LIỆU IPN HANDS (GESTURE REPLAY VIEWER)
====================================================================================
Giúp bạn xem trực quan dưới dạng Video (30 FPS hoặc quay chậm Slow-Motion) cách những
người tình nguyện trong bộ dữ liệu IPN HandS thực hiện từng cử chỉ trước camera:
    [1] swipe_left  (TLT - Throw Left  : Quẹt tay sang trái)
    [2] swipe_right (TRT - Throw Right : Quẹt tay sang phải)
    [3] point_1f    (P1F - Point 1F    : Giơ 1 ngón trỏ)
    [4] point_2f    (P2F - Point 2F    : Giơ 2 ngón trỏ + giữa)
    [5] zoom_in     (ZIN - Zoom In     : Chụm ngón cái-trỏ rồi bung xoè rộng)
    [6] zoom_out    (ZOT - Zoom Out    : Xoè rộng ngón cái-trỏ rồi khép chụm vào)
    [7] click       (Click             : Cử chỉ bấm / chạm chọn)
    [0] idle        (NOG - Non-gesture : Trạng thái nghỉ)

Phím điều khiển trên cửa sổ Video:
    - Phím [0..7] : Chọn lớp cử chỉ muốn xem
    - Phím [N]    : Chuyển sang mẫu video khác của cùng cử chỉ đó
    - Phím [S]    : Bật / Tắt chế độ quay chậm (Slow-Motion 10 FPS vs Thực tế 30 FPS)
    - Phím [M]    : Đảo góc nhìn (Góc nhìn Gương giống Webcam của bạn vs Góc nhìn Camera gốc)
    - Phím [Q/ESC]: Thoát
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

from common.config import HAND_CONNECTIONS, NUM_CLASSES, PROCESSED_DIR

CLASS_INFO = {
    0: ("0: idle (NOG - Trang thai nghi)", "De tay tu nhien hoac giu yen"),
    1: ("1: swipe_left (TLT - Next Slide)", "Mo ban tay, gat ngang tu PHAI sang TRAI"),
    2: ("2: swipe_right (TRT - Prev Slide)", "Mo ban tay, gat ngang tu TRAI sang PHAI"),
    3: ("3: point_1f (P1F - But viet Slide)", "Nam 3 ngon ut/ap ut/giua, chi duoi 1 NGON TRO"),
    4: ("4: point_2f (P2F - Xoa muc Slide)", "Duoi thang 2 NGON (Tro + Giua), gap 2 ngon ut"),
    5: ("5: zoom_in (ZIN - Phong to)", "Giu yen co tay: CHUM ngon cai & tro -> BUNG XOE RONG"),
    6: ("6: zoom_out (ZOT - Thu nho)", "Giu yen co tay: XOE RONG ngon cai & tro -> KHEP CHUM SAT"),
    7: ("7: click (OP2 - Click Chuot Trai)", "Giu yen co tay: CHOP MO XOE BAN TAY 2 LAN LIEN TIEP (Mo - Khep - Mo)"),
}


def draw_gesture_frame(
    pts_21x3: np.ndarray,
    wrist_trail: List[Tuple[int, int]],
    tip8_trail: List[Tuple[int, int]],
    cls_id: int,
    sample_idx: int,
    total_samples: int,
    frame_idx: int,
    mirror_view: bool,
    slow_mo: bool,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)

    # Vẽ lưới toạ độ mờ để dễ quan sát vị trí và biên độ di chuyển của tay
    for gx in range(0, width, 160):
        cv2.line(canvas, (gx, 95), (gx, height - 50), (45, 45, 45), 1)
    for gy in range(95, height - 50, 120):
        cv2.line(canvas, (0, gy), (width, gy), (45, 45, 45), 1)

    pts = pts_21x3.copy()
    if mirror_view:
        # Lật gương giống hệt như khi bạn nhìn mình trong cửa sổ Webcam ở Bước 7
        pts[:, 0] = 1.0 - pts[:, 0]

    pts_px = [
        (
            int(np.clip(pts[i, 0] * width, 0, width - 1)),
            int(np.clip(pts[i, 1] * height, 0, height - 1)),
        )
        for i in range(21)
    ]

    wrist_trail.append(pts_px[0])
    tip8_trail.append(pts_px[8])

    # Vẽ quỹ đạo di chuyển của cổ tay (khớp 0 - màu cam) và đầu ngón trỏ (khớp 8 - màu xanh lá)
    for k in range(1, len(wrist_trail)):
        cv2.line(canvas, wrist_trail[k - 1], wrist_trail[k], (0, 140, 255), 3, cv2.LINE_AA)
    for k in range(1, len(tip8_trail)):
        cv2.line(canvas, tip8_trail[k - 1], tip8_trail[k], (80, 220, 80), 2, cv2.LINE_AA)

    # Vẽ đường nét đứt nối giữa đầu ngón cái (khớp 4) và đầu ngón trỏ (khớp 8) để nhìn rõ Zoom In / Out
    if cls_id in (5, 6):
        dist_px = int(np.hypot(pts_px[4][0] - pts_px[8][0], pts_px[4][1] - pts_px[8][1]))
        cv2.line(canvas, pts_px[4], pts_px[8], (0, 255, 255), 2, cv2.LINE_AA)
        mid_x = (pts_px[4][0] + pts_px[8][0]) // 2
        mid_y = (pts_px[4][1] + pts_px[8][1]) // 2 - 12
        cv2.putText(
            canvas,
            f"Khoang cach: {dist_px}px",
            (mid_x - 60, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Vẽ các đoạn xương nối 21 khớp
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(canvas, pts_px[start_idx], pts_px[end_idx], (200, 200, 200), 2, cv2.LINE_AA)

    # Vẽ 21 khớp điểm mốc
    for i, pt in enumerate(pts_px):
        color = (0, 140, 255) if i == 0 else ((80, 220, 80) if i in (4, 8, 12, 16, 20) else (255, 255, 255))
        radius = 7 if i in (0, 4, 8) else 4
        cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)

    # Thanh Header thông tin cử chỉ
    cv2.rectangle(canvas, (0, 0), (width, 85), (20, 20, 20), -1)
    cv2.line(canvas, (0, 85), (width, 85), (0, 200, 255), 2)

    title, desc = CLASS_INFO[cls_id]
    cv2.putText(canvas, f"DANG XEM: {title}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
    cv2.putText(canvas, f"HD THUC HIEN: {desc}", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

    status_r = f"Mau: #{sample_idx + 1}/{total_samples}  |  Frame: {frame_idx + 1:02d}/30  |  {'SLOW-MO 10FPS' if slow_mo else '30 FPS REALTIME'}"
    cv2.putText(canvas, status_r, (width - 560, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

    mirror_txt = "GOC NHIN: GUONG (Webcam)" if mirror_view else "GOC NHIN: CAMERA GOC (IPN HandS)"
    cv2.putText(canvas, mirror_txt, (width - 560, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 50), 1)

    # Thanh Footer phím bấm điều khiển
    cv2.rectangle(canvas, (0, height - 42), (width, height), (15, 15, 15), -1)
    cv2.line(canvas, (0, height - 42), (width, height - 42), (80, 80, 80), 1)
    controls = "[0..7]: Chon cu chi | [N]: Mau tiep theo | [S]: Bat/Tat Slow-Mo | [M]: Doi Goc Nhin Guong | [Q/ESC]: Thoat"
    cv2.putText(canvas, controls, (25, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    return canvas


def run_gesture_replay(processed_dir: Path = PROCESSED_DIR) -> None:
    raw30_test_path = processed_dir / "raw30_test.npy"
    y_test_path = processed_dir / "y_test.npy"

    if not raw30_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu toa do goc tai: {raw30_test_path}. "
            "Hay dam bao da chay xong Buoc 2!"
        )

    print("[*] Dang nap tap du lieu toa do goc (T=30 frames)...")
    raw_test = np.load(raw30_test_path)
    y_test = np.load(y_test_path)

    # Gom cac mau theo tung lop
    class_samples = {cid: [] for cid in range(NUM_CLASSES)}
    for idx, cid in enumerate(y_test):
        if int(cid) in class_samples:
            class_samples[int(cid)].append(raw_test[idx].reshape(30, 21, 3))

    for cid in range(NUM_CLASSES):
        print(f"  - Lop {CLASS_INFO[cid][0]}: Co tong cong {len(class_samples[cid])} mau video mau.")

    cur_cls = 1
    sample_indices = {cid: 0 for cid in range(NUM_CLASSES)}
    mirror_view = True
    slow_mo = False

    win_name = "IPN HandS - Visual Gesture Replay (Nhan [Q] de thoat)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    print("\n" + "=" * 80)
    print("DA MO CUA SO VIDEO MO PHONG!")
    print("Ban hay bam cac phim sau tren cua so video de trai nghiem:")
    print("  - Phim [1]: Xem cu chi Quet Trai (Next Slide)")
    print("  - Phim [2]: Xem cu chi Quet Phai (Prev Slide)")
    print("  - Phim [3]: Xem cu chi Chi 1 Ngon Tro (But ve Slide)")
    print("  - Phim [4]: Xem cu chi Chi 2 Ngon (Xoa muc Slide)")
    print("  - Phim [5]: Xem cu chi Zoom In (Phong to Slide)")
    print("  - Phim [6]: Xem cu chi Zoom Out (Thu nho Slide)")
    print("  - Phim [7]: Xem cu chi Click 1 Ngon Tro (Left Click)")
    print("  - Phim [0]: Xem trang thai Dung Yen (Idle)")
    print("  - Phim [N]: Chuyen sang nguoi khac/video khac")
    print("  - Phim [S]: Bat/Tat che do quay cham Slow-Mo de nhin ro tung frame")
    print("  - Phim [M]: Doi goc nhin (Webcam guong vs goc)")
    print("=" * 80 + "\n")

    running = True
    while running:
        samples = class_samples[cur_cls]
        if not samples:
            cur_cls = (cur_cls + 1) % NUM_CLASSES
            continue

        s_idx = sample_indices[cur_cls] % len(samples)
        seq_30x21x3 = samples[s_idx]

        wrist_trail: List[Tuple[int, int]] = []
        tip8_trail: List[Tuple[int, int]] = []

        for f_idx in range(30):
            frame_pts = seq_30x21x3[f_idx]
            canvas = draw_gesture_frame(
                frame_pts,
                wrist_trail,
                tip8_trail,
                cls_id=cur_cls,
                sample_idx=s_idx,
                total_samples=len(samples),
                frame_idx=f_idx,
                mirror_view=mirror_view,
                slow_mo=slow_mo,
            )
            cv2.imshow(win_name, canvas)

            delay_ms = 100 if slow_mo else 33
            key = cv2.waitKey(delay_ms) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                running = False
                break
            elif ord("0") <= key <= ord("7"):
                cur_cls = key - ord("0")
                break
            elif key in (ord("n"), ord("N")):
                sample_indices[cur_cls] = (sample_indices[cur_cls] + 1) % len(samples)
                break
            elif key in (ord("s"), ord("S")):
                slow_mo = not slow_mo
            elif key in (ord("m"), ord("M")):
                mirror_view = not mirror_view
                wrist_trail.clear()
                tip8_trail.clear()

        if running:
            # Dung nhe o frame cuoi 0.3 giay truoc khi lap lai
            cv2.waitKey(300)

    cv2.destroyAllWindows()
    print("[*] Da dong cua so phat lai video.")


if __name__ == "__main__":
    run_gesture_replay()

