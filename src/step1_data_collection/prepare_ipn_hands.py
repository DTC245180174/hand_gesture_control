"""
BƯỚC 1 (PHẦN A): TRÍCH XUẤT PHÂN ĐOẠN CỬ CHỈ TỪ BỘ DỮ LIỆU IPN HANDS
====================================================================
Đọc trực tiếp dữ liệu từ `data/ipn_hands/`:
- `temporal_annotations/*.txt`
- `landmarks_gt/*.json`
Khai thác tối đa dữ liệu thực tế từ 200 videos:
1. Với các phân đoạn dài (`idle`, `point_1f`, `point_2f`): Quét tìm các cửa sổ 32 frames
   mà bàn tay thực sự xuất hiện trong khung hình (lấy cả lúc giơ tĩnh và lúc rê ngón tay vẽ/xoá).
2. Với các cử chỉ động (`swipe_left`, `swipe_right`, `zoom_in`, `zoom_out`) ở tập Train:
   Trích xuất thêm các cửa sổ trượt lệch pha thực tế (`±shift frames` từ chính các frame liền kề
   trong video gốc) để mô hình học sát với cơ chế Sliding Window khi chạy thực tế.
3. Bảo toàn 100% các mẫu `webcam_*.npy` cá nhân của người dùng.
"""

from __future__ import annotations
import csv
import glob
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from common.config import (
    CLASS_NAMES,
    IPN_HANDS_DIR,
    RAW_SEQ_DIR,
)

# Ánh xạ từ class_id trong classes.txt của IPN HandS sang 8 lớp PowerPoint
IPN_ID_TO_PPT_CLASS = {
    0:  ("NOG", "idle"),         # Non-gesture
    7:  ("TLT", "swipe_left"),   # Throw left  -> Next Slide
    8:  ("TRT", "swipe_right"),  # Throw right -> Prev Slide
    1:  ("P1F", "point_1f"),     # Pointing with 1 finger -> Pen / Laser
    2:  ("P2F", "point_2f"),     # Pointing with 2 fingers -> Eraser
    12: ("ZIN", "zoom_in"),      # Zoom in -> Phóng to
    13: ("ZOT", "zoom_out"),     # Zoom out -> Thu nhỏ
    9:  ("OP2", "click"),        # Open twice (Chớp mở bàn tay 2 lần) -> Left Click
}

CONTINUOUS_CLASSES = {"idle", "point_1f", "point_2f"}


def ensure_unzipped(ipn_dir: str) -> Tuple[str, str]:
    """Kiểm tra hoặc giải nén annotations và landmarks."""
    annot_dirs = glob.glob(os.path.join(ipn_dir, "**", "temporal_annotations"), recursive=True)
    lm_dirs = (
        glob.glob(os.path.join(ipn_dir, "**", "landmarks_gt"), recursive=True)
        + glob.glob(os.path.join(ipn_dir, "**", "landmarks_mp_hands"), recursive=True)
    )

    if not (annot_dirs and lm_dirs):
        for zpath in glob.glob(os.path.join(ipn_dir, "*.zip")):
            print(f"[*] Đang giải nén {os.path.basename(zpath)}...")
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(ipn_dir)
        annot_dirs = glob.glob(os.path.join(ipn_dir, "**", "temporal_annotations"), recursive=True)
        lm_dirs = (
            glob.glob(os.path.join(ipn_dir, "**", "landmarks_gt"), recursive=True)
            + glob.glob(os.path.join(ipn_dir, "**", "landmarks_mp_hands"), recursive=True)
        )

    if not (annot_dirs and lm_dirs):
        raise FileNotFoundError(f"Không tìm thấy `temporal_annotations` hoặc `landmarks_gt` trong {ipn_dir}")

    return annot_dirs[0], lm_dirs[0]


def load_official_splits(ipn_dir: str) -> Tuple[set[str], set[str]]:
    """Đọc `splits/train_list.txt` và `splits/test_list.txt`."""
    train_videos, test_videos = set(), set()
    for p in glob.glob(os.path.join(ipn_dir, "**", "train_list.txt"), recursive=True):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    train_videos.add(vid)

    for p in glob.glob(os.path.join(ipn_dir, "**", "test_list.txt"), recursive=True):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    test_videos.add(vid)

    return train_videos, test_videos


def extract_frame_21_keypoints(frame_info: dict) -> np.ndarray:
    """Trích xuất 21 khớp 3D (x, y, z) từ 1 frame trong file JSON của IPN HandS."""
    if not isinstance(frame_info, dict):
        return np.zeros(63, dtype=np.float32)

    hands = frame_info.get("hand_landmarks", [])
    if not hands:
        return np.zeros(63, dtype=np.float32)

    hand_obj = hands[0]
    lm_dict = hand_obj.get("landmarks", hand_obj)

    coords = []
    for k in range(21):
        pt = lm_dict.get(str(k), lm_dict.get(k, [0.0, 0.0, 0.0]))
        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
            coords.extend([float(pt[0]), float(pt[1]), float(pt[2])])
        elif isinstance(pt, (list, tuple)) and len(pt) == 2:
            coords.extend([float(pt[0]), float(pt[1]), 0.0])
        else:
            coords.extend([0.0, 0.0, 0.0])

    return np.array(coords, dtype=np.float32)


def _extract_sequence_slice(
    video_lms: Dict[str, dict], video_id: str, sub_start: int, sub_end: int
) -> Tuple[np.ndarray, int]:
    """Lấy mảng (T, 63) và đếm số frame thực sự có bàn tay trong đoạn [sub_start, sub_end]."""
    seq_frames = []
    valid_count = 0
    for f_idx in range(sub_start, sub_end + 1):
        key = f"{video_id}_{f_idx:06d}.jpg"
        f_info = video_lms.get(key, {})
        kpts = extract_frame_21_keypoints(f_info)
        if not np.allclose(kpts, 0.0, atol=1e-6):
            valid_count += 1
        seq_frames.append(kpts)
    if not seq_frames:
        return np.zeros((0, 63), dtype=np.float32), 0
    return np.stack(seq_frames, axis=0), valid_count


def _find_valid_continuous_windows(
    video_lms: Dict[str, dict],
    video_id: str,
    start_f: int,
    end_f: int,
    ppt_class: str = "idle",
    win_len: int = 32,
    max_windows: int = 2,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Quét phân đoạn dài (`idle`, `point_1f`, `point_2f`) để tìm các cửa sổ `win_len`
    mà bàn tay thực sự xuất hiện rõ trên khung hình (>= 75% số frame hợp lệ):
    - Với `idle` ở tập Train: Ưu tiên lấy sát mép đầu (`start_f` - lúc vừa thu tay về sau cử chỉ trước)
      và sát mép cuối (`end_f - win_len + 1` - lúc đang giơ tay lên chuẩn bị làm cử chỉ tiếp theo),
      giúp mô hình học trực tiếp các chuyển động chuyển tiếp là `idle`.
    - Với `point_1f` / `point_2f`: Lấy ở mốc 25%, 55%, 80% để bắt cả lúc giơ tĩnh và lúc đang rê tay vẽ.
    """
    dur = end_f - start_f + 1
    if dur < 20:
        return []
    actual_win = min(win_len, dur)
    if max_windows == 1 or dur < 50:
        candidates = [start_f + (dur - actual_win) // 2]
    elif ppt_class == "idle":
        # Bắt chính xác khoảnh khắc vừa thu tay về (start_f), vừa đưa tay lên (end_f - actual_win + 1), và khúc giữa
        candidates = [
            start_f,
            max(start_f, end_f - actual_win + 1),
            start_f + (dur - actual_win) // 2,
        ]
    else:
        candidates = [
            start_f + int((dur - actual_win) * 0.25),
            start_f + int((dur - actual_win) * 0.55),
            start_f + int((dur - actual_win) * 0.80),
        ]

    # Loại bỏ các vị trí trùng lặp nhưng giữ nguyên thứ tự ưu tiên
    unique_candidates: List[int] = []
    for c in candidates:
        if all(abs(c - prev) >= 10 for prev in unique_candidates):
            unique_candidates.append(c)

    results: List[Tuple[int, int, np.ndarray]] = []
    for s_idx in unique_candidates:
        e_idx = s_idx + actual_win - 1
        seq_np, valid_cnt = _extract_sequence_slice(video_lms, video_id, s_idx, e_idx)
        if len(seq_np) >= 15 and valid_cnt >= int(0.75 * len(seq_np)):
            results.append((s_idx, e_idx, seq_np))
            if len(results) >= max_windows:
                break
    return results


def slice_ipn_hands_dataset(
    ipn_dir: str = str(IPN_HANDS_DIR),
    raw_seq_dir: str = str(RAW_SEQ_DIR),
    max_continuous_per_video_train: int = 2,
    max_continuous_per_video_test: int = 1,
) -> None:
    annot_dir, lm_dir = ensure_unzipped(ipn_dir)
    _train_videos, test_videos = load_official_splits(ipn_dir)

    print("=" * 88)
    print("BƯỚC 1A: KHAI THÁC TOÀN DIỆN 8 CỬ CHỈ TỪ 200 VIDEOS BỘ DỮ LIỆU IPN HandS")
    print("=" * 88)

    os.makedirs(raw_seq_dir, exist_ok=True)
    for cname in CLASS_NAMES:
        cdir = os.path.join(raw_seq_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        # Chỉ dọn các file IPN cũ (không bao giờ đụng vào file webcam_*.npy của người dùng)
        for old_f in glob.glob(os.path.join(cdir, "*.npy")):
            if not os.path.basename(old_f).startswith("webcam_"):
                try:
                    os.remove(old_f)
                except OSError:
                    pass

    metadata_rows = []
    stats = {cname: {"train": 0, "test": 0, "total": 0, "lengths": []} for cname in CLASS_NAMES}

    annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.txt")))
    print(f"[*] Tìm thấy {len(annot_files)} file temporal annotations.")

    for af in annot_files:
        video_id = os.path.splitext(os.path.basename(af))[0]
        json_path = os.path.join(lm_dir, f"{video_id}.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as jf:
            video_lms = json.load(jf)

        is_test = video_id in test_videos
        split_name = "test" if is_test else "train"
        max_cont = max_continuous_per_video_test if is_test else max_continuous_per_video_train

        cont_extracted = {cname: 0 for cname in CONTINUOUS_CLASSES}

        with open(af, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"[\s,;]+", line.strip())
                if len(parts) < 3:
                    continue
                try:
                    ipn_cid = int(parts[0])
                    start_f = int(parts[1])
                    end_f = int(parts[2])
                except ValueError:
                    continue

                if ipn_cid not in IPN_ID_TO_PPT_CLASS:
                    continue

                ipn_code, ppt_class = IPN_ID_TO_PPT_CLASS[ipn_cid]
                dur = end_f - start_f + 1

                # 1. Xử lý nhóm cử chỉ liên tục (idle, point_1f, point_2f)
                if ppt_class in CONTINUOUS_CLASSES:
                    max_cont_cls = (
                        (2 if ppt_class == "idle" else 1)
                        if is_test
                        else (2 if ppt_class == "idle" else max_cont)
                    )
                    if cont_extracted[ppt_class] >= max_cont_cls:
                        continue
                    rem_quota = max_cont_cls - cont_extracted[ppt_class]
                    win_per_seg = 1 if is_test else min(2, rem_quota)
                    windows = _find_valid_continuous_windows(
                        video_lms,
                        video_id,
                        start_f,
                        end_f,
                        ppt_class=ppt_class,
                        win_len=32,
                        max_windows=win_per_seg,
                    )
                    for s_f, e_f, seq_np in windows:
                        out_name = f"{video_id}_{ipn_code}_f{s_f:06d}_{e_f:06d}.npy"
                        out_path = os.path.join(raw_seq_dir, ppt_class, out_name)
                        np.save(out_path, seq_np)

                        stats[ppt_class][split_name] += 1
                        stats[ppt_class]["total"] += 1
                        stats[ppt_class]["lengths"].append(len(seq_np))
                        metadata_rows.append({
                            "file": f"{ppt_class}/{out_name}",
                            "video_id": video_id,
                            "split": split_name,
                            "ipn_id": ipn_cid,
                            "ipn_code": ipn_code,
                            "ppt_class": ppt_class,
                        })
                        cont_extracted[ppt_class] += 1
                        if cont_extracted[ppt_class] >= max_cont_cls:
                            break
                    continue

                # 2. Xử lý nhóm cử chỉ rời rạc (swipe_left, swipe_right, zoom_in, zoom_out, click)
                if dur < 8 or dur > 160:
                    continue

                # Chỉ lưu 1 phân đoạn gốc chuẩn [start_f, end_f] trên đĩa (không nhân bản trùng _sL/_sR)
                # để mỗi mẫu trong epoch là độc lập; biến thiên thời gian sẽ xử lý động trong Dataset
                crops: List[Tuple[int, int, str]] = [(start_f, end_f, "")]

                for s_f, e_f, suffix in crops:
                    seq_np, valid_cnt = _extract_sequence_slice(video_lms, video_id, s_f, e_f)
                    if len(seq_np) < 8:
                        continue
                    if suffix and valid_cnt < int(0.80 * len(seq_np)):
                        continue

                    out_name = f"{video_id}_{ipn_code}_f{s_f:06d}_{e_f:06d}{suffix}.npy"
                    out_path = os.path.join(raw_seq_dir, ppt_class, out_name)
                    np.save(out_path, seq_np)

                    stats[ppt_class][split_name] += 1
                    stats[ppt_class]["total"] += 1
                    stats[ppt_class]["lengths"].append(len(seq_np))
                    metadata_rows.append({
                        "file": f"{ppt_class}/{out_name}",
                        "video_id": video_id,
                        "split": split_name,
                        "ipn_id": ipn_cid,
                        "ipn_code": ipn_code,
                        "ppt_class": ppt_class,
                    })

    meta_csv_path = os.path.join(raw_seq_dir, "metadata.csv")
    with open(meta_csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(
            cf, fieldnames=["file", "video_id", "split", "ipn_id", "ipn_code", "ppt_class"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nBẢNG THỐNG KÊ MẪU TRÍCH XUẤT TỪ 200 VIDEOS IPN HandS:")
    print("-" * 72)
    print(f"{'Lớp cử chỉ':<15} | {'Train Pool':>12} | {'Test (52 vids)':>15} | {'Tổng mẫu':>10}")
    print("-" * 72)
    for cname in CLASS_NAMES:
        st = stats[cname]
        print(f"{cname:<15} | {st['train']:>12d} | {st['test']:>15d} | {st['total']:>10d}")
    print("-" * 72)
    print(f"[HOÀN THÀNH BƯỚC 1A] Đã xuất {len(metadata_rows)} phân đoạn IPN HandS vào {raw_seq_dir}")


if __name__ == "__main__":
    slice_ipn_hands_dataset()
