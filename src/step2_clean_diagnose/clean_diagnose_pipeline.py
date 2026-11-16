"""
BƯỚC 2: QUY TRÌNH LÀM SẠCH, ĐỒNG BỘ ĐỘ DÀI T=30 & ĐÓNG GÓI TENSOR
==================================================================
Tự động nạp toàn bộ các chuỗi thô từ `data/raw_sequences/` (IPN HandS + Webcam cá nhân):
1. Nội suy frame khuyết
2. Resample về T=30 frames
3. Kiểm định động học
4. Trích xuất đặc trưng động 66D
5. Chia train/val/test theo video_id chống rò rỉ dữ liệu
6. Cân bằng mẫu Webcam và lưu các file `.npy` vào `data/processed/`.
"""

from __future__ import annotations
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from common.config import (
    CLASS_NAME_TO_ID,
    DIAGNOSTICS_DIR,
    PROCESSED_DIR,
    RAW_SEQ_DIR,
    SEQ_LEN,
)
from step2_clean_diagnose.feature_extractor import (
    extract_66d_dynamic_features,
    interpolate_missing_frames,
    temporal_resample,
)
from step2_clean_diagnose.kinematic_audit import audit_sequence_kinematics

RANDOM_SEED = 42


def run_clean_and_diagnose_pipeline(
    raw_seq_dir: str = str(RAW_SEQ_DIR),
    processed_dir: str = str(PROCESSED_DIR),
    diag_dir: str = str(DIAGNOSTICS_DIR),
) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    meta_path = os.path.join(raw_seq_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Không tìm thấy {meta_path}. Hãy chạy `prepare_ipn_hands.py` trước!"
        )

    print("=" * 88)
    print("BƯỚC 2: LÀM SẠCH, ĐỒNG BỘ ĐỘ DÀI T=30 & TRÍCH XUẤT 66 ĐẶC TRƯNG ĐỘNG")
    print("=" * 88)

    with open(meta_path, "r", encoding="utf-8") as cf:
        rows = list(csv.DictReader(cf))

    # Bổ sung các file webcam_*.npy vào danh sách xử lý
    recorded_files = {r["file"] for r in rows}
    webcam_added_count = 0
    for cname in CLASS_NAME_TO_ID:
        cdir = os.path.join(raw_seq_dir, cname)
        if not os.path.exists(cdir):
            continue
        for fpath in sorted(glob.glob(os.path.join(cdir, "webcam_*.npy"))):
            rel_f = f"{cname}/{os.path.basename(fpath)}"
            if rel_f not in recorded_files:
                rows.append({
                    "file": rel_f,
                    "video_id": "webcam_user",
                    "split": "train",
                    "ipn_id": -1,
                    "ipn_code": "WEB",
                    "ppt_class": cname,
                })
                webcam_added_count += 1

    # Tách danh sách video thuộc train theo NGƯỜI QUAY (Subject ID = phần trước '_#')
    # Đảm bảo 4 video của cùng 1 người không bao giờ vừa nằm ở Train vừa nằm ở Val!
    train_pool_videos = sorted(
        {r["video_id"] for r in rows if r["split"] == "train" and r["video_id"] != "webcam_user"}
    )
    subject_to_videos: dict[str, list[str]] = {}
    for vid in train_pool_videos:
        subj_id = vid.split("_#")[0]
        subject_to_videos.setdefault(subj_id, []).append(vid)

    all_subjects = sorted(subject_to_videos.keys())
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(all_subjects)

    num_val_subjects = max(1, int(round(len(all_subjects) * 0.16)))  # 6/37 người (~24 videos)
    val_subjects = set(all_subjects[:num_val_subjects])

    val_videos_set = {vid for subj in val_subjects for vid in subject_to_videos[subj]}
    train_videos_set = (set(train_pool_videos) - val_videos_set) | {"webcam_user"}

    splits_data = {
        "train": {"X": [], "y": [], "raw30": [], "videos": []},
        "val":   {"X": [], "y": [], "raw30": [], "videos": []},
        "test":  {"X": [], "y": [], "raw30": [], "videos": []},
    }
    clean_webcam_pool = {cname: [] for cname in CLASS_NAME_TO_ID}
    webcam_audit_logs = []

    class_diagnostics = {
        cname: {
            "raw_total": 0,
            "dropped_high_missing": 0,
            "interpolated_sequences": 0,
            "clean_kept": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "webcam_raw": 0,
            "webcam_clean": 0,
            "webcam_rejected": 0,
            "raw_len_min": 9999,
            "raw_len_max": 0,
            "raw_lengths": [],
            "net_dx_mean": [],
            "net_dy_mean": [],
        }
        for cname in CLASS_NAME_TO_ID
    }
    dropped_details = []
    val_video_class_count: dict[tuple[str, str], int] = {}

    for r in rows:
        cname = r["ppt_class"]
        if cname not in CLASS_NAME_TO_ID:
            continue
        cid = CLASS_NAME_TO_ID[cname]
        fpath = os.path.join(raw_seq_dir, r["file"])
        if not os.path.exists(fpath):
            continue

        seq_raw = np.load(fpath).astype(np.float32)
        t_raw = seq_raw.shape[0]
        vid = r["video_id"]
        is_webcam = vid == "webcam_user"

        cdiag = class_diagnostics[cname]
        cdiag["raw_total"] += 1
        if is_webcam:
            cdiag["webcam_raw"] += 1
        cdiag["raw_len_min"] = min(cdiag["raw_len_min"], int(t_raw))
        cdiag["raw_len_max"] = max(cdiag["raw_len_max"], int(t_raw))
        cdiag["raw_lengths"].append(int(t_raw))

        seq_clean, num_missing, missing_ratio, is_valid = interpolate_missing_frames(seq_raw)
        if not is_valid:
            cdiag["dropped_high_missing"] += 1
            if is_webcam:
                cdiag["webcam_rejected"] += 1
                webcam_audit_logs.append({
                    "file": r["file"],
                    "class": cname,
                    "status": "LOẠI (MISSING)",
                    "reason": f"Mất dấu bàn tay {num_missing}/{t_raw} frames ({missing_ratio*100:.1f}% > 25%)",
                })
            dropped_details.append({
                "file": r["file"],
                "class": cname,
                "num_frames": int(t_raw),
                "missing_frames": int(num_missing),
                "missing_ratio": round(float(missing_ratio), 4),
            })
            continue

        seq_30x63 = temporal_resample(seq_clean, target_len=SEQ_LEN)
        target_split = "test" if r["split"] == "test" else ("val" if vid in val_videos_set else "train")
        # Tập Validation cân bằng 1:1 như tập Test: Không đưa lát trượt _sL/_sR và chỉ lấy tối đa 1 mẫu/video/lớp
        if target_split == "val":
            if r["file"].endswith("_sL.npy") or r["file"].endswith("_sR.npy"):
                continue
            max_val_per_vid = 2 if cname == "idle" else 1
            if val_video_class_count.get((vid, cname), 0) >= max_val_per_vid:
                continue

        is_kin_ok, kin_reason, _kin_metrics = audit_sequence_kinematics(
            seq_30x63, cname, is_webcam=is_webcam, num_missing=num_missing, split_name=target_split
        )
        if not is_kin_ok:
            cdiag["dropped_high_missing"] += 1
            if is_webcam:
                cdiag["webcam_rejected"] += 1
                webcam_audit_logs.append({
                    "file": r["file"],
                    "class": cname,
                    "status": "LOẠI (LỖI THAO TÁC)",
                    "reason": kin_reason,
                })
            dropped_details.append({
                "file": r["file"],
                "class": cname,
                "num_frames": int(t_raw),
                "missing_frames": int(num_missing),
                "missing_ratio": round(float(missing_ratio), 4),
                "kinematic_reject_reason": kin_reason,
            })
            continue

        if num_missing > 0:
            cdiag["interpolated_sequences"] += 1
        cdiag["clean_kept"] += 1

        net_dx = float(seq_30x63[-1, 0] - seq_30x63[0, 0])
        net_dy = float(seq_30x63[-1, 1] - seq_30x63[0, 1])
        cdiag["net_dx_mean"].append(net_dx)
        cdiag["net_dy_mean"].append(net_dy)

        feat_30x66 = extract_66d_dynamic_features(seq_30x63)

        if is_webcam:
            cdiag["webcam_clean"] += 1
            clean_webcam_pool[cname].append((feat_30x66, cid, seq_30x63, vid, r["file"]))
        else:
            if target_split == "test":
                cdiag["test_count"] += 1
            elif target_split == "val":
                cdiag["val_count"] += 1
                val_video_class_count[(vid, cname)] = val_video_class_count.get((vid, cname), 0) + 1
            else:
                cdiag["train_count"] += 1

            splits_data[target_split]["X"].append(feat_30x66)
            splits_data[target_split]["y"].append(cid)
            splits_data[target_split]["raw30"].append(seq_30x63)
            splits_data[target_split]["videos"].append(vid)

    # Tách mẫu Webcam đạt chuẩn thành 80% Train / 20% Val độc lập (Chống học vẹt & chống rò rỉ dữ liệu)
    webcam_val_X, webcam_val_y, webcam_val_raw = [], [], []
    if webcam_added_count > 0:
        rng_web = random.Random(42)
        print("\n" + "=" * 96)
        print("BÁO CÁO KIỂM ĐỊNH & PHÂN CHIA MẪU WEBCAM CÁ NHÂN (80% TRAIN / 20% VAL ĐỘC LẬP)")
        print("=" * 96)
        for cname in CLASS_NAME_TO_ID:
            cdiag = class_diagnostics[cname]
            clean_list = list(clean_webcam_pool[cname])
            n_clean = len(clean_list)
            if n_clean == 0:
                print(
                    f"{cname:<13} | Thu: {cdiag['webcam_raw']:<3} | Chuan: 0   | "
                    f"Loai: {cdiag['webcam_rejected']:<2} | Train: 0   | Val (Doc lap): 0"
                )
                continue

            rng_web.shuffle(clean_list)
            n_val = max(1, int(round(0.15 * n_clean))) if n_clean >= 6 else 0
            val_part = clean_list[:n_val]
            train_part = clean_list[n_val:]

            # Giảm hệ số nhân lặp (rep) xuống tối đa x2 chỉ cho lớp quá ít mẫu (<25), còn lại rep=1
            # để giảm overfit học vẹt trên tập Train, kết hợp tăng cường dữ liệu (mirror + 3D tilt) lúc train
            rep = (
                1
                if (cname == "idle" or len(train_part) >= 25)
                else 2
            )
            added_train = len(train_part) * rep
            cdiag["train_count"] += added_train

            for feat_30x66, cid, seq_30x63, vid, _ in train_part:
                for _ in range(rep):
                    splits_data["train"]["X"].append(feat_30x66)
                    splits_data["train"]["y"].append(cid)
                    splits_data["train"]["raw30"].append(seq_30x63)
                    splits_data["train"]["videos"].append(vid)

            # Lưu riêng 20% mẫu Webcam độc lập ra X_webcam_clean.npy & raw30_webcam_clean.npy, tuyệt đối KHÔNG trộn vào X_val.npy
            for feat_30x66, cid, seq_30x63, vid, _ in val_part:
                webcam_val_X.append(feat_30x66)
                webcam_val_y.append(cid)
                webcam_val_raw.append(seq_30x63)

            print(
                f"{cname:<13} | Thu: {cdiag['webcam_raw']:<3} | Chuan: {n_clean:<3} | "
                f"Loai: {cdiag['webcam_rejected']:<2} | Nhan: x{rep:<1} | "
                f"Train: {added_train:<3} | Holdout Webcam: {len(val_part)} mau"
            )

    if webcam_val_X:
        np.save(os.path.join(processed_dir, "X_webcam_clean.npy"), np.array(webcam_val_X, dtype=np.float32))
        np.save(os.path.join(processed_dir, "y_webcam_clean.npy"), np.array(webcam_val_y, dtype=np.int64))
        np.save(os.path.join(processed_dir, "raw30_webcam_clean.npy"), np.array(webcam_val_raw, dtype=np.float32))

    for sp_name, sp_dict in splits_data.items():
        X_arr = np.array(sp_dict["X"], dtype=np.float32)
        y_arr = np.array(sp_dict["y"], dtype=np.int64)
        raw30_arr = np.array(sp_dict["raw30"], dtype=np.float32)

        np.save(os.path.join(processed_dir, f"X_{sp_name}.npy"), X_arr)
        np.save(os.path.join(processed_dir, f"y_{sp_name}.npy"), y_arr)
        np.save(os.path.join(processed_dir, f"raw30_{sp_name}.npy"), raw30_arr)

    serialized_diagnostics = {}
    for cname, cdiag in class_diagnostics.items():
        raw_lens = cdiag.pop("raw_lengths", [])
        dx_list = cdiag.pop("net_dx_mean", [])
        dy_list = cdiag.pop("net_dy_mean", [])
        cdiag["raw_len_avg"] = round(float(np.mean(raw_lens)), 1) if raw_lens else 0.0
        cdiag["mean_wrist_net_dx"] = round(float(np.mean(dx_list)), 3) if dx_list else 0.0
        cdiag["mean_wrist_net_dy"] = round(float(np.mean(dy_list)), 3) if dy_list else 0.0
        serialized_diagnostics[cname] = cdiag

    report_summary = {
        "target_frames": SEQ_LEN,
        "feature_dim": 66,
        "max_missing_ratio_threshold": 0.25,
        "video_splits": {
            "train_videos": len(train_videos_set),
            "val_videos": len(val_videos_set),
            "test_videos": len({r["video_id"] for r in rows if r["split"] == "test"}),
        },
        "tensor_shapes": {
            sp: {
                "X": list(np.load(os.path.join(processed_dir, f"X_{sp}.npy")).shape),
                "y": list(np.load(os.path.join(processed_dir, f"y_{sp}.npy")).shape),
            }
            for sp in ("train", "val", "test")
        },
        "class_diagnostics": serialized_diagnostics,
        "webcam_audit_logs": webcam_audit_logs,
        "dropped_sequences": dropped_details,
    }

    with open(os.path.join(diag_dir, "data_report.json"), "w", encoding="utf-8") as rf:
        json.dump(report_summary, rf, indent=2, ensure_ascii=False)

    print("\n[HOÀN THÀNH BƯỚC 2] Đã lưu các Tensor hoàn chỉnh vào:", processed_dir)


if __name__ == "__main__":
    run_clean_and_diagnose_pipeline()

