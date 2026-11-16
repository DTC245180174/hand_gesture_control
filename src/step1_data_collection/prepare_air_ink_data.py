"""
BƯỚC 1 & 2 (AIR-INK): CHUẨN BỊ DỮ LIỆU 9 LỚP HÌNH HỌC CHO MODEL 3 (AIRINKRESCNN)
===================================================================================
Tạo bộ tensor uint8 chuẩn 64x64 vào `data/air_ink/processed/` cho **9 lớp hình học**:
`line`, `circle`, `rectangle`, `triangle`, `arrow`, `checkmark`, `cross`, `star`, `freeform`.

1. Sinh quỹ đạo giả lập đa biến thể (nét run tay, đa độ dày 1..3px, xoay nhẹ)
   bảo đảm đồng nhất 100% miền nét vẽ không trung.
2. Gộp thêm mẫu Webcam thật từ `data/air_ink/raw_webcam/<shape>/*.npz` (nếu người dùng đã thu
   qua `--step 1c`), tách riêng 20% phiên thu làm Holdout độc lập.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from common.config import (
    AIR_INK_CLASSES,
    AIR_INK_IMG_SIZE,
    AIR_INK_NAME_TO_ID,
    AIR_INK_PROCESSED_DIR,
    AIR_INK_RAW_DIR,
    AIR_INK_SHAPE_CLASSES,
    DIAGNOSTICS_DIR,
    NUM_AIR_INK_CLASSES,
)
from step2_clean_diagnose.ink_rasterizer import (
    generate_synthetic_air_strokes,
    rasterize_strokes_to_64x64,
)


def _generate_synthetic_split(
    classes: Sequence[str],
    per_class: int,
    seed: int,
    thickness_choices: Tuple[int, ...] = (1, 2, 2, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for cname in classes:
        cid = AIR_INK_NAME_TO_ID[cname]
        for _ in range(per_class):
            strokes = generate_synthetic_air_strokes(cname, rng)
            thick = int(rng.choice(thickness_choices))
            pad = float(rng.uniform(0.12, 0.22))
            img = rasterize_strokes_to_64x64(
                strokes,
                size=AIR_INK_IMG_SIZE,
                thickness=thick,
                pad_ratio=pad,
                blur_sigma=0.40,
            )
            xs.append(img)
            ys.append(cid)
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64)


def _load_webcam_air_ink(
    raw_dir: Path = AIR_INK_RAW_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tải mẫu Air-Ink thật từ `data/air_ink/raw_webcam/` nếu người dùng đã thu bằng Step 1C."""
    tr_x: List[np.ndarray] = []
    tr_y: List[int] = []
    ho_x: List[np.ndarray] = []
    ho_y: List[int] = []

    if not raw_dir.exists():
        empty_x = np.zeros((0, AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE), dtype=np.uint8)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_x, empty_y, empty_x, empty_y

    for cname in AIR_INK_CLASSES:
        cdir = raw_dir / cname
        if not cdir.exists():
            continue
        files = sorted(cdir.glob("*.npz"))
        if not files:
            continue
        cid = AIR_INK_NAME_TO_ID[cname]
        sessions: Dict[str, List[np.ndarray]] = {}
        for fpath in files:
            try:
                data = np.load(fpath, allow_pickle=True)
                img64 = np.asarray(data["img64"], dtype=np.uint8)
                sess = str(data["session_id"].item()) if "session_id" in data else fpath.stem[:15]
                if img64.shape == (AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE):
                    sessions.setdefault(sess, []).append(img64)
            except Exception:
                continue

        sess_keys = sorted(sessions.keys())
        n_hold = max(1, int(round(len(sess_keys) * 0.2))) if len(sess_keys) >= 2 else 0
        hold_set = set(sess_keys[-n_hold:]) if n_hold > 0 else set()
        for sk in sess_keys:
            for im in sessions[sk]:
                if sk in hold_set:
                    ho_x.append(im)
                    ho_y.append(cid)
                else:
                    tr_x.append(im)
                    tr_y.append(cid)

    def _pack(lx: List[np.ndarray], ly: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if not lx:
            return np.zeros((0, AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
        return np.stack(lx, axis=0), np.array(ly, dtype=np.int64)

    return _pack(tr_x, tr_y) + _pack(ho_x, ho_y)


def prepare_air_ink_datasets(
    synth_train: int = 1200,
    synth_val: int = 200,
    synth_test: int = 280,
) -> Dict[str, object]:
    """
    Xây dựng toàn bộ tập dữ liệu 9 lớp Hình học cho Model 3 (AirInkResCNN),
    lưu thành tensor uint8 (N, 64, 64) vào `data/air_ink/processed/`.
    """
    AIR_INK_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("BƯỚC 1 & 2 (AIR-INK): XÂY DỰNG DỮ LIỆU 9 LỚP HÌNH HỌC 64x64")
    print("=" * 84)

    classes = tuple(AIR_INK_SHAPE_CLASSES)
    print(f"[1/2] Đang sinh quỹ đạo giả lập {len(classes)} lớp Hình học (ngón trỏ vẽ không trung)...")
    x_s_tr, y_s_tr = _generate_synthetic_split(classes, synth_train, seed=301, thickness_choices=(1, 2, 2, 3))
    x_s_va, y_s_va = _generate_synthetic_split(classes, synth_val, seed=302, thickness_choices=(2,))
    x_s_te, y_s_te = _generate_synthetic_split(classes, synth_test, seed=303, thickness_choices=(2,))
    print(f"  -> Synthetic 64x64: Train={len(y_s_tr)} | Val={len(y_s_va)} | Test={len(y_s_te)}")

    x_wc_tr, y_wc_tr, x_wc_ho, y_wc_ho = _load_webcam_air_ink()
    print(f"[2/2] Dữ liệu Webcam thực tế (`raw_webcam/`): Train={len(y_wc_tr)} | Holdout={len(y_wc_ho)}")

    X_train = np.concatenate([x_s_tr, x_wc_tr], axis=0) if len(y_wc_tr) > 0 else x_s_tr
    y_train = np.concatenate([y_s_tr, y_wc_tr], axis=0) if len(y_wc_tr) > 0 else y_s_tr
    X_val = x_s_va
    y_val = y_s_va
    X_test = x_s_te
    y_test = y_s_te

    shuf = np.random.default_rng(777).permutation(len(y_train))
    X_train = X_train[shuf]
    y_train = y_train[shuf]

    np.save(AIR_INK_PROCESSED_DIR / "X_train.npy", X_train)
    np.save(AIR_INK_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(AIR_INK_PROCESSED_DIR / "X_val.npy", X_val)
    np.save(AIR_INK_PROCESSED_DIR / "y_val.npy", y_val)
    np.save(AIR_INK_PROCESSED_DIR / "X_test.npy", X_test)
    np.save(AIR_INK_PROCESSED_DIR / "y_test.npy", y_test)

    wc_x_file = AIR_INK_PROCESSED_DIR / "X_webcam_holdout.npy"
    wc_y_file = AIR_INK_PROCESSED_DIR / "y_webcam_holdout.npy"
    if len(y_wc_ho) > 0:
        np.save(wc_x_file, x_wc_ho)
        np.save(wc_y_file, y_wc_ho)
    else:
        for stale in (wc_x_file, wc_y_file):
            if stale.exists():
                stale.unlink()

    fg_tr = float((X_train > 40).mean() * 100.0)
    fg_te = float((X_test > 40).mean() * 100.0)

    counts_tr = np.bincount(y_train, minlength=NUM_AIR_INK_CLASSES).tolist()
    counts_va = np.bincount(y_val, minlength=NUM_AIR_INK_CLASSES).tolist()
    counts_te = np.bincount(y_test, minlength=NUM_AIR_INK_CLASSES).tolist()

    report = {
        "image_size": AIR_INK_IMG_SIZE,
        "num_classes": NUM_AIR_INK_CLASSES,
        "class_names": AIR_INK_CLASSES,
        "splits": {
            "train": list(X_train.shape),
            "val": list(X_val.shape),
            "test": list(X_test.shape),
            "webcam_holdout": list(x_wc_ho.shape),
        },
        "foreground_density_pct": {
            "train_strokes": round(fg_tr, 2),
            "test_strokes": round(fg_te, 2),
        },
        "train_counts_min_max": [int(min(counts_tr)), int(max(counts_tr))],
        "val_counts_min_max": [int(min(counts_va)), int(max(counts_va))],
        "test_counts_min_max": [int(min(counts_te)), int(max(counts_te))],
    }
    report_path = DIAGNOSTICS_DIR / "air_ink_data_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("-" * 84)
    print(f"[HOÀN TẤT DỮ LIỆU AIR-INK {NUM_AIR_INK_CLASSES} LỚP HÌNH HỌC]")
    print(f"  - X_train : {X_train.shape} ({X_train.nbytes / (1024*1024):.1f} MB)")
    print(f"  - X_val   : {X_val.shape} ({X_val.nbytes / (1024*1024):.1f} MB)")
    print(f"  - X_test  : {X_test.shape} ({X_test.nbytes / (1024*1024):.1f} MB)")
    print(f"  - Mật độ nét trắng (>40): Train={fg_tr:.2f}% | Test={fg_te:.2f}%")
    print(f"  - Báo cáo: {report_path}")
    return report


if __name__ == "__main__":
    prepare_air_ink_datasets()
