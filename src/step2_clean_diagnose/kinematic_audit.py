"""
MODULE KIỂM ĐỊNH CHẤT LƯỢNG ĐỘNG HỌC (KINEMATIC QUALITY AUDIT)
==============================================================
Kiểm tra tính hợp lệ cơ sinh học của từng chuỗi (30, 63):
- Phát hiện các mẫu Webcam bị lỗi thao tác (zoom_in nhưng đã xòe tay trước, cụp ngón tay khi viết, mất dấu nhiều frame).
- Lọc bỏ các mẫu IPN HandS dị biệt/ngược hướng trong tập Train.
"""

from __future__ import annotations
import numpy as np


def audit_sequence_kinematics(
    seq_30x63: np.ndarray,
    cname: str,
    is_webcam: bool,
    num_missing: int,
    split_name: str,
) -> tuple[bool, str, dict]:
    """Kiểm tra chất lượng động học của chuỗi cử chỉ 30 frames."""
    pts = seq_30x63.reshape(30, 21, 3)
    wrist = pts[:, 0, :]
    palm = np.linalg.norm(pts[:, 9, :] - wrist, axis=1) + 1e-6
    ap = np.linalg.norm(pts[:, 8, :] - pts[:, 4, :], axis=1) / palm
    ap_travel = float(np.sum(np.abs(np.diff(ap))))

    ap_start = float(np.mean(ap[:5]))
    ap_end = float(np.mean(ap[-5:]))
    ap_delta = ap_end - ap_start
    net_dx = float(wrist[-1, 0] - wrist[0, 0])
    net_dy = float(wrist[-1, 1] - wrist[0, 1])

    metrics = {
        "net_dx": round(net_dx, 3),
        "net_dy": round(net_dy, 3),
        "ap_start": round(ap_start, 2),
        "ap_end": round(ap_end, 2),
        "ap_delta": round(ap_delta, 2),
        "ap_travel": round(ap_travel, 2),
        "missing": int(num_missing),
    }

    if is_webcam:
        if num_missing > 7:
            return False, f"Mất dấu bàn tay {num_missing}/30 frames (>7 frames)", metrics
        idx_ext_mean = float(np.mean(np.linalg.norm(pts[:, 8, :] - wrist, axis=1) / palm))
        if cname == "zoom_in":
            if ap_delta < 0.14:
                return (
                    False,
                    f"Độ bung ngón quá nhỏ (ap: {ap_start:.2f} -> {ap_end:.2f}, delta={ap_delta:+.2f} < +0.14)",
                    metrics,
                )
        elif cname == "zoom_out":
            if ap_delta > -0.14:
                return (
                    False,
                    f"Độ khép ngón quá nhỏ (ap: {ap_start:.2f} -> {ap_end:.2f}, delta={ap_delta:+.2f} > -0.14)",
                    metrics,
                )
        elif cname == "point_1f":
            if idx_ext_mean < 0.90:
                return (
                    False,
                    f"Ngón trỏ chưa duỗi rõ (idx_ext_mean={idx_ext_mean:.2f} < 0.90)",
                    metrics,
                )
        elif cname == "point_2f":
            if idx_ext_mean < 0.90:
                return (
                    False,
                    f"Hai ngón chưa duỗi rõ (idx_ext_mean={idx_ext_mean:.2f} < 0.90)",
                    metrics,
                )
        elif cname == "swipe_left":
            if net_dx < 0.08:
                return (
                    False,
                    f"Biên độ vuốt trái quá ngắn hoặc ngược hướng (net_dx={net_dx:+.3f} < +0.08)",
                    metrics,
                )
        elif cname == "swipe_right":
            if net_dx > -0.08:
                return (
                    False,
                    f"Biên độ vuốt phải quá ngắn hoặc ngược hướng (net_dx={net_dx:+.3f} > -0.08)",
                    metrics,
                )
        elif cname == "click":
            if abs(net_dx) > 0.25:
                return (
                    False,
                    f"Cổ tay dịch ngang quá mạnh dễ nhầm với swipe (net_dx={net_dx:+.3f})",
                    metrics,
                )
            if ap_travel < 0.25:
                return (
                    False,
                    f"Chưa chớp mở bàn tay đủ 2 nhịp (ap_travel={ap_travel:.2f} < 0.25)",
                    metrics,
                )
        return True, "Đạt chuẩn động học Webcam", metrics

    # Với mẫu IPN HandS thuộc tập Train
    if split_name == "train":
        if cname == "swipe_left" and net_dx < 0.08:
            return False, f"IPN swipe_left biên độ yếu/ngược ({net_dx:+.3f})", metrics
        if cname == "swipe_right" and net_dx > -0.08:
            return False, f"IPN swipe_right biên độ yếu/ngược ({net_dx:+.3f})", metrics
        if cname == "zoom_in" and ap_delta < 0.08:
            return False, f"IPN zoom_in không bung ngón ({ap_delta:+.2f})", metrics
        if cname == "zoom_out" and ap_delta > -0.08:
            return False, f"IPN zoom_out không khép ngón ({ap_delta:+.2f})", metrics
        if cname in ("point_1f", "point_2f") and abs(net_dx) > 0.35:
            return False, f"IPN {cname} dịch ngang quá lớn dễ nhầm swipe ({net_dx:+.3f})", metrics
        if cname == "click" and abs(net_dx) > 0.24:
            return False, f"IPN {cname} dịch ngang quá lớn dễ nhầm swipe ({net_dx:+.3f})", metrics

    return True, "OK", metrics

