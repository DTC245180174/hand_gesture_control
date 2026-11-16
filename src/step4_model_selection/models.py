"""
BƯỚC 4: THIẾT KẾ & XÂY DỰNG CÁC KIẾN TRÚC MẠNG HỌC SÂU CHUỖI THỜI GIAN
========================================================================
1. TemporalCNN1D: Tích chập 1D dọc theo trục thời gian + Global Pooling.
2. BiLSTMClassifier: Mạng LSTM hai chiều 2 tầng.
3. HybridConvBiGRUAttention: Dual-Stream Dilated Conv1D + BiGRU + Temporal Attention.
   (Kiến trúc khuyến nghị chính thức cho Step 5, 6, 7).
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import (
    AIR_INK_IMG_SIZE,
    DIAGNOSTICS_DIR,
    FEATURE_DIM,
    NUM_AIR_INK_CLASSES,
    NUM_CLASSES,
    NUM_POSE_CLASSES,
    POSE_FEATURE_DIM,
    SEQ_LEN,
)


class TemporalCNN1D(nn.Module):
    """Kiến trúc Temporal 1D-CNN gọn nhẹ (~62k tham số)."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout1d(dropout * 0.4),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        avg_pool = h.mean(dim=-1)
        max_pool = h.amax(dim=-1)
        feat = torch.cat([avg_pool, max_pool], dim=-1)
        return self.classifier(feat)


class BiLSTMClassifier(nn.Module):
    """Kiến trúc BiLSTM gọn nhẹ 1 tầng (~75k tham số)."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 56,
        num_layers: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        out, _ = self.bilstm(h)
        mean_pool = out.mean(dim=1)
        last_step = out[:, -1, :]
        feat = torch.cat([mean_pool, last_step], dim=-1)
        return self.classifier(feat)


class TemporalAttentionPool(nn.Module):
    """Tầng Gộp Chú ý Thời gian (Temporal Attention Pooling)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn_net(h).squeeze(-1)
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha


class HybridConvBiGRUAttention(nn.Module):
    """
    Kiến trúc Học sâu Chuỗi thời gian Tối ưu Chống Overfitting (~68,000 tham số):
    1. Pose Stream (64D): Multi-Scale Dilated Conv1D (k=3, k=5, dilated=2) + Squeeze-and-Excitation + Channel Dropout1d.
    2. Dedicated Kinematic Stream (32D): Tách riêng 8 tín hiệu vận tốc và độ mở kẹp ngón tay.
    3. 1-Layer BiGRU (96D) + Tri-Modal Head (288D -> 64D -> 8 lớp): Attention Context + MaxPool + Temporal Delta.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        conv_dim: int = 64,
        kin_dim: int = 32,
        gru_hidden: int = 48,
        num_gru_layers: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.pose_norm = nn.LayerNorm(input_dim)
        self.kin_norm = nn.LayerNorm(8)

        b1_dim = conv_dim // 2
        b2_dim = conv_dim // 4
        b3_dim = conv_dim - b1_dim - b2_dim

        self.conv_k3 = nn.Sequential(
            nn.Conv1d(input_dim, b1_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(b1_dim),
            nn.GELU(),
        )
        self.conv_k5 = nn.Sequential(
            nn.Conv1d(input_dim, b2_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(b2_dim),
            nn.GELU(),
        )
        self.conv_dilated = nn.Sequential(
            nn.Conv1d(input_dim, b3_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(b3_dim),
            nn.GELU(),
        )
        self.res_proj = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(conv_dim),
        )
        self.se_gate = nn.Sequential(
            nn.Linear(conv_dim, conv_dim // 4),
            nn.GELU(),
            nn.Linear(conv_dim // 4, conv_dim),
            nn.Sigmoid(),
        )
        self.channel_drop = nn.Dropout1d(dropout * 0.40)

        self.kin_branch = nn.Sequential(
            nn.Conv1d(8, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.GELU(),
            nn.Conv1d(24, kin_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(kin_dim),
            nn.GELU(),
        )

        fusion_dim = conv_dim + kin_dim
        self.stem_dropout = nn.Dropout(dropout * 0.5)

        self.bigru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * 2
        self.attention = TemporalAttentionPool(gru_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T=30, F=66)
        idx_tip_xy = x[:, :, 24:26]
        idx_tip_vel = torch.zeros_like(idx_tip_xy)
        idx_tip_vel[:, 1:, :] = (idx_tip_xy[:, 1:, :] - idx_tip_xy[:, :-1, :]) * 5.0

        kin_in = torch.cat([x[:, :, 0:3], x[:, :, 63:66], idx_tip_vel], dim=-1)  # (B, 30, 8)
        kin_normed = self.kin_norm(kin_in).transpose(1, 2)                       # (B, 8, 30)
        kin_feat = self.kin_branch(kin_normed).transpose(1, 2)                   # (B, 30, 32)

        # 2. Luồng 1: Multi-Scale Dilated Pose Stream + Channel Dropout1d
        pose_normed = self.pose_norm(x).transpose(1, 2)  # (B, 66, 30)
        c3 = self.conv_k3(pose_normed)                   # (B, 32, 30)
        c5 = self.conv_k5(pose_normed)                   # (B, 16, 30)
        cd = self.conv_dilated(pose_normed)              # (B, 16, 30)
        res = self.res_proj(pose_normed)                 # (B, 64, 30)

        pose_stem = F.gelu(torch.cat([c3, c5, cd], dim=1) + res)         # (B, 64, 30)
        se_weights = self.se_gate(pose_stem.mean(dim=-1)).unsqueeze(-1)  # (B, 64, 1)
        pose_feat = self.channel_drop(pose_stem * se_weights).transpose(1, 2)  # (B, 30, 64)

        # 3. Hợp nhất 2 Luồng + BiGRU 1 tầng
        fused = self.stem_dropout(torch.cat([pose_feat, kin_feat], dim=-1))  # (B, 30, 96)
        gru_out, _ = self.bigru(fused)                                       # (B, 30, 96)

        # 4. Tri-Modal Aggregation: Attention + MaxPool + Temporal Delta (Cuối - Đầu)
        attn_context, attn_weights = self.attention(gru_out)
        max_pool = gru_out.amax(dim=1)                                       # (B, 96)
        early_state = gru_out[:, :8, :].mean(dim=1)                          # (B, 96)
        late_state = gru_out[:, -8:, :].mean(dim=1)                          # (B, 96)
        delta_pool = late_state - early_state                                # (B, 96)

        combined = torch.cat([attn_context, max_pool, delta_pool], dim=-1)   # (B, 288)
        logits = self.classifier(combined)                                   # (B, 8)

        if return_attn:
            return logits, attn_weights
        return logits


class StaticPoseNet(nn.Module):
    """
    Model 1 (Option B): Mạng nơ-ron hình dáng tĩnh 1 frame huấn luyện từ đầu (~45k tham số).
    Nhận đặc trưng hình học bất biến xoay 2D trên 21 khớp (B, 6, 21) và phân loại:
      - 0: other_pose (idle, swipe, zoom, click)
      - 1: point_1f   (giơ 1 ngón trỏ)
      - 2: point_2f   (giơ 2 ngón Trỏ + Giữa)
    """

    def __init__(
        self,
        in_channels: int = POSE_FEATURE_DIM,
        num_classes: int = NUM_POSE_CLASSES,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.joint_conv = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels * 21, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 + 128, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=6, J=21)
        conv_out = self.joint_conv(x)
        conv_feat = torch.cat([conv_out.mean(dim=-1), conv_out.amax(dim=-1)], dim=-1)
        mlp_feat = self.global_mlp(x.flatten(start_dim=1))
        return self.classifier(torch.cat([conv_feat, mlp_feat], dim=-1))


class _SEBlock2d(nn.Module):
    """Squeeze-and-Excitation nhẹ cho đặc trưng không gian 2D."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hid),
            nn.GELU(),
            nn.Linear(hid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _ResSEBlock2d(nn.Module):
    """Khối Residual Conv2D + GroupNorm + Squeeze-and-Excitation (bất biến batch-size)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.se = _SEBlock2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.se(self.gn2(self.conv2(out)))
        return F.gelu(out + self.shortcut(x))


class AirInkResCNN(nn.Module):
    """
    Model 3: Mạng CNN Residual + Quadrant Spatial Pooling tự huấn luyện từ đầu
    nhận ảnh nét vẽ không gian (B, 1, 64, 64) và phân loại 9 lớp hình học vẽ không trung
    (`line`, `circle`, `rectangle`, `triangle`, `arrow`, `checkmark`, `cross`, `star`, `freeform`).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_AIR_INK_CLASSES,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 24),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
        )
        self.stage1 = _ResSEBlock2d(24, 48, stride=2)    # 32x32 -> 16x16
        self.stage2 = _ResSEBlock2d(48, 80, stride=2)    # 16x16 -> 8x8
        self.stage3 = _ResSEBlock2d(80, 128, stride=2)   # 8x8 -> 4x4

        # Bảo toàn vị trí góc phần tư (2x2 = 4 ô) để phân biệt hướng mũi tên, dấu tích, góc tam giác
        self.quad_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        feat_dim = 128 * 4 + 128  # 640 chiều

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        q = self.quad_pool(x).flatten(start_dim=1)
        m = self.max_pool(x).flatten(start_dim=1)
        return self.classifier(torch.cat([q, m], dim=-1))


def build_air_ink_model(
    in_channels: int = 1,
    num_classes: int = NUM_AIR_INK_CLASSES,
    dropout: float = 0.25,
) -> nn.Module:
    return AirInkResCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


def build_pose_model(
    in_channels: int = POSE_FEATURE_DIM,
    num_classes: int = NUM_POSE_CLASSES,
    dropout: float = 0.20,
) -> nn.Module:
    return StaticPoseNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


def build_model(
    arch_name: str = "hybrid",
    input_dim: int = FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
) -> nn.Module:
    arch = arch_name.lower().strip()
    if arch in ("hybrid", "hybridconvbigruattention"):
        return HybridConvBiGRUAttention(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif arch in ("cnn1d", "temporalcnn1d"):
        return TemporalCNN1D(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    elif arch in ("bilstm", "bilstmclassifier"):
        return BiLSTMClassifier(input_dim=input_dim, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Kiến trúc không hợp lệ: '{arch_name}'. Chọn: 'hybrid', 'cnn1d', 'bilstm'.")


@torch.no_grad()
def _measure_module_benchmark(
    model: nn.Module, x_single: torch.Tensor, x_batch: torch.Tensor
) -> Dict[str, object]:
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_kb = round(num_params * 4 / 1024.0, 2)

    if hasattr(model, "attention"):
        out_batch, attn_w = model(x_batch, return_attn=True)
        attn_shape = list(attn_w.shape)
    else:
        out_batch = model(x_batch)
        attn_shape = None

    single_times, batch_times = [], []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = model(x_single)
        single_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _ = model(x_batch)
        batch_times.append((time.perf_counter() - t1) * 1000.0)

    single_lat = round(float(sorted(single_times)[len(single_times) // 2]), 3)
    batch_lat = round(float(sorted(batch_times)[len(batch_times) // 2]), 3)
    fps = round(1000.0 / max(single_lat, 1e-3), 1)
    entry: Dict[str, object] = {
        "model_class": model.__class__.__name__,
        "trainable_parameters": int(num_params),
        "model_size_kb": size_kb,
        "input_shape": list(x_batch.shape),
        "output_shape": list(out_batch.shape),
        "single_latency_ms": single_lat,
        "batch32_latency_ms": batch_lat,
        "max_theoretical_fps": fps,
    }
    if attn_shape is not None:
        entry["attention_weights_shape"] = attn_shape
    return entry


@torch.no_grad()
def benchmark_models() -> Dict[str, dict]:
    """Khởi tạo, đo số lượng tham số và tốc độ suy luận thực tế của các kiến trúc mạng."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    x_batch = torch.randn(32, SEQ_LEN, FEATURE_DIM)
    x_single = torch.randn(1, SEQ_LEN, FEATURE_DIM)

    results = {}
    print("=" * 84)
    print("BƯỚC 4: SO SÁNH & BENCHMARK KIẾN TRÚC TRIPLE-MODEL (POSE + SEQUENCE + AIR-INK CNN)")
    print("=" * 84)
    print(f"{'Kiến trúc':<12} | {'Lớp Model':<26} | {'Tham số':>10} | {'Dung lượng':>10} | {'Latency':>18}")
    print("-" * 84)

    benchmarks = [
        (arch_key, build_model(arch_key), x_single, x_batch)
        for arch_key in ("cnn1d", "bilstm", "hybrid")
    ]
    benchmarks.append((
        "static_pose",
        build_pose_model(),
        torch.randn(1, POSE_FEATURE_DIM, 21),
        torch.randn(32, POSE_FEATURE_DIM, 21),
    ))
    benchmarks.append((
        "air_ink",
        build_air_ink_model(),
        torch.randn(1, 1, AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE),
        torch.randn(32, 1, AIR_INK_IMG_SIZE, AIR_INK_IMG_SIZE),
    ))

    for arch_key, mod, xs, xb in benchmarks:
        info = _measure_module_benchmark(mod, xs, xb)
        results[arch_key] = info
        print(
            f"{arch_key:<12} | {info['model_class']:<26} | {info['trainable_parameters']:>10,d} | "
            f"{info['model_size_kb']:>7.2f} KB | {info['single_latency_ms']:>9.3f} ms (~{int(info['max_theoretical_fps'])} FPS)"
        )

    out_path = DIAGNOSTICS_DIR / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("-" * 84)
    print(f"[HOÀN TẤT BƯỚC 4] Đã lưu báo cáo so sánh kiến trúc tại: {out_path}")
    return results


if __name__ == "__main__":
    benchmark_models()

