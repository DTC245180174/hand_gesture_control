"""
GIAO DIỆN BẢNG ĐIỀU KHIỂN ỨNG DỤNG DESKTOP (MODERN GUI LAUNCHER)
================================================================
Bảng điều khiển trung tâm (Control Center) thiết kế hiện đại, trực quan cho PowerPoint Hand Control:
- Chia luồng thao tác rõ ràng qua 3 tab vừa vặn trên mọi kích thước màn hình Laptop/Desktop:
  1. Thiết Lập Nhanh & Chế Độ (5 chế độ 1 ngón + sơ đồ luồng trực tiếp, Camera, Bộ chọn góc PiP 2x2, Bảo vệ thuyết trình)
  2. Cử Chỉ & Phím Tắt (Tra cứu đủ 8 cử chỉ động, phím tắt 1/2/3/C/D/M/F8/F9/Q và 5 hình AI)
  3. Tinh Chỉnh AI & Thu Mẫu (Ngưỡng tin cậy, Cooldown, Smoothing, thông số 3 model & công cụ Step 1B/1C)
- Thanh tiêu đề hiển thị trạng thái mô hình + nút Khôi phục mặc định.
- Thanh tác vụ đáy (Footer) luôn cố định với nút ▶ BẮT ĐẦU ĐIỀU KHIỂN POWERPOINT.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import HAND_LANDMARKER_TASK_PATH, MODELS_DIR, OUTPUTS_DIR

SETTINGS_FILE = OUTPUTS_DIR / "user_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera_idx": 0,
    "pip_corner": "bottom_right",
    "conf_threshold": 0.85,
    "cooldown_sec": 2.0,
    "pointer_smoothing": 0.60,
    "pen_draw_click": True,
    "point_1f_mode": "laser_ai",
    "ppt_only_lock": True,
    "bottom_deadzone": True,
    "audio_beep": True,
    "start_pip": False,
    "start_paused": False,
}

CORNER_LABELS_TO_KEY = {
    "Góc Dưới - Phải (Mặc định)": "bottom_right",
    "Góc Dưới - Trái": "bottom_left",
    "Góc Trên - Phải": "top_right",
    "Góc Trên - Trái": "top_left",
}
CORNER_KEY_TO_LABEL = {v: k for k, v in CORNER_LABELS_TO_KEY.items()}

VALID_POINT_1F_MODES = ("laser_ai", "laser_pen", "ai_shapes", "pen", "laser")

MODE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "key": "laser_ai",
        "title": "Laser + AI Nắn Hình",
        "badge": "KHUYÊN DÙNG",
        "accent": "#10B981",
        "active_bg": "#0D2B24",
        "summary": "Chỉ Laser mượt; giữ yên 0.7s để vẽ, thả tay AI tự nắn hình chuẩn rồi về Laser.",
        "flow": [
            ("🔴 Laser", "#EF4444"),
            ("⏱ 0.7s", "#F59E0B"),
            ("✏️ Vẽ tay", "#38BDF8"),
            ("✨ AI Hình", "#10B981"),
            ("🔴 Về Laser", "#EF4444"),
        ],
        "tip": "Không cần đổi công cụ tay: vẽ xong nét tự biến thành hình chuẩn rồi trả về Laser.",
    },
    {
        "key": "laser_pen",
        "title": "Laser + Bút Vẽ Tay",
        "badge": "KẾT HỢP",
        "accent": "#14B8A6",
        "active_bg": "#0D292B",
        "summary": "Chỉ Laser mượt; giữ yên 0.7s để vẽ tự do, nghỉ 1.0s tự quay về Laser.",
        "flow": [
            ("🔴 Laser", "#EF4444"),
            ("⏱ 0.7s", "#F59E0B"),
            ("✏️ Vẽ tự do", "#38BDF8"),
            ("⏸ 1.0s", "#94A3B8"),
            ("🔴 Về Laser", "#EF4444"),
        ],
        "tip": "Phù hợp khi cần vừa chỉ nội dung vừa khoanh tròn hoặc viết chú thích tự do lên slide.",
    },
    {
        "key": "ai_shapes",
        "title": "Chỉ Vẽ & AI Nắn Hình",
        "badge": "VẼ NHANH",
        "accent": "#F59E0B",
        "active_bg": "#2B210E",
        "summary": "Giơ 1 ngón vẽ ngay; dừng 0.35s tự nắn Tròn / Vuông / Tam giác / Mũi tên / Sao.",
        "flow": [
            ("✏️ Vẽ ngay", "#38BDF8"),
            ("⏱ 0.35s", "#F59E0B"),
            ("✨ AI Nắn Hình", "#10B981"),
            ("📌 Ghim Slide", "#A78BFA"),
        ],
        "tip": "Tiện khi vẽ liên tiếp nhiều hình sơ đồ. Giữ 2 ngón (Trỏ + Giữa) để bật cục tẩy.",
    },
    {
        "key": "pen",
        "title": "Chỉ Bút Mực PowerPoint (Ctrl+P)",
        "badge": "THUẦN BÚT",
        "accent": "#38BDF8",
        "active_bg": "#0C2438",
        "summary": "Bật bút mực chính chủ PowerPoint — giơ 1 ngón trỏ để nhấn kéo vẽ tự do.",
        "flow": [
            ("🖊️ Bật Ctrl+P", "#38BDF8"),
            ("☝️ Kéo vẽ nét tự do", "#60A5FA"),
            ("✌️ 2 ngón = Cục tẩy", "#EC4899"),
        ],
        "tip": "Nét mực giữ nguyên trong slide trình chiếu; nhấn phím [C] để xóa hết mực.",
    },
    {
        "key": "laser",
        "title": "Chỉ Tia Laser Đỏ (Ctrl+L)",
        "badge": "CHỈ TRỎ",
        "accent": "#F43F5E",
        "active_bg": "#2C111B",
        "summary": "Chỉ di chuyển tia Laser đỏ chính chủ PowerPoint, không bao giờ vẽ mực.",
        "flow": [
            ("🔴 Bật Ctrl+L", "#F43F5E"),
            ("☝️ Trỏ mượt theo tay", "#FB7185"),
            ("🛡️ Không để lại mực", "#94A3B8"),
        ],
        "tip": "Chế độ an toàn nhất khi bạn chỉ cần chỉ điểm trên slide mà không cần vẽ.",
    },
]
MODE_DEF_MAP: Dict[str, Dict[str, Any]] = {item["key"]: item for item in MODE_DEFINITIONS}


def _enable_windows_dpi_awareness() -> None:
    """Bật chế độ hiển thị sắc nét (High-DPI) trên màn hình Windows."""
    if os.name == "nt":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def load_user_settings() -> Dict[str, Any]:
    """Đọc cấu hình người dùng đã lưu từ `outputs/user_settings.json`."""
    cfg = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                cfg.update(saved)
                if "point_1f_mode" not in saved and "pen_draw_click" in saved:
                    cfg["point_1f_mode"] = "pen" if bool(saved["pen_draw_click"]) else "laser"
        except Exception:
            pass
    return cfg


def save_user_settings(cfg: Dict[str, Any]) -> None:
    """Lưu cấu hình người dùng xuống `outputs/user_settings.json`."""
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def load_trained_model_metadata() -> Dict[str, Any]:
    """Đọc trực tiếp thông số thực tế từ `best_gesture_model.pt` và `best_air_ink_model.pt`."""
    ckpt_path = MODELS_DIR / "best_gesture_model.pt"
    air_ink_path = MODELS_DIR / "best_air_ink_model.pt"
    info: Dict[str, Any] = {
        "exists": ckpt_path.exists(),
        "path": str(ckpt_path),
        "arch": "N/A",
        "epoch": 0,
        "val_acc": 0.0,
        "val_f1": 0.0,
        "webcam_acc": 0.0,
        "size_mb": 0.0,
        "air_ink_exists": air_ink_path.exists(),
        "air_ink_epoch": 0,
        "air_ink_acc": 0.0,
        "mp_ready": HAND_LANDMARKER_TASK_PATH.exists(),
    }
    if not ckpt_path.exists() and not air_ink_path.exists():
        return info

    try:
        import torch

        if ckpt_path.exists():
            info["size_mb"] = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            info["arch"] = str(ckpt.get("arch_name", "hybrid")).upper()
            info["epoch"] = int(ckpt.get("best_epoch", 0))
            metrics = ckpt.get("best_metrics", {})
            info["val_acc"] = float(metrics.get("val_acc", 0.0))
            info["val_f1"] = float(metrics.get("val_f1_macro", metrics.get("val_f1", 0.0)))
            info["webcam_acc"] = float(metrics.get("webcam_clean_acc", 0.0))

        if air_ink_path.exists():
            a_ckpt = torch.load(air_ink_path, map_location="cpu", weights_only=False)
            a_metrics = a_ckpt.get("best_metrics", {})
            info["air_ink_epoch"] = int(a_ckpt.get("best_epoch", 0))
            info["air_ink_acc"] = float(a_metrics.get("val_acc", 0.0))
    except Exception as e:
        info["error"] = str(e)
    return info


class PowerPointHandControlApp:
    """Cửa sổ bảng điều khiển PowerPoint Hand Control thiết kế theo từng tab trực quan."""

    BG_ROOT = "#0B1120"
    BG_SURFACE = "#111827"
    BG_CARD = "#162032"
    BORDER_SUBTLE = "#24324A"
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#CBD5E1"
    TEXT_MUTED = "#94A3B8"
    ACCENT_SKY = "#38BDF8"
    ACCENT_EMERALD = "#10B981"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PowerPoint Hand Control — AI Gesture Control Center")
        self.root.configure(bg=self.BG_ROOT)

        self._apply_responsive_geometry()

        self.user_cfg = load_user_settings()
        self.model_meta = load_trained_model_metadata()

        saved_cam = int(self.user_cfg.get("camera_idx", 0))
        cam_options = [
            "0 - Webcam mặc định (Laptop / USB chính)",
            "1 - Camera ngoài (USB Webcam phụ)",
            "2 - Camera ảo / OBS Virtual Camera",
        ]
        default_cam_str = cam_options[saved_cam] if 0 <= saved_cam < len(cam_options) else cam_options[0]

        saved_corner_key = str(self.user_cfg.get("pip_corner", "bottom_right"))
        default_corner_str = CORNER_KEY_TO_LABEL.get(saved_corner_key, "Góc Dưới - Phải (Mặc định)")

        self.camera_var = tk.StringVar(value=default_cam_str)
        self.corner_var = tk.StringVar(value=default_corner_str)
        self.threshold_var = tk.DoubleVar(value=float(self.user_cfg.get("conf_threshold", 0.85)))
        self.cooldown_var = tk.DoubleVar(value=float(self.user_cfg.get("cooldown_sec", 2.0)))
        self.smoothing_var = tk.DoubleVar(value=float(self.user_cfg.get("pointer_smoothing", 0.60)))

        raw_mode = str(self.user_cfg.get("point_1f_mode", "")).lower().strip()
        if raw_mode not in VALID_POINT_1F_MODES:
            raw_mode = "laser_ai" if self.model_meta.get("air_ink_exists", False) else (
                "pen" if bool(self.user_cfg.get("pen_draw_click", True)) else "laser"
            )
        if raw_mode in ("laser_ai", "ai_shapes") and not self.model_meta.get("air_ink_exists", False):
            raw_mode = "laser_pen"

        self.point_1f_mode_var = tk.StringVar(value=raw_mode)
        self.pen_draw_var = tk.BooleanVar(value=(raw_mode != "laser"))

        self.ppt_lock_var = tk.BooleanVar(value=bool(self.user_cfg.get("ppt_only_lock", True)))
        self.deadzone_var = tk.BooleanVar(value=bool(self.user_cfg.get("bottom_deadzone", True)))
        self.beep_var = tk.BooleanVar(value=bool(self.user_cfg.get("audio_beep", True)))
        self.start_pip_var = tk.BooleanVar(value=bool(self.user_cfg.get("start_pip", False)))
        self.start_paused_var = tk.BooleanVar(value=bool(self.user_cfg.get("start_paused", False)))

        mode_title = MODE_DEF_MAP.get(raw_mode, MODE_DEFINITIONS[0])["title"]
        self.status_var = tk.StringVar(value=f"Sẵn sàng — Chế độ: {mode_title}  •  Tự lưu cấu hình")

        self._mode_card_widgets: Dict[str, Dict[str, Any]] = {}
        self._corner_buttons: Dict[str, tk.Button] = {}

        self._configure_ttk_styles()
        self._build_ui()
        self._refresh_pen_mode_buttons()
        self._refresh_corner_buttons()

    def _apply_responsive_geometry(self) -> None:
        """Tính kích thước cửa sổ vừa vặn theo màn hình và căn giữa."""
        sw = max(960, int(self.root.winfo_screenwidth() or 1280))
        sh = max(680, int(self.root.winfo_screenheight() or 800))
        win_w = min(1080, max(920, sw - 48))
        win_h = min(805, max(650, sh - 55))
        pos_x = max(0, (sw - win_w) // 2)
        pos_y = max(0, (sh - win_h) // 2 - 10)
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(min(920, sw - 40), min(640, sh - 50))

    def _configure_ttk_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Dark.TCombobox",
            fieldbackground="#1E293B",
            background="#1E293B",
            foreground="#F8FAFC",
            arrowcolor=self.ACCENT_SKY,
            bordercolor="#334155",
            lightcolor="#334155",
            darkcolor="#0F172A",
            padding=5,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#1E293B")],
            foreground=[("readonly", "#F8FAFC")],
            selectbackground=[("readonly", "#0284C7")],
            selectforeground=[("readonly", "#FFFFFF")],
        )

        style.configure(
            "App.TNotebook",
            background=self.BG_ROOT,
            borderwidth=0,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            "App.TNotebook.Tab",
            background="#162032",
            foreground="#94A3B8",
            padding=[16, 7],
            font=("Segoe UI", 9, "bold"),
            borderwidth=0,
            focuscolor="#1E293B",
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", "#1E293B"), ("active", "#1A273D")],
            foreground=[("selected", self.ACCENT_SKY), ("active", "#E2E8F0")],
        )

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=0)

        self._build_header()
        self._build_notebook_tabs()
        self._build_footer()

    # ------------------------------------------------------------------
    # HÀNG 0: HEADER GỌN SẮC NÉT & MODEL PILLS
    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=self.BG_SURFACE, padx=18, pady=7)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        title_block = tk.Frame(header, bg=self.BG_SURFACE)
        title_block.grid(row=0, column=0, sticky="w")

        tk.Label(
            title_block,
            text="POWERPOINT HAND CONTROL",
            font=("Segoe UI", 15, "bold"),
            fg=self.ACCENT_SKY,
            bg=self.BG_SURFACE,
        ).pack(side=tk.LEFT)

        tk.Label(
            title_block,
            text="  •  Điều khiển thuyết trình AI (IPN HandS + MediaPipe)",
            font=("Segoe UI", 9),
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
        ).pack(side=tk.LEFT, pady=(3, 0))

        tk.Button(
            header,
            text="↺  Khôi phục mặc định",
            command=self._on_reset_defaults,
            bg="#1E293B",
            fg=self.TEXT_SECONDARY,
            activebackground="#334155",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=12,
            pady=5,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground="#334155",
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        pill_row = tk.Frame(header, bg=self.BG_SURFACE)
        pill_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))

        m = self.model_meta
        if m["exists"]:
            self._add_status_pill(pill_row, "AI Cử Chỉ", f"{m['arch']} Ep{m['epoch']} · {m['val_acc']:.1f}%", "#38BDF8")
            if m["webcam_acc"] > 0:
                self._add_status_pill(pill_row, "Webcam Calib", f"{m['webcam_acc']:.1f}%", "#A78BFA")
            else:
                self._add_status_pill(pill_row, "Val F1-Macro", f"{m['val_f1']:.1f}%", "#A78BFA")
        else:
            self._add_status_pill(pill_row, "AI Cử Chỉ", "Chưa có model", "#F43F5E")

        if m.get("air_ink_exists", False):
            self._add_status_pill(
                pill_row,
                "AI Nắn Hình (M3)",
                f"Ep{m['air_ink_epoch']} · {m['air_ink_acc']:.1f}%",
                "#10B981",
            )
        else:
            self._add_status_pill(pill_row, "AI Nắn Hình (M3)", "Chưa huấn luyện", "#F59E0B")

        mp_ok = bool(m.get("mp_ready", False))
        self._add_status_pill(
            pill_row,
            "MediaPipe Hands",
            "Sẵn sàng" if mp_ok else "Thiếu task file",
            "#2DD4BF" if mp_ok else "#F43F5E",
        )

        tk.Frame(self.root, bg="#1E293B", height=1).grid(row=0, column=0, sticky="sew")

    def _add_status_pill(self, parent: tk.Widget, label: str, value: str, accent: str) -> None:
        pill = tk.Frame(
            parent,
            bg="#0F172A",
            padx=9,
            pady=3,
            highlightthickness=1,
            highlightbackground="#24324A",
        )
        pill.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(pill, text="●", font=("Segoe UI", 8), fg=accent, bg="#0F172A").pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(pill, text=f"{label}:", font=("Segoe UI", 8, "bold"), fg=self.TEXT_MUTED, bg="#0F172A").pack(
            side=tk.LEFT
        )
        tk.Label(pill, text=f" {value}", font=("Segoe UI", 8, "bold"), fg=self.TEXT_PRIMARY, bg="#0F172A").pack(
            side=tk.LEFT
        )

    # ------------------------------------------------------------------
    # HÀNG 1: NOTEBOOK 3 TAB VỪA VẶN KHÔNG CẮT XÉN
    # ------------------------------------------------------------------
    def _build_notebook_tabs(self) -> None:
        wrapper = tk.Frame(self.root, bg=self.BG_ROOT, padx=16, pady=6)
        wrapper.grid(row=1, column=0, sticky="nsew")
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(wrapper, style="App.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_setup = tk.Frame(self.notebook, bg=self.BG_ROOT, pady=6)
        self.tab_guide = tk.Frame(self.notebook, bg=self.BG_ROOT, pady=6)
        self.tab_advanced = tk.Frame(self.notebook, bg=self.BG_ROOT, pady=6)

        self.notebook.add(self.tab_setup, text="  🎯  Thiết Lập Nhanh & Chế Độ  ")
        self.notebook.add(self.tab_guide, text="  ✋  Cử Chỉ & Phím Tắt  ")
        self.notebook.add(self.tab_advanced, text="  ⚙️  Tinh Chỉnh AI & Thu Mẫu  ")

        self._build_tab_setup()
        self._build_tab_guide()
        self._build_tab_advanced()

    # ------------------------------------------------------------------
    # TAB 1: THIẾT LẬP NHANH & CHẾ ĐỘ
    # ------------------------------------------------------------------
    def _build_tab_setup(self) -> None:
        self.tab_setup.grid_columnconfigure(0, weight=56, uniform="t1col")
        self.tab_setup.grid_columnconfigure(1, weight=44, uniform="t1col")
        self.tab_setup.grid_rowconfigure(0, weight=1)

        # ---- Cột trái: 5 Thẻ Chế độ + Quy trình trực quan ----
        left_card = tk.Frame(
            self.tab_setup,
            bg=self.BG_SURFACE,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        tk.Label(
            left_card,
            text="CHẾ ĐỘ KHI GIƠ 1 NGÓN TRỎ (POINT 1F)",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w")

        tk.Label(
            left_card,
            text="Nhấp vào thẻ để chọn cách hoạt động khi giơ 1 ngón trỏ trước camera:",
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(1, 6))

        cards_host = tk.Frame(left_card, bg=self.BG_SURFACE)
        cards_host.pack(fill=tk.X)

        for spec in MODE_DEFINITIONS:
            self._create_mode_card(cards_host, spec)

        # Khung Luồng hoạt động trực quan ở đáy cột trái
        flow_box = tk.Frame(
            left_card,
            bg="#0F172A",
            padx=10,
            pady=7,
            highlightthickness=1,
            highlightbackground="#24324A",
        )
        flow_box.pack(fill=tk.X, pady=(6, 0))

        self.flow_title_lbl = tk.Label(
            flow_box,
            text="",
            font=("Segoe UI", 8, "bold"),
            fg=self.ACCENT_SKY,
            bg="#0F172A",
            anchor="w",
        )
        self.flow_title_lbl.pack(fill=tk.X)

        self.flow_steps_frame = tk.Frame(flow_box, bg="#0F172A")
        self.flow_steps_frame.pack(fill=tk.X, pady=(4, 3))

        self.flow_tip_lbl = tk.Label(
            flow_box,
            text="",
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg="#0F172A",
            anchor="w",
            justify=tk.LEFT,
            wraplength=515,
        )
        self.flow_tip_lbl.pack(fill=tk.X)

        # ---- Cột phải: Camera + Góc PiP 2x2 + Bảo vệ thuyết trình ----
        right_col = tk.Frame(self.tab_setup, bg=self.BG_ROOT)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right_col.grid_columnconfigure(0, weight=1)
        right_col.grid_rowconfigure(0, weight=0)
        right_col.grid_rowconfigure(1, weight=1)

        dev_card = tk.Frame(
            right_col,
            bg=self.BG_SURFACE,
            padx=14,
            pady=7,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        dev_card.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        tk.Label(
            dev_card,
            text="CAMERA & VỊ TRÍ CỬA SỔ NHỎ (PiP)",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 4))

        cam_row = tk.Frame(dev_card, bg=self.BG_SURFACE)
        cam_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(
            cam_row,
            text="Camera:",
            font=("Segoe UI", 8, "bold"),
            fg=self.TEXT_SECONDARY,
            bg=self.BG_SURFACE,
            width=8,
            anchor="w",
        ).pack(side=tk.LEFT)
        self.cam_combo = ttk.Combobox(
            cam_row,
            textvariable=self.camera_var,
            style="Dark.TCombobox",
            values=[
                "0 - Webcam mặc định (Laptop / USB chính)",
                "1 - Camera ngoài (USB Webcam phụ)",
                "2 - Camera ảo / OBS Virtual Camera",
            ],
            state="readonly",
            font=("Segoe UI", 8),
        )
        self.cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            dev_card,
            text="Góc đặt cửa sổ Camera nổi trên Slide PowerPoint:",
            font=("Segoe UI", 8, "bold"),
            fg=self.TEXT_SECONDARY,
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 3))

        screen_mock = tk.Frame(
            dev_card,
            bg="#0B1120",
            padx=6,
            pady=4,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        screen_mock.pack(fill=tk.X)
        screen_mock.grid_columnconfigure(0, weight=1, uniform="pipgrid")
        screen_mock.grid_columnconfigure(1, weight=1, uniform="pipgrid")

        corner_specs = [
            ("top_left", "↖ Trên - Trái", 0, 0),
            ("top_right", "↗ Trên - Phải", 0, 1),
            ("bottom_left", "↙ Dưới - Trái", 2, 0),
            ("bottom_right", "↘ Dưới - Phải", 2, 1),
        ]
        for key, text, r_idx, c_idx in corner_specs:
            btn = tk.Button(
                screen_mock,
                text=text,
                command=lambda k=key: self._select_pip_corner(k),
                font=("Segoe UI", 8, "bold"),
                relief="flat",
                pady=3,
                cursor="hand2",
            )
            pad_x = (0, 5) if c_idx == 0 else (5, 0)
            btn.grid(row=r_idx, column=c_idx, sticky="ew", padx=pad_x)
            self._corner_buttons[key] = btn

        tk.Label(
            screen_mock,
            text="[ Khung Slide PowerPoint Toàn Màn Hình ]",
            font=("Segoe UI", 7),
            fg="#64748B",
            bg="#0B1120",
            pady=1,
        ).grid(row=1, column=0, columnspan=2)

        # Card dưới phải: 5 Tuỳ chọn an toàn gọn đẹp, hiện đủ 100%
        safe_card = tk.Frame(
            right_col,
            bg=self.BG_SURFACE,
            padx=14,
            pady=7,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        safe_card.grid(row=1, column=0, sticky="nsew")

        tk.Label(
            safe_card,
            text="AN TOÀN & HỖ TRỢ THUYẾT TRÌNH",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 2))

        self._create_toggle_card(
            safe_card,
            self.ppt_lock_var,
            "Khóa mục tiêu vào PowerPoint",
            "Tự tạm chờ khi bạn mở Chrome, Zalo hoặc Desktop.",
            wrap=385,
        )
        self._create_toggle_card(
            safe_card,
            self.deadzone_var,
            "Vùng nghỉ hạ tay 15% mép dưới",
            "Bỏ qua bàn tay khi thả lỏng ngang bụng/ngực.",
            wrap=385,
        )
        self._create_toggle_card(
            safe_card,
            self.start_pip_var,
            "Mở ngay cửa sổ góc nhỏ (PiP)",
            "Camera nổi xuyên chuột, không mất khi bật F5.",
            wrap=385,
        )
        self._create_toggle_card(
            safe_card,
            self.beep_var,
            "Âm báo 'Tít' nhẹ khi nhận lệnh",
            "Phản hồi âm thanh ngắn khi chuyển Slide / Zoom.",
            wrap=385,
        )
        self._create_toggle_card(
            safe_card,
            self.start_paused_var,
            "Khởi động Tạm dừng (PAUSED)",
            "Mở Camera căn góc trước; bấm [F9] khi bắt đầu.",
            wrap=385,
        )

    def _create_mode_card(self, parent: tk.Widget, spec: Dict[str, Any]) -> None:
        key = spec["key"]
        card = tk.Frame(
            parent,
            bg=self.BG_CARD,
            padx=10,
            pady=4,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
            cursor="hand2",
        )
        card.pack(fill=tk.X, pady=(0, 3))

        top = tk.Frame(card, bg=self.BG_CARD, cursor="hand2")
        top.pack(fill=tk.X)

        dot_lbl = tk.Label(
            top,
            text="○",
            font=("Segoe UI", 10, "bold"),
            fg=spec["accent"],
            bg=self.BG_CARD,
            cursor="hand2",
        )
        dot_lbl.pack(side=tk.LEFT, padx=(0, 6))

        title_lbl = tk.Label(
            top,
            text=spec["title"],
            font=("Segoe UI", 9, "bold"),
            fg=self.TEXT_PRIMARY,
            bg=self.BG_CARD,
            cursor="hand2",
        )
        title_lbl.pack(side=tk.LEFT)

        badge_lbl = tk.Label(
            top,
            text=spec["badge"],
            font=("Segoe UI", 7, "bold"),
            fg="#0F172A",
            bg=spec["accent"],
            padx=6,
            pady=0,
            cursor="hand2",
        )
        badge_lbl.pack(side=tk.RIGHT)

        desc_lbl = tk.Label(
            card,
            text=spec["summary"],
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_CARD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=515,
            cursor="hand2",
        )
        desc_lbl.pack(fill=tk.X, padx=(18, 0), pady=(0, 0))

        for w in (card, top, dot_lbl, title_lbl, badge_lbl, desc_lbl):
            w.bind("<Button-1>", lambda _e, m=key: self._set_point_1f_mode(m))

        self._mode_card_widgets[key] = {
            "card": card,
            "top": top,
            "dot": dot_lbl,
            "title": title_lbl,
            "badge": badge_lbl,
            "desc": desc_lbl,
            "spec": spec,
        }
        setattr(self, f"btn_mode_{key}", card)

    def _select_pip_corner(self, corner_key: str) -> None:
        self.corner_var.set(CORNER_KEY_TO_LABEL.get(corner_key, "Góc Dưới - Phải (Mặc định)"))
        self._refresh_corner_buttons()

    def _refresh_corner_buttons(self) -> None:
        active_key = self._get_selected_corner_key()
        for key, btn in self._corner_buttons.items():
            if key == active_key:
                btn.config(bg="#0284C7", fg="#FFFFFF", activebackground="#0369A1")
            else:
                btn.config(bg="#1E293B", fg="#94A3B8", activebackground="#334155")

    # ------------------------------------------------------------------
    # TAB 2: HƯỚNG DẪN CỬ CHỈ & PHÍM TẮT (HIỂN THỊ ĐẦY ĐỦ CẢ 8 CỬ CHỈ)
    # ------------------------------------------------------------------
    def _build_tab_guide(self) -> None:
        self.tab_guide.grid_columnconfigure(0, weight=54, uniform="t2col")
        self.tab_guide.grid_columnconfigure(1, weight=46, uniform="t2col")
        self.tab_guide.grid_rowconfigure(0, weight=1)

        # ---- Cột trái: Bảng 8 cử chỉ tay ----
        gest_card = tk.Frame(
            self.tab_guide,
            bg=self.BG_SURFACE,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        gest_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        tk.Label(
            gest_card,
            text="8 CỬ CHỈ TAY ĐỘNG ĐIỀU KHIỂN SLIDE",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 6))

        gestures_info = [
            ("0 · IDLE", "#64748B", "Không gửi lệnh", "Đứng yên, nắm hoặc mở tay tự nhiên (vùng nghỉ 15% dưới)."),
            ("1 · SWIPE LEFT", "#10B981", "Next Slide (→)", "Gạt gọn bàn tay dứt khoát từ Phải sang Trái."),
            ("2 · SWIPE RIGHT", "#F59E0B", "Prev Slide (←)", "Gạt gọn bàn tay dứt khoát từ Trái sang Phải."),
            ("3 · POINT 1F", "#0EA5E9", "Laser / Vẽ / AI Hình", "Giơ thẳng 1 ngón trỏ (theo chế độ ở Tab 1)."),
            ("4 · POINT 2F", "#EC4899", "Cục tẩy (Ctrl+E)", "Giơ 2 ngón (Trỏ + Giữa) để xóa hình hoặc nét mực."),
            ("5 · ZOOM IN", "#EAB308", "Phóng to (Ctrl + '=')", "Chụm tay rồi bung xòe rộng 5 ngón tay."),
            ("6 · ZOOM OUT", "#2563EB", "Thu nhỏ (Ctrl + '-')", "Xòe rộng 5 ngón tay rồi chụm khép gọn vào."),
            ("7 · CLICK", "#8B5CF6", "Click chuột trái", "Chớp mở-xòe bàn tay 2 lần liên tiếp (khi bật [M])."),
        ]
        for code, color, action, detail in gestures_info:
            row = tk.Frame(
                gest_card,
                bg=self.BG_CARD,
                padx=8,
                pady=3,
                highlightthickness=1,
                highlightbackground=self.BORDER_SUBTLE,
            )
            row.pack(fill=tk.X, pady=1)
            row.grid_columnconfigure(1, weight=1)

            tk.Label(
                row,
                text=code,
                font=("Consolas", 8, "bold"),
                fg="#FFFFFF",
                bg=color,
                width=16,
                pady=1,
            ).grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 8))

            tk.Label(
                row,
                text=action,
                font=("Segoe UI", 8, "bold"),
                fg=self.TEXT_PRIMARY,
                bg=self.BG_CARD,
                anchor="w",
            ).grid(row=0, column=1, sticky="ew")

            tk.Label(
                row,
                text=detail,
                font=("Segoe UI", 7),
                fg=self.TEXT_MUTED,
                bg=self.BG_CARD,
                anchor="w",
                wraplength=390,
                justify=tk.LEFT,
            ).grid(row=1, column=1, sticky="ew")

        # ---- Cột phải: Phím tắt nhanh + 5 Hình AI ----
        right_col = tk.Frame(self.tab_guide, bg=self.BG_ROOT)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right_col.grid_columnconfigure(0, weight=1)
        right_col.grid_rowconfigure(0, weight=0)
        right_col.grid_rowconfigure(1, weight=1)

        key_card = tk.Frame(
            right_col,
            bg=self.BG_SURFACE,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        key_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(
            key_card,
            text="PHÍM TẮT NHANH TRONG LÚC THUYẾT TRÌNH",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 6))

        hotkeys = [
            ("[1]", "Bật nhanh Tia Laser Đỏ (Ctrl+L)"),
            ("[2]", "Bật Bút Vẽ Tay / Hủy AI Nắn Hình"),
            ("[3]", "Trả về Con Trỏ Chuột Mặc Định"),
            ("[C]", "Xóa toàn bộ nét mực trên Slide (Phím E)"),
            ("[D]", "Đổi vòng chế độ 1 ngón: Laser+AI / AI / Bút / Laser"),
            ("[M]", "Bật / Tắt chế độ Click chuột tương tác"),
            ("[F8] / [P]", "Thu nhỏ / Phóng to cửa sổ Camera góc màn hình"),
            ("[F9] / [O]", "Tạm dừng / Tiếp tục nhận diện AI (PAUSE)"),
            ("[Q] / [ESC]", "Dừng khẩn cấp (Kill-Switch) và thoát an toàn"),
        ]
        for hk, desc in hotkeys:
            r = tk.Frame(key_card, bg=self.BG_SURFACE)
            r.pack(fill=tk.X, pady=2)
            r.grid_columnconfigure(1, weight=1)

            tk.Label(
                r,
                text=hk,
                font=("Consolas", 8, "bold"),
                fg=self.ACCENT_SKY,
                bg="#1E293B",
                width=11,
                pady=1,
                highlightthickness=1,
                highlightbackground="#334155",
            ).grid(row=0, column=0, sticky="w", padx=(0, 9))

            tk.Label(
                r,
                text=desc,
                font=("Segoe UI", 8),
                fg=self.TEXT_SECONDARY,
                bg=self.BG_SURFACE,
                anchor="w",
            ).grid(row=0, column=1, sticky="ew")

        shapes_card = tk.Frame(
            right_col,
            bg=self.BG_SURFACE,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        shapes_card.grid(row=1, column=0, sticky="nsew")

        tk.Label(
            shapes_card,
            text="5 HÌNH AI TỰ ĐỘNG NẮN ĐẸP (MODEL 3)",
            font=("Segoe UI", 9, "bold"),
            fg=self.ACCENT_EMERALD,
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 6))

        chip_row = tk.Frame(shapes_card, bg=self.BG_SURFACE)
        chip_row.pack(fill=tk.X, pady=(0, 7))
        shapes_list = ("○ Tròn", "▭ Chữ Nhật", "△ Tam Giác", "↗ Mũi Tên", "★ Ngôi Sao")
        for idx, shape_name in enumerate(shapes_list):
            chip_row.grid_columnconfigure(idx, weight=1)
            tk.Label(
                chip_row,
                text=shape_name,
                font=("Segoe UI", 8, "bold"),
                fg="#E2E8F0",
                bg="#1E293B",
                padx=4,
                pady=3,
                highlightthickness=1,
                highlightbackground="#334155",
            ).grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 4, 0))

        tk.Label(
            shapes_card,
            text="• Ở chế độ Laser + AI Nắn Hình: giữ ngón trỏ yên ~0.7s cho tới khi vòng tròn chuyển xanh lá rồi vẽ 1 nét liền mạch.\n• Giữ nhẹ cuối nét 0.35s để AI nhận diện và vẽ lại hình vector sắc nét lên slide.",
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=405,
        ).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # TAB 3: TINH CHỈNH AI & CÔNG CỤ THU MẪU
    # ------------------------------------------------------------------
    def _build_tab_advanced(self) -> None:
        self.tab_advanced.grid_columnconfigure(0, weight=54, uniform="t3col")
        self.tab_advanced.grid_columnconfigure(1, weight=46, uniform="t3col")
        self.tab_advanced.grid_rowconfigure(0, weight=1)

        left_card = tk.Frame(
            self.tab_advanced,
            bg=self.BG_SURFACE,
            padx=15,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        tk.Label(
            left_card,
            text="TINH CHỈNH THAM SỐ AI & CON TRỎ",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 6))

        self.thresh_lbl = self._create_slider_row(
            parent=left_card,
            title="Ngưỡng độ tin cậy AI (Confidence Threshold)",
            variable=self.threshold_var,
            from_=0.60,
            to=0.98,
            res=0.01,
            fmt_fn=lambda v: f"{float(v) * 100:.0f}%",
            hint="Cao hơn (85–92%) hạn chế nhận nhầm; thấp hơn bắt cử chỉ nhanh hơn.",
        )
        self.cool_lbl = self._create_slider_row(
            parent=left_card,
            title="Thời gian nghỉ chống lặp lệnh (Cooldown)",
            variable=self.cooldown_var,
            from_=0.5,
            to=4.0,
            res=0.1,
            fmt_fn=lambda v: f"{float(v):.1f} giây",
            hint="Khoảng chờ sau khi chuyển Slide/Zoom để hạ tay về an toàn.",
        )
        self.smooth_lbl = self._create_slider_row(
            parent=left_card,
            title="Độ mượt chống rung Bút vẽ / Laser (Smoothing)",
            variable=self.smoothing_var,
            from_=0.00,
            to=0.90,
            res=0.05,
            fmt_fn=lambda v: f"{float(v) * 100:.0f}%",
            hint="Tăng lên 60–75% nếu tay rung khi viết; giảm xuống 30–40% để bám nhanh hơn.",
        )

        diag_box = tk.Frame(
            left_card,
            bg="#0F172A",
            padx=11,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        diag_box.pack(fill=tk.X, pady=(4, 0))

        tk.Label(
            diag_box,
            text="CHI TIẾT CÁC MÔ HÌNH AI ĐANG NẠP",
            font=("Segoe UI", 8, "bold"),
            fg=self.ACCENT_SKY,
            bg="#0F172A",
        ).pack(anchor="w", pady=(0, 3))

        m = self.model_meta
        gesture_line = (
            f"• Model 1/2 (Cử Chỉ Động): {m['arch']} | Ep{m['epoch']} | "
            f"Val Acc {m['val_acc']:.2f}% | Val F1 {m['val_f1']:.2f}% | "
            f"Webcam {m['webcam_acc']:.1f}% ({m['size_mb']} MB)"
            if m["exists"]
            else "• Model 1/2 (Cử Chỉ Động): Chưa tìm thấy file checkpoint (.pt)"
        )
        ink_line = (
            f"• Model 3 (AI Nắn Hình): AirInkResCNN | Ep{m['air_ink_epoch']} | Val Acc {m['air_ink_acc']:.2f}%"
            if m.get("air_ink_exists", False)
            else "• Model 3 (AI Nắn Hình): Chưa huấn luyện (chạy python main.py --step 5-ink)"
        )
        for line in (gesture_line, ink_line):
            tk.Label(
                diag_box,
                text=line,
                font=("Segoe UI", 8),
                fg=self.TEXT_SECONDARY,
                bg="#0F172A",
                anchor="w",
                justify=tk.LEFT,
                wraplength=465,
            ).pack(fill=tk.X, pady=1)

        right_card = tk.Frame(
            self.tab_advanced,
            bg=self.BG_SURFACE,
            padx=15,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        tk.Label(
            right_card,
            text="CÔNG CỤ THU MẪU DỮ LIỆU CÁ NHÂN",
            font=("Segoe UI", 9, "bold"),
            fg="#FBBF24",
            bg=self.BG_SURFACE,
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            right_card,
            text="Dùng để ghi thêm mẫu tay hoặc nét vẽ riêng theo thói quen của bạn trước khi huấn luyện lại mô hình:",
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_SURFACE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=400,
        ).pack(fill=tk.X, pady=(0, 10))

        self._create_tool_action_card(
            parent=right_card,
            title="Step 1B — Thu mẫu Cử Chỉ Webcam",
            desc="Ghi lại chuỗi khung xương 30 frames cho 8 lớp cử chỉ (trượt / zoom / trỏ / click) để tinh chỉnh AI.",
            btn_text="📹  Mở Công Cụ Thu Mẫu Cử Chỉ (Step 1B)",
            btn_bg="#1E293B",
            command=self._on_start_collector,
        )

        self._create_tool_action_card(
            parent=right_card,
            title="Step 1C — Thu mẫu Hình Vẽ Không Trung",
            desc="Ghi lại mẫu vẽ 5 hình học (Tròn, Chữ nhật, Tam giác, Mũi tên, Ngôi sao) cho Model 3.",
            btn_text="✏️  Mở Công Cụ Thu Mẫu Air-Ink (Step 1C)",
            btn_bg="#1E293B",
            command=self._on_start_air_ink_collector,
        )

        tk.Label(
            right_card,
            text="Cấu hình hiện tại tự động lưu vào `outputs/user_settings.json` mỗi khi bạn bấm Bắt đầu hoặc Khôi phục mặc định.",
            font=("Segoe UI", 8),
            fg="#64748B",
            bg=self.BG_SURFACE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=400,
        ).pack(fill=tk.X, pady=(6, 0))

    def _create_slider_row(
        self,
        parent: tk.Widget,
        title: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        res: float,
        fmt_fn: Callable[[Any], str],
        hint: str,
    ) -> tk.Label:
        box = tk.Frame(
            parent,
            bg=self.BG_CARD,
            padx=11,
            pady=6,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        box.pack(fill=tk.X, pady=(0, 7))

        top = tk.Frame(box, bg=self.BG_CARD)
        top.pack(fill=tk.X)

        tk.Label(top, text=title, font=("Segoe UI", 9, "bold"), fg=self.TEXT_PRIMARY, bg=self.BG_CARD).pack(
            side=tk.LEFT
        )
        val_lbl = tk.Label(
            top,
            text=fmt_fn(variable.get()),
            font=("Segoe UI", 9, "bold"),
            fg=self.ACCENT_SKY,
            bg=self.BG_CARD,
        )
        val_lbl.pack(side=tk.RIGHT)

        scale = tk.Scale(
            box,
            from_=from_,
            to=to,
            resolution=res,
            orient=tk.HORIZONTAL,
            variable=variable,
            showvalue=False,
            bg=self.BG_CARD,
            fg="#E2E8F0",
            troughcolor="#0B1120",
            highlightthickness=0,
            activebackground=self.ACCENT_SKY,
            bd=0,
            sliderlength=20,
            width=9,
            command=lambda v: val_lbl.config(text=fmt_fn(v)),
        )
        scale.pack(fill=tk.X, pady=(2, 2))

        tk.Label(
            box,
            text=hint,
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_CARD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=460,
        ).pack(fill=tk.X)
        return val_lbl

    def _create_toggle_card(
        self,
        parent: tk.Widget,
        var: tk.BooleanVar,
        title: str,
        desc: str,
        wrap: int = 385,
    ) -> None:
        card = tk.Frame(
            parent,
            bg=self.BG_CARD,
            padx=8,
            pady=2,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        card.pack(fill=tk.X, pady=1)

        cb = tk.Checkbutton(
            card,
            text=title,
            variable=var,
            bg=self.BG_CARD,
            fg=self.TEXT_PRIMARY,
            selectcolor="#0F172A",
            activebackground=self.BG_CARD,
            activeforeground=self.ACCENT_SKY,
            font=("Segoe UI", 8, "bold"),
            anchor="w",
            cursor="hand2",
            pady=0,
        )
        cb.pack(fill=tk.X)

        sub = tk.Label(
            card,
            text=desc,
            font=("Segoe UI", 7),
            fg=self.TEXT_MUTED,
            bg=self.BG_CARD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=wrap,
            cursor="hand2",
        )
        sub.pack(fill=tk.X, padx=(20, 0))
        sub.bind("<Button-1>", lambda _e: var.set(not var.get()))

    def _create_tool_action_card(
        self,
        parent: tk.Widget,
        title: str,
        desc: str,
        btn_text: str,
        btn_bg: str,
        command: Callable[[], None],
    ) -> None:
        card = tk.Frame(
            parent,
            bg=self.BG_CARD,
            padx=12,
            pady=9,
            highlightthickness=1,
            highlightbackground=self.BORDER_SUBTLE,
        )
        card.pack(fill=tk.X, pady=(0, 9))

        tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg=self.TEXT_PRIMARY, bg=self.BG_CARD).pack(
            anchor="w"
        )
        tk.Label(
            card,
            text=desc,
            font=("Segoe UI", 8),
            fg=self.TEXT_MUTED,
            bg=self.BG_CARD,
            anchor="w",
            justify=tk.LEFT,
            wraplength=385,
        ).pack(fill=tk.X, pady=(2, 7))

        tk.Button(
            card,
            text=btn_text,
            command=command,
            bg=btn_bg,
            fg="#E2E8F0",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
            highlightthickness=1,
            highlightbackground="#334155",
        ).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # HÀNG 2: FOOTER HÀNH ĐỘNG CỐ ĐỊNH
    # ------------------------------------------------------------------
    def _build_footer(self) -> None:
        tk.Frame(self.root, bg="#1E293B", height=1).grid(row=2, column=0, sticky="new")

        footer = tk.Frame(self.root, bg=self.BG_SURFACE, padx=18, pady=8)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=0)

        status_box = tk.Frame(footer, bg=self.BG_SURFACE)
        status_box.grid(row=0, column=0, sticky="ew", padx=(0, 14))

        tk.Label(
            status_box,
            text="●",
            font=("Segoe UI", 9),
            fg=self.ACCENT_EMERALD,
            bg=self.BG_SURFACE,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(
            status_box,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=self.TEXT_SECONDARY,
            bg=self.BG_SURFACE,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_start = tk.Button(
            footer,
            text="▶   BẮT ĐẦU ĐIỀU KHIỂN POWERPOINT",
            command=self._on_start_deployment,
            bg="#059669",
            fg="#FFFFFF",
            activebackground="#047857",
            activeforeground="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=24,
            pady=10,
            cursor="hand2",
        )
        self.btn_start.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # TRẠNG THÁI & LOGIC TƯƠNG TÁC
    # ------------------------------------------------------------------
    def _set_point_1f_mode(self, mode: str) -> None:
        m = mode.lower().strip()
        if m in ("laser_ai", "ai_shapes") and not self.model_meta.get("air_ink_exists", False):
            messagebox.showwarning(
                "Chưa có Model 3 (AI Nắn Hình)",
                "Chưa tìm thấy best_air_ink_model.pt.\nHãy chạy `python main.py --step 5-ink` trước khi bật AI Nắn Hình.",
            )
            return
        if m not in VALID_POINT_1F_MODES:
            m = DEFAULT_SETTINGS["point_1f_mode"]
        self.point_1f_mode_var.set(m)
        self.pen_draw_var.set(m != "laser")
        self._refresh_pen_mode_buttons()
        mode_title = MODE_DEF_MAP.get(m, MODE_DEFINITIONS[0])["title"]
        self.status_var.set(f"Sẵn sàng — Chế độ: {mode_title}  •  Tự lưu cấu hình")

    def _refresh_pen_mode_buttons(self) -> None:
        mode = self.point_1f_mode_var.get()
        ai_ready = bool(self.model_meta.get("air_ink_exists", False))

        for key, refs in self._mode_card_widgets.items():
            spec = refs["spec"]
            accent = spec["accent"]
            is_active = key == mode
            is_disabled = key in ("laser_ai", "ai_shapes") and not ai_ready

            if is_active:
                bg = spec["active_bg"]
                border = accent
                thickness = 2
                dot_text = "◉"
                dot_fg = accent
                title_fg = "#FFFFFF"
                desc_fg = "#CBD5E1"
                badge_bg = accent
                badge_fg = "#0B1120"
            elif is_disabled:
                bg = "#111827"
                border = "#1E293B"
                thickness = 1
                dot_text = "○"
                dot_fg = "#475569"
                title_fg = "#64748B"
                desc_fg = "#475569"
                badge_bg = "#1E293B"
                badge_fg = "#64748B"
            else:
                bg = self.BG_CARD
                border = self.BORDER_SUBTLE
                thickness = 1
                dot_text = "○"
                dot_fg = "#64748B"
                title_fg = self.TEXT_PRIMARY
                desc_fg = self.TEXT_MUTED
                badge_bg = "#24324A"
                badge_fg = "#CBD5E1"

            refs["card"].config(bg=bg, highlightbackground=border, highlightthickness=thickness)
            refs["top"].config(bg=bg)
            refs["dot"].config(text=dot_text, fg=dot_fg, bg=bg)
            refs["title"].config(fg=title_fg, bg=bg)
            refs["badge"].config(bg=badge_bg, fg=badge_fg)
            refs["desc"].config(fg=desc_fg, bg=bg)

        self._refresh_mode_flow_preview(mode)

    def _refresh_mode_flow_preview(self, mode: str) -> None:
        spec = MODE_DEF_MAP.get(mode, MODE_DEFINITIONS[0])
        self.flow_title_lbl.config(text=f"Luồng hoạt động — {spec['title']}:", fg=spec["accent"])
        self.flow_tip_lbl.config(text=spec["tip"])

        for child in self.flow_steps_frame.winfo_children():
            child.destroy()

        steps = spec["flow"]
        for idx, (label_text, step_color) in enumerate(steps):
            chip = tk.Label(
                self.flow_steps_frame,
                text=label_text,
                font=("Segoe UI", 8, "bold"),
                fg=step_color,
                bg="#162032",
                padx=5,
                pady=2,
                highlightthickness=1,
                highlightbackground="#24324A",
            )
            chip.pack(side=tk.LEFT)
            if idx < len(steps) - 1:
                tk.Label(
                    self.flow_steps_frame,
                    text="→",
                    font=("Segoe UI", 8, "bold"),
                    fg="#64748B",
                    bg="#0F172A",
                    padx=1,
                ).pack(side=tk.LEFT)

    def _get_selected_camera_id(self) -> int:
        raw = self.camera_var.get().strip()
        try:
            return int(raw.split("-")[0].strip())
        except Exception:
            return 0

    def _get_selected_corner_key(self) -> str:
        return CORNER_LABELS_TO_KEY.get(self.corner_var.get(), "bottom_right")

    def _persist_current_settings(self) -> None:
        cfg = {
            "camera_idx": self._get_selected_camera_id(),
            "pip_corner": self._get_selected_corner_key(),
            "conf_threshold": round(float(self.threshold_var.get()), 2),
            "cooldown_sec": round(float(self.cooldown_var.get()), 1),
            "pointer_smoothing": round(float(self.smoothing_var.get()), 2),
            "pen_draw_click": bool(self.pen_draw_var.get()),
            "point_1f_mode": str(self.point_1f_mode_var.get()),
            "ppt_only_lock": bool(self.ppt_lock_var.get()),
            "bottom_deadzone": bool(self.deadzone_var.get()),
            "audio_beep": bool(self.beep_var.get()),
            "start_pip": bool(self.start_pip_var.get()),
            "start_paused": bool(self.start_paused_var.get()),
        }
        save_user_settings(cfg)

    def _on_reset_defaults(self) -> None:
        self.camera_var.set("0 - Webcam mặc định (Laptop / USB chính)")
        self.corner_var.set("Góc Dưới - Phải (Mặc định)")
        self._refresh_corner_buttons()

        self.threshold_var.set(DEFAULT_SETTINGS["conf_threshold"])
        self.cooldown_var.set(DEFAULT_SETTINGS["cooldown_sec"])
        self.smoothing_var.set(DEFAULT_SETTINGS["pointer_smoothing"])
        self.thresh_lbl.config(text="85%")
        self.cool_lbl.config(text="2.0 giây")
        self.smooth_lbl.config(text="60%")

        default_mode = (
            DEFAULT_SETTINGS["point_1f_mode"]
            if self.model_meta.get("air_ink_exists", False)
            else "laser_pen"
        )
        self.point_1f_mode_var.set(default_mode)
        self.pen_draw_var.set(DEFAULT_SETTINGS["pen_draw_click"])
        self._refresh_pen_mode_buttons()

        self.ppt_lock_var.set(DEFAULT_SETTINGS["ppt_only_lock"])
        self.deadzone_var.set(DEFAULT_SETTINGS["bottom_deadzone"])
        self.beep_var.set(DEFAULT_SETTINGS["audio_beep"])
        self.start_pip_var.set(DEFAULT_SETTINGS["start_pip"])
        self.start_paused_var.set(DEFAULT_SETTINGS["start_paused"])
        self._persist_current_settings()
        self.status_var.set("↺ Đã khôi phục toàn bộ cài đặt về mặc định chuẩn.")

    def _on_start_deployment(self) -> None:
        if not self.model_meta["exists"]:
            messagebox.showerror(
                "Thiếu mô hình AI",
                f"Không tìm thấy trọng số mô hình tại:\n{self.model_meta['path']}",
            )
            return

        self._persist_current_settings()
        cam_id = self._get_selected_camera_id()
        corner_key = self._get_selected_corner_key()
        threshold = float(self.threshold_var.get())
        cooldown = float(self.cooldown_var.get())
        smoothing = float(self.smoothing_var.get())
        point_1f_mode = str(self.point_1f_mode_var.get())
        pen_draw = point_1f_mode != "laser"
        ppt_lock = bool(self.ppt_lock_var.get())
        deadzone = bool(self.deadzone_var.get())
        beep = bool(self.beep_var.get())
        start_pip = bool(self.start_pip_var.get())
        start_paused = bool(self.start_paused_var.get())

        self.status_var.set("Đang mở Camera và khởi chạy mô hình AI điều khiển PowerPoint...")
        self.root.update_idletasks()
        self.root.withdraw()

        try:
            from step7_deployment.deploy import run_deployment_pipeline

            run_deployment_pipeline(
                camera_id=cam_id,
                conf_threshold=threshold,
                cooldown_sec=cooldown,
                start_paused=start_paused,
                pen_draw_click=pen_draw,
                start_pip=start_pip,
                pip_corner=corner_key,
                ppt_only_lock=ppt_lock,
                bottom_deadzone=deadzone,
                pointer_smoothing=smoothing,
                audio_beep=beep,
                point_1f_mode=point_1f_mode,
            )
            self.status_var.set("✔ Đã kết thúc phiên điều khiển PowerPoint an toàn. Cấu hình đã được lưu.")
        except Exception as e:
            self.status_var.set(f"✖ Lỗi khi chạy điều khiển: {e}")
            messagebox.showerror("Lỗi thực thi", f"Đã xảy ra lỗi khi mở Camera / Mô hình:\n\n{e}")
        finally:
            self.root.deiconify()
            self.root.lift()

    def _on_start_collector(self) -> None:
        cam_id = self._get_selected_camera_id()
        self.status_var.set("Đang mở công cụ thu thập mẫu Webcam cá nhân (Step 1B)...")
        self.root.update_idletasks()
        self.root.withdraw()

        try:
            from step1_data_collection.collect_webcam import run_webcam_collector

            run_webcam_collector(camera_id=cam_id)
            self.status_var.set("✔ Đã đóng công cụ thu thập mẫu Webcam.")
        except Exception as e:
            self.status_var.set(f"✖ Lỗi khi mở công cụ thu mẫu: {e}")
            messagebox.showerror("Lỗi thu thập mẫu", f"Không thể mở công cụ thu mẫu:\n\n{e}")
        finally:
            self.root.deiconify()
            self.root.lift()

    def _on_start_air_ink_collector(self) -> None:
        cam_id = self._get_selected_camera_id()
        self.status_var.set("Đang mở công cụ thu mẫu Hình vẽ Air-Ink (Step 1C)...")
        self.root.update_idletasks()
        self.root.withdraw()

        try:
            from step1_data_collection.collect_air_ink import run_air_ink_collector

            run_air_ink_collector(camera_id=cam_id)
            self.status_var.set("✔ Đã đóng công cụ thu mẫu Air-Ink.")
        except Exception as e:
            self.status_var.set(f"✖ Lỗi khi mở công cụ thu mẫu Air-Ink: {e}")
            messagebox.showerror("Lỗi thu thập Air-Ink", f"Không thể mở công cụ Step 1C:\n\n{e}")
        finally:
            self.root.deiconify()
            self.root.lift()


def launch_gui_app() -> None:
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    PowerPointHandControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui_app()
