"""
CẤU HÌNH DỰ ÁN & HẰNG SỐ TOÀN CỤC (PROJECT CONFIGURATION)
==========================================================
Chứa toàn bộ các hằng số, đường dẫn, danh sách lớp cử chỉ và ánh xạ điều khiển PowerPoint
dùng chung xuyên suốt 7 bước của dự án.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def get_bundle_root() -> Path:
    """
    Trả về đường dẫn gốc chứa tài nguyên tĩnh (như outputs/models/best_gesture_model.pt).
    - Khi chạy từ file đóng gói PyInstaller (.exe): Trỏ tới thư mục giải nén `sys._MEIPASS`.
    - Khi chạy từ mã nguồn (.py): Trỏ tới thư mục gốc dự án `d:\\SIC`.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent.parent.parent


def get_app_root() -> Path:
    """
    Trả về thư mục gốc nơi đặt ứng dụng (dùng để lưu dữ liệu phát sinh như mẫu Webcam mới).
    - Khi chạy từ file .exe: Trỏ tới thư mục chứa file `PowerPointHandControl.exe`.
    - Khi chạy từ mã nguồn .py: Trỏ tới thư mục gốc dự án `d:\\SIC`.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


# Đường dẫn thư mục cốt lõi
SRC_DIR = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = get_bundle_root()
PROJECT_ROOT = get_app_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_SEQ_DIR = DATA_DIR / "raw_sequences"
PROCESSED_DIR = DATA_DIR / "processed"
IPN_HANDS_DIR = DATA_DIR / "ipn_hands"
AIR_INK_DIR = DATA_DIR / "air_ink"
AIR_INK_RAW_DIR = AIR_INK_DIR / "raw_webcam"
AIR_INK_PROCESSED_DIR = AIR_INK_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# Ưu tiên thư mục models cạnh file .exe (nếu người dùng cập nhật model mới),
# nếu không có sẽ tự động lấy từ gói tài nguyên đóng gói sẵn trong BUNDLE_ROOT (_internal)
_EXTERNAL_MODELS_DIR = OUTPUTS_DIR / "models"
_BUNDLED_MODELS_DIR = BUNDLE_ROOT / "outputs" / "models"
MODELS_DIR = (
    _EXTERNAL_MODELS_DIR
    if (_EXTERNAL_MODELS_DIR / "best_gesture_model.pt").exists()
    else _BUNDLED_MODELS_DIR
)
FIGURES_DIR = OUTPUTS_DIR / "figures"
DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

# Cấu hình chuẩn Tensor chuỗi thời gian, tư thế tĩnh & Air-Ink (Triple-Model AI)
SEQ_LEN = 30           # 30 frames liên tiếp (~1.0 giây ở 30 FPS)
FEATURE_DIM = 66       # 63 đặc trưng hình dáng/hình học ngón tay + 3 vận tốc cổ tay
NUM_CLASSES = 8        # 8 lớp điều khiển PowerPoint chuẩn (0..7)
POSE_FEATURE_DIM = 9   # 9 kênh hình học bất biến xoay 2D trên 21 khớp (9, 21)
POSE_CLASS_NAMES: List[str] = ["other_pose", "point_1f", "point_2f"]
NUM_POSE_CLASSES = len(POSE_CLASS_NAMES)

# Cấu hình chuẩn Model 3 (AirInkResCNN - Nhận diện 9 lớp hình học vẽ không trung)
AIR_INK_IMG_SIZE = 64
AIR_INK_SHAPE_CLASSES: List[str] = [
    "line",       # 0: Gạch ngang / gạch chân / gạch dọc
    "circle",     # 1: Khoanh tròn / Elip
    "rectangle",  # 2: Đóng khung chữ nhật / vuông
    "triangle",   # 3: Tam giác
    "arrow",      # 4: Mũi tên chỉ hướng
    "checkmark",  # 5: Dấu tích đúng
    "cross",      # 6: Dấu X gạch bỏ
    "star",       # 7: Ngôi sao nhấn mạnh
    "freeform",   # 8: Nét vẽ tự do (không nắn hình)
]
AIR_INK_CLASSES: List[str] = list(AIR_INK_SHAPE_CLASSES)
NUM_AIR_INK_CLASSES = len(AIR_INK_CLASSES)
AIR_INK_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(AIR_INK_CLASSES)}
AIR_INK_SHAPE_IDS: Tuple[int, ...] = tuple(range(NUM_AIR_INK_CLASSES))

# Danh sách 8 lớp cử chỉ chuẩn
CLASS_NAMES: List[str] = [
    "idle",        # 0: Đứng yên / Không thao tác
    "swipe_left",  # 1: Quẹt sang trái -> Next Slide (Right Arrow)
    "swipe_right", # 2: Quẹt sang phải -> Previous Slide (Left Arrow)
    "point_1f",    # 3: Giơ 1 ngón trỏ -> Bút vẽ (Ctrl+P) / Tia Laser (Ctrl+L)
    "point_2f",    # 4: Giơ 2 ngón (Trỏ + Giữa) -> Xoá mực (Ctrl+E)
    "zoom_in",     # 5: Chụm rồi bung ngón -> Phóng to Slide (Ctrl + '=')
    "zoom_out",    # 6: Xòe rồi chụm ngón -> Thu nhỏ Slide (Ctrl + '-')
    "click",       # 7: Chớp mở bàn tay 2 lần (OP2) -> Click chuột trái (Left Click)
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS_NAME: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

# Nhóm cử chỉ rời rạc (chạy 1 lần kích hoạt phím/chuột + Cooldown)
DISCRETE_CLASSES: Tuple[int, ...] = (1, 2, 5, 6, 7)

# Ánh xạ hành động PowerPoint & màu sắc BGR phục vụ hiển thị trên HUD
ACTION_DISPLAY: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    0: ("IDLE (Khong thao tac)", (180, 180, 180)),
    1: ("NEXT SLIDE (Right Arrow)", (50, 220, 50)),
    2: ("PREV SLIDE (Left Arrow)", (50, 200, 255)),
    3: ("SPATIAL PEN / LASER (1 Ngon)", (0, 255, 180)),
    4: ("SPATIAL ERASER (Ctrl+E + Hold Click)", (240, 120, 255)),
    5: ("ZOOM IN (Ctrl + '=')", (0, 215, 255)),
    6: ("ZOOM OUT (Ctrl + '-')", (255, 160, 50)),
    7: ("LEFT CLICK (Click Chuot Trai)", (0, 235, 255)),
}

# Hướng dẫn thu thập mẫu Webcam
GESTURE_GUIDES: Dict[int, str] = {
    0: "IDLE: De tay mo tu nhien hoac nam ho giu yen truoc camera",
    1: "SWIPE LEFT (Next): Dat tay ben PHAI -> Khi bat dau ghi, gat ngang sang TRAI",
    2: "SWIPE RIGHT (Prev): Dat tay ben TRAI -> Khi bat dau ghi, gat ngang sang PHAI",
    3: "POINT 1F (But ve): Giu 1 NGON TRO (thu ca goc nghieng, gan/xa va chia thang camera)",
    4: "POINT 2F (Xoa muc): Giu 2 NGON TRO+GIUA (thu ca goc nghieng, gan/xa va chia thang camera)",
    5: "ZOOM IN (Phong to): Khi chuan bi CHUM ngon -> Khi ghi BUNG XOE RONG ra",
    6: "ZOOM OUT (Thu nho): Khi chuan bi XOE RONG -> Khi ghi KHEP CHUM SAT lai",
    7: "CLICK (Chuot trai - OP2): Giu yen co tay -> CHOP MO XOE BAN TAY 2 LAN LIEN TIEP (Mo-Khep-Mo)",
}

# 21 đoạn nối khung xương bàn tay chuẩn Google MediaPipe
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20), # Ngón út
    (5, 9), (9, 13), (13, 17), (0, 17),    # Lòng bàn tay
]

# URL & đường dẫn file task MediaPipe Hands (Python 3.14)
HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_TASK_PATH = (
    _BUNDLED_MODELS_DIR / "hand_landmarker.task"
    if (_BUNDLED_MODELS_DIR / "hand_landmarker.task").exists()
    else MODELS_DIR / "hand_landmarker.task"
)

