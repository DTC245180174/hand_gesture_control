"""
KỊCH BẢN ĐÓNG GÓI ỨNG DỤNG DESKTOP TỰ ĐỘNG (APPLICATION BUILDER)
================================================================
Đóng gói toàn bộ hệ thống PowerPoint Hand Control thành ứng dụng Windows độc lập (.exe):
1. Tự động thiết kế & xuất biểu tượng ứng dụng `assets/app_icon.ico` (chuẩn 256x256).
2. Kiểm tra và nhúng trực tiếp trọng số đã huấn luyện `outputs/models/best_gesture_model.pt`
   cùng `outputs/models/hand_landmarker.task` vào bên trong gói ứng dụng.
3. Chạy PyInstaller để xuất bản `dist/PowerPointHandControl/PowerPointHandControl.exe`.
4. Tạo công cụ `Tao_Shortcut_Ra_Desktop.vbs` để ghim biểu tượng ra màn hình chính Windows.

Cách dùng:
    python build_app.py                # Đóng gói chuẩn --onedir (Mở ứng dụng tức thì < 1.5s)
    python build_app.py --onefile      # Đóng gói thành 1 file .exe duy nhất
    python build_app.py --zip          # Đóng gói kèm nén thành file .zip Portable để gửi máy khác
"""

from __future__ import annotations
import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def generate_app_icon(icon_path: Path) -> None:
    """
    Tự động vẽ biểu tượng ứng dụng (Slide PowerPoint + Khung xương bàn tay AI 21 khớp)
    và đóng gói thành định dạng chuẩn Windows `.ico` (256x256 RGBA PNG-packed).
    """
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    size = 256
    img = np.zeros((size, size, 4), dtype=np.uint8)

    # Nền bo góc màu xanh đen hiện đại (#121826)
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(bgr, (12, 12), (size - 12, size - 12), (38, 24, 18), -1, cv2.LINE_AA)
    cv2.rectangle(bgr, (12, 12), (size - 12, size - 12), (255, 200, 0), 6, cv2.LINE_AA)

    # Khung Slide PowerPoint màu cam đỏ ở góc trên
    cv2.rectangle(bgr, (36, 40), (220, 155), (45, 75, 210), -1, cv2.LINE_AA)
    cv2.rectangle(bgr, (36, 40), (220, 155), (120, 180, 255), 3, cv2.LINE_AA)
    cv2.putText(bgr, "PPT", (52, 112), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)

    # Vẽ các đốt khung xương bàn tay AI toả sáng (Cyan & Vàng)
    wrist = (145, 224)
    fingers = [
        [(145, 224), (105, 200), (82, 175)],               # Ngón cái
        [(145, 224), (125, 165), (115, 108), (108, 65)],   # Ngón trỏ (Point 1F)
        [(145, 224), (148, 162), (150, 112), (152, 72)],   # Ngón giữa
        [(145, 224), (170, 170), (178, 132)],              # Ngón áp út
        [(145, 224), (190, 182), (202, 152)],              # Ngón út
    ]
    for chain in fingers:
        for i in range(len(chain) - 1):
            cv2.line(bgr, chain[i], chain[i + 1], (255, 230, 0), 5, cv2.LINE_AA)
        for idx, pt in enumerate(chain):
            r = 8 if idx == len(chain) - 1 else 5
            color = (0, 255, 180) if idx == len(chain) - 1 else (255, 255, 255)
            cv2.circle(bgr, pt, r, color, -1, cv2.LINE_AA)
    cv2.circle(bgr, wrist, 9, (0, 215, 255), -1, cv2.LINE_AA)

    # Tạo kênh Alpha bo tròn nhẹ ở viền
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (12, 12), (size - 12, size - 12), 255, -1)
    img[:, :, :3] = bgr
    img[:, :, 3] = mask

    ok, png_buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Không thể mã hoá dữ liệu PNG cho icon.")
    png_bytes = png_buf.tobytes()

    # Đóng gói chuẩn cấu trúc nhị phân Windows .ICO (ICONDIR 6 bytes + ICONDIRENTRY 16 bytes + PNG data)
    ico_header = struct.pack("<HHH", 0, 1, 1)
    ico_entry = struct.pack(
        "<BBBBHHII",
        0,                  # 0 = 256px width
        0,                  # 0 = 256px height
        0,                  # Color palette
        0,                  # Reserved
        1,                  # Color planes
        32,                 # Bits per pixel (32-bit RGBA)
        len(png_bytes),     # Size of image data
        6 + 16,             # Offset of image data
    )
    with open(icon_path, "wb") as f:
        f.write(ico_header + ico_entry + png_bytes)
    print(f"[ICON] Đã tạo biểu tượng ứng dụng -> {icon_path}")


def ensure_pyinstaller_installed() -> None:
    """Kiểm tra PyInstaller, nếu chưa có sẽ tự động cài đặt qua pip."""
    try:
        import PyInstaller  # noqa: F401
        print("[CHECK] Đã tìm thấy PyInstaller trong môi trường Python hiện tại.")
    except ImportError:
        print("[INSTALL] Chưa có PyInstaller -> Đang tự động cài đặt `pip install pyinstaller`...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def create_desktop_shortcut_script(app_dir: Path, exe_name: str = "PowerPointHandControl.exe") -> None:
    """Tạo file VBScript gọn nhẹ giúp người dùng tạo Shortcut ra Desktop chỉ bằng 1 cú nhấp đúp."""
    vbs_path = app_dir / "Tao_Shortcut_Ra_Desktop.vbs"
    vbs_content = f'''Set oWS = WScript.CreateObject("WScript.Shell")
sDesktop = oWS.SpecialFolders("Desktop")
sAppDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sLinkFile = sDesktop & "\\PowerPoint Hand Control.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = sAppDir & "\\{exe_name}"
oLink.WorkingDirectory = sAppDir
oLink.Description = "Dieu khien PowerPoint bang cu chi tay dong AI"
oLink.IconLocation = sAppDir & "\\{exe_name},0"
oLink.Save
MsgBox "Da tao bieu tuong 'PowerPoint Hand Control' ngoai man hinh Desktop thanh cong!", 64, "PowerPoint Hand Control"
'''
    vbs_path.write_text(vbs_content, encoding="utf-8")
    print(f"[SHORTCUT] Đã tạo tiện ích tạo Shortcut -> {vbs_path}")


def build_application(onefile: bool = False, make_zip: bool = False, console: bool = False) -> None:
    print("=" * 84)
    print("ĐÓNG GÓI ỨNG DỤNG DESKTOP: POWERPOINT HAND CONTROL (.EXE)")
    print("=" * 84)

    # 1. Kiểm tra trọng số mô hình bắt buộc & trọng số bổ sung (PoseNet + Air-Ink Model 3)
    best_model_pt = MODELS_DIR / "best_gesture_model.pt"
    best_pose_pt = MODELS_DIR / "best_pose_model.pt"
    best_air_ink_pt = MODELS_DIR / "best_air_ink_model.pt"
    mp_task_file = MODELS_DIR / "hand_landmarker.task"
    if not best_model_pt.exists():
        raise FileNotFoundError(
            f"Không tìm thấy trọng số mô hình tại {best_model_pt}. Hãy chạy huấn luyện Bước 5 trước."
        )
    if not mp_task_file.exists():
        # Tự động tải file hand_landmarker.task nếu chưa có
        sys.path.insert(0, str(SRC_DIR))
        from common.mediapipe_tracker import MediaPipeHandTracker
        MediaPipeHandTracker._ensure_task_model_downloaded()

    print(f"  - Model chính      : {best_model_pt} ({best_model_pt.stat().st_size / (1024*1024):.2f} MB)")
    if best_pose_pt.exists():
        print(f"  - PoseNet (M1)     : {best_pose_pt} ({best_pose_pt.stat().st_size / (1024*1024):.2f} MB)")
    if best_air_ink_pt.exists():
        print(f"  - Air-Ink (M3)     : {best_air_ink_pt} ({best_air_ink_pt.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - MediaPipe Task   : {mp_task_file} ({mp_task_file.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Tạo icon .ico
    icon_path = ASSETS_DIR / "app_icon.ico"
    generate_app_icon(icon_path)

    # 3. Đảm bảo đã cài PyInstaller
    ensure_pyinstaller_installed()

    # 4. Xây dựng danh sách tham số PyInstaller
    app_name = "PowerPointHandControl"
    sep = ";" if sys.platform.startswith("win") else ":"

    data_args = [
        f"--add-data={best_model_pt}{sep}outputs/models",
        f"--add-data={mp_task_file}{sep}outputs/models",
        f"--add-data={icon_path}{sep}assets",
    ]
    if best_pose_pt.exists():
        data_args.append(f"--add-data={best_pose_pt}{sep}outputs/models")
    if best_air_ink_pt.exists():
        data_args.append(f"--add-data={best_air_ink_pt}{sep}outputs/models")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        f"--name={app_name}",
        "--onedir" if not onefile else "--onefile",
        "--console" if console else "--windowed",
        f"--icon={icon_path}",
        f"--paths={SRC_DIR}",
        # Nhúng trọng số AI & MediaPipe task vào đúng cấu trúc thư mục outputs/models
        *data_args,
        # Nhúng tài nguyên nội bộ của MediaPipe
        "--collect-data=mediapipe",
        # Đảm bảo nạp đầy đủ các module nội bộ của dự án
        "--hidden-import=app_gui",
        "--hidden-import=common.config",
        "--hidden-import=common.mediapipe_tracker",
        "--hidden-import=common.win32_input",
        "--hidden-import=step1_data_collection.collect_webcam",
        "--hidden-import=step1_data_collection.collect_air_ink",
        "--hidden-import=step2_clean_diagnose.feature_extractor",
        "--hidden-import=step2_clean_diagnose.ink_rasterizer",
        "--hidden-import=step4_model_selection.models",
        "--hidden-import=step7_deployment.smart_shape",
        "--hidden-import=step7_deployment.controller",
        "--hidden-import=step7_deployment.hud",
        "--hidden-import=step7_deployment.deploy",
        "--hidden-import=win32com.client",
    ]

    # Loại bỏ các thư viện chỉ dùng lúc vẽ biểu đồ / huấn luyện để giảm mạnh dung lượng .exe
    excluded_packages = [
        "matplotlib",
        "seaborn",
        "sklearn",
        "scikit-learn",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tensorboard",
        "torchvision",
        "torchaudio",
    ]
    for pkg in excluded_packages:
        pyinstaller_cmd.append(f"--exclude-module={pkg}")

    pyinstaller_cmd.append(str(PROJECT_ROOT / "main.py"))

    print("\n[PYINSTALLER] Đang tiến hành biên dịch và đóng gói ứng dụng...")
    subprocess.check_call(pyinstaller_cmd, cwd=str(PROJECT_ROOT))

    if onefile:
        exe_output = DIST_DIR / f"{app_name}.exe"
        print("\n" + "=" * 84)
        print("HOÀN TẤT ĐÓNG GÓI CHẾ ĐỘ 1 FILE (--ONEFILE)!")
        print(f"  -> File thực thi: {exe_output}")
        print("=" * 84)
    else:
        bundle_dir = DIST_DIR / app_name
        exe_output = bundle_dir / f"{app_name}.exe"
        create_desktop_shortcut_script(bundle_dir, f"{app_name}.exe")

        if make_zip:
            zip_base = DIST_DIR / f"{app_name}_Portable_Win64"
            print(f"[ZIP] Đang nén gói Portable -> {zip_base}.zip ...")
            shutil.make_archive(str(zip_base), "zip", root_dir=str(DIST_DIR), base_dir=app_name)
            print(f"[ZIP] Đã tạo xong gói phân phối: {zip_base}.zip")

        print("\n" + "=" * 84)
        print("HOÀN TẤT ĐÓNG GÓI ỨNG DỤNG DESKTOP THÀNH CÔNG!")
        print(f"  1. File chạy chính     : {exe_output}")
        print(f"  2. Tạo Shortcut nhanh  : {bundle_dir / 'Tao_Shortcut_Ra_Desktop.vbs'}")
        print("=" * 84)


def main() -> None:
    parser = argparse.ArgumentParser(description="Đóng gói PowerPoint Hand Control thành ứng dụng .exe")
    parser.add_argument("--onefile", action="store_true", help="Đóng gói thành 1 file .exe duy nhất")
    parser.add_argument("--zip", action="store_true", help="Nén thư mục ứng dụng thành file .zip Portable")
    parser.add_argument("--console", action="store_true", help="Giữ cửa sổ dòng lệnh để xem log trực tiếp")
    args = parser.parse_args()

    build_application(onefile=args.onefile, make_zip=args.zip, console=args.console)


if __name__ == "__main__":
    main()
