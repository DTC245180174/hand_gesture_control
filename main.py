"""
POWERPOINT HAND CONTROL - ĐIỂM CHẠY TOÀN DỰ ÁN (PROJECT ENTRY POINT)
=====================================================================
Điều khiển toàn bộ quy trình 7 bước hoặc mở Giao diện Ứng dụng Desktop (GUI):

Cách dùng:
    python main.py --gui           # Mở Giao diện Bảng điều khiển Desktop (GUI Launcher)
    python main.py --step 1a       # Cắt phân đoạn dữ liệu từ IPN HandS
    python main.py --step 1b       # Thu thập mẫu Webcam cá nhân
    python main.py --step 2        # Làm sạch, đồng bộ T=30 & trích xuất 66D tensor
    python main.py --step 3        # Trực quan hoá dữ liệu (sinh 4 biểu đồ EDA)
    python main.py --step 3-replay # Mở trình phát video mô phỏng cử chỉ gốc IPN HandS
    python main.py --step 4        # Khởi tạo và benchmark 3 kiến trúc mô hình
    python main.py --step 5        # Huấn luyện mô hình Dual-Model (Model 1 + Model 2) từ đầu
    python main.py --step 5-ink    # Huấn luyện Model 3 (AirInkResCNN - 9 lớp hình học) từ đầu
    python main.py --step 6        # Đánh giá độc lập Dual-Model trên tập Test & Safety Gate Audit
    python main.py --step 6-ink    # Đánh giá độc lập Model 3 (9 lớp hình học Air-Ink)
    python main.py --step 7        # Triển khai điều khiển PowerPoint thời gian thực

Tuỳ chọn triển khai nhanh:
    python main.py                 # Mặc định khởi chạy Bước 7 (hoặc mở GUI khi chạy từ file .exe)
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Đảm bảo thư mục src nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _ensure_safe_stdio_for_windowed_exe() -> None:
    """
    Khi đóng gói PyInstaller với `--noconsole` (`console=False`), `sys.stdout` và `sys.stderr`
    có thể là `None`. Hàm này chuyển hướng an toàn vào file `app_runtime.log` để lệnh `print()`
    không bao giờ gây lỗi.
    """
    if sys.stdout is None or sys.stderr is None:
        try:
            app_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else PROJECT_ROOT
            log_file = open(app_dir / "app_runtime.log", "a", encoding="utf-8", buffering=1)
            if sys.stdout is None:
                sys.stdout = log_file
            if sys.stderr is None:
                sys.stderr = log_file
        except Exception:
            devnull = open(os.devnull, "w", encoding="utf-8")
            if sys.stdout is None:
                sys.stdout = devnull
            if sys.stderr is None:
                sys.stderr = devnull


def main() -> None:
    _ensure_safe_stdio_for_windowed_exe()

    # Nếu người dùng nhấp đúp mở trực tiếp file .exe (không truyền tham số dòng lệnh) -> Mở GUI Launcher
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        from app_gui import launch_gui_app
        launch_gui_app()
        return

    parser = argparse.ArgumentParser(
        description="PowerPoint Hand Control - Hệ thống nhận diện cử chỉ động 7 bước",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        default="7",
        choices=["gui", "1a", "1b", "1c", "2", "2-ink", "3", "3-replay", "3-ink", "4", "5", "5-ink", "6", "6-ink", "7"],
        help="Chọn bước cần thực hiện trong quy trình dự án (Mặc định: 7 - Triển khai thực tế)",
    )
    parser.add_argument("--gui", action="store_true", help="Mở Giao diện Bảng điều khiển Desktop (GUI Launcher)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (dùng cho 1b, 1c và 7)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Ngưỡng xác suất an toàn (dùng cho 7, mặc định: 0.85)")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Thời gian Cooldown sau mỗi lệnh (giây, mặc định: 2.0s)")
    parser.add_argument("--paused", action="store_true", help="Khởi động Bước 7 ở chế độ PAUSED")
    parser.add_argument(
        "--point-1f-mode",
        type=str,
        default="laser_ai",
        choices=["laser_ai", "laser_pen", "ai_shapes", "pen", "laser"],
        help="Chế độ khi giơ 1 ngón trỏ (Bước 7): laser_ai | laser_pen | ai_shapes | pen | laser",
    )
    parser.add_argument("--arch", type=str, default="hybrid", help="Kiến trúc mạng (dùng cho 5: hybrid, cnn1d, bilstm)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs huấn luyện tối đa (5: 70, 5-ink: 14)")
    parser.add_argument("--batch-size", type=int, default=None, help="Kích thước batch (5: 32, 5-ink: 160)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (5: 1e-3, 5-ink: 2.0e-3)")
    parser.add_argument("--replay", action="store_true", help="Chạy nhanh trình phát video mô phỏng cử chỉ gốc")

    args = parser.parse_args()

    if args.gui or args.step.lower() == "gui":
        from app_gui import launch_gui_app
        launch_gui_app()
        return

    if args.replay or args.step == "3-replay":
        from step3_visualization.view_ipn_gestures import run_gesture_replay
        run_gesture_replay()
        return

    step = args.step.lower()

    if step == "1a":
        from step1_data_collection.prepare_ipn_hands import slice_ipn_hands_dataset
        slice_ipn_hands_dataset()
    elif step == "1b":
        from step1_data_collection.collect_webcam import run_webcam_collector
        run_webcam_collector(camera_id=args.camera)
    elif step == "1c":
        from step1_data_collection.collect_air_ink import run_air_ink_collector
        run_air_ink_collector(camera_id=args.camera)
    elif step == "2":
        from step2_clean_diagnose.clean_diagnose_pipeline import run_clean_and_diagnose_pipeline
        run_clean_and_diagnose_pipeline()
    elif step == "2-ink":
        from step1_data_collection.prepare_air_ink_data import prepare_air_ink_datasets
        prepare_air_ink_datasets()
    elif step == "3":
        from step3_visualization.plot_diagnostics import run_visualizations
        run_visualizations()
    elif step == "3-ink":
        from step3_visualization.visualize_air_ink import run_air_ink_visualizations
        run_air_ink_visualizations()
    elif step == "4":
        from step4_model_selection.models import benchmark_models
        benchmark_models()
    elif step == "5":
        from step5_training.train import run_training_pipeline
        run_training_pipeline(
            arch=args.arch,
            epochs=70 if args.epochs is None else args.epochs,
            batch_size=32 if args.batch_size is None else args.batch_size,
            lr=1e-3 if args.lr is None else args.lr,
        )
    elif step == "5-ink":
        from step5_training.train_air_ink import train_air_ink_model
        train_air_ink_model(
            epochs=14 if args.epochs is None else args.epochs,
            batch_size=160 if args.batch_size is None else args.batch_size,
            lr=2.0e-3 if args.lr is None else args.lr,
        )
    elif step == "6":
        from step6_evaluation.evaluate import run_evaluation_pipeline
        run_evaluation_pipeline()
    elif step == "6-ink":
        from step6_evaluation.evaluate_air_ink import run_air_ink_evaluation_pipeline
        run_air_ink_evaluation_pipeline()
    elif step == "7":
        from step7_deployment.deploy import run_deployment_pipeline
        run_deployment_pipeline(
            camera_id=args.camera,
            conf_threshold=args.threshold,
            cooldown_sec=args.cooldown,
            start_paused=args.paused,
            pen_draw_click=(args.point_1f_mode != "laser"),
            point_1f_mode=args.point_1f_mode,
        )
    else:
        print(f"[!] Bước không xác định: {args.step}")


if __name__ == "__main__":
    main()
