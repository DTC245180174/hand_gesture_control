"""
MODULE ĐIỀU KHIỂN PHẦN CỨNG WIN32 & LỌC TỌA ĐỘ CON TRỎ (WIN32 INPUT & FILTER)
=============================================================================
Cung cấp các hàm Win32 API cấp thấp độ trễ 0.005ms:
- Gửi phím phần cứng (Ctrl+P, Ctrl+E, F5, v.v.)
- Di chuyển & kéo rê chuột phần cứng (mouse_event MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
- Bộ lọc OneEuroFilter2D chống rung tay khi vẽ trên PowerPoint
- Điều khiển cửa sổ PiP (Picture-in-Picture) luôn nổi (TopMost) và xuyên chuột (Click-Through)
"""

from __future__ import annotations
import ctypes
import ctypes.wintypes as wintypes
import math
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
except Exception:
    pyautogui = None

_USER32 = ctypes.windll.user32 if os.name == "nt" else None
if _USER32 is not None:
    try:
        _USER32.SetProcessDPIAware()
    except Exception:
        pass
    try:
        # Khai báo chính xác kiểu con trỏ 64-bit cho HWND trên Windows x64 (tránh lỗi HWND_TOPMOST = -1 bị cắt 32-bit)
        _USER32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        _USER32.FindWindowW.restype = wintypes.HWND
        _USER32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        _USER32.GetAncestor.restype = wintypes.HWND
        _USER32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        _USER32.GetWindow.restype = wintypes.HWND
        _USER32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        _USER32.GetWindowLongW.restype = ctypes.c_long
        _USER32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
        _USER32.SetWindowLongW.restype = ctypes.c_long
        _USER32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.COLORREF, ctypes.c_ubyte, wintypes.DWORD]
        _USER32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        _USER32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        _USER32.SetWindowPos.restype = wintypes.BOOL
        _USER32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        _USER32.GetWindowRect.restype = wintypes.BOOL
        _USER32.IsIconic.argtypes = [wintypes.HWND]
        _USER32.IsIconic.restype = wintypes.BOOL
        _USER32.IsWindowVisible.argtypes = [wintypes.HWND]
        _USER32.IsWindowVisible.restype = wintypes.BOOL
        _USER32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        _USER32.ShowWindow.restype = wintypes.BOOL
        _USER32.SetForegroundWindow.argtypes = [wintypes.HWND]
        _USER32.SetForegroundWindow.restype = wintypes.BOOL
        _USER32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        _USER32.GetWindowTextLengthW.restype = ctypes.c_int
        _USER32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        _USER32.GetWindowTextW.restype = ctypes.c_int
        _USER32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        _USER32.GetClassNameW.restype = ctypes.c_int
        _USER32.GetForegroundWindow.argtypes = []
        _USER32.GetForegroundWindow.restype = wintypes.HWND
    except Exception:
        pass

# Hằng số sự kiện chuột và phím Win32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_PRIOR = 0x21  # Page Up (Previous Slide)
VK_NEXT = 0x22   # Page Down (Next Slide)
VK_LEFT = 0x25
VK_RIGHT = 0x27
GWL_EXSTYLE = -20
GA_ROOT = 2
GW_HWNDPREV = 3
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
LWA_ALPHA = 0x00000002
HWND_TOPMOST = wintypes.HWND(-1)
HWND_NOTOPMOST = wintypes.HWND(-2)
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040


def win32_send_ctrl_key(char_key: str) -> None:
    """Gửi tổ hợp phím phần cứng `Ctrl + <char_key>` qua Win32 keybd_event tới PowerPoint."""
    ensure_powerpoint_foreground()
    if _USER32 is not None:
        vk = ord(char_key.upper())
        scan_ctrl = _USER32.MapVirtualKeyW(VK_CONTROL, 0)
        scan_key = _USER32.MapVirtualKeyW(vk, 0)
        _USER32.keybd_event(VK_CONTROL, scan_ctrl, 0, 0)
        _USER32.keybd_event(vk, scan_key, 0, 0)
        time.sleep(0.015)
        _USER32.keybd_event(vk, scan_key, KEYEVENTF_KEYUP, 0)
        _USER32.keybd_event(VK_CONTROL, scan_ctrl, KEYEVENTF_KEYUP, 0)
    elif pyautogui is not None:
        pyautogui.hotkey("ctrl", char_key.lower())


def win32_send_ppt_slide_key(direction: str = "next") -> None:
    """
    Đảm bảo cửa sổ PowerPoint đang có Focus và gửi phím phần cứng chuyển Slide:
    - direction == "next": Gửi VK_NEXT (PageDown - 0x22), hoạt động cả khi đang bật Bút vẽ (Ctrl+P) / Laser (Ctrl+L).
    - direction == "prev": Gửi VK_PRIOR (PageUp - 0x21).
    """
    ensure_powerpoint_foreground()
    vk = VK_NEXT if direction == "next" else VK_PRIOR
    if _USER32 is not None:
        scan = _USER32.MapVirtualKeyW(vk, 0)
        _USER32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY, 0)
        time.sleep(0.018)
        _USER32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
    elif pyautogui is not None:
        pyautogui.press("pagedown" if direction == "next" else "pageup")


def to_win32_abs(x: int, y: int, screen_w: int, screen_h: int) -> Tuple[int, int]:
    """Chuyển toạ độ pixel màn hình sang hệ toạ độ tuyệt đối 0..65535 của Win32 Mouse Input."""
    abs_x = int(round(np.clip(x, 0, screen_w - 1) * 65535.0 / max(1, screen_w - 1)))
    abs_y = int(round(np.clip(y, 0, screen_h - 1) * 65535.0 / max(1, screen_h - 1)))
    return abs_x, abs_y


def win32_move_cursor(x: int, y: int, screen_w: int = 1920, screen_h: int = 1080) -> None:
    """Di chuyển con trỏ chuột bằng sự kiện phần cứng MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE."""
    if _USER32 is not None:
        abs_x, abs_y = to_win32_abs(int(x), int(y), screen_w, screen_h)
        _USER32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
    elif pyautogui is not None:
        pyautogui.moveTo(int(x), int(y), _pause=False)


def win32_drag_cursor(
    prev_x: int,
    prev_y: int,
    cur_x: int,
    cur_y: int,
    screen_w: int = 1920,
    screen_h: int = 1080,
    steps: int = 10,
    step_delay_sec: float = 0.0,
) -> None:
    """Nội suy các điểm trung gian và gửi chuỗi sự kiện MOVE | ABSOLUTE để vẽ nét liền mạch."""
    if _USER32 is not None:
        steps = max(1, int(steps))
        for s in range(1, steps + 1):
            alpha = s / float(steps)
            ix = int(round(prev_x + alpha * (cur_x - prev_x)))
            iy = int(round(prev_y + alpha * (cur_y - prev_y)))
            abs_x, abs_y = to_win32_abs(ix, iy, screen_w, screen_h)
            _USER32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
            if step_delay_sec > 0.0 and s < steps:
                time.sleep(step_delay_sec)
    elif pyautogui is not None:
        pyautogui.moveTo(int(cur_x), int(cur_y), _pause=False)


def win32_mouse_down(x: int, y: int, screen_w: int = 1920, screen_h: int = 1080) -> None:
    """Nhấn giữ chuột trái tại (x, y) bằng Win32 mouse_event."""
    if _USER32 is not None:
        abs_x, abs_y = to_win32_abs(int(x), int(y), screen_w, screen_h)
        _USER32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        time.sleep(0.01)
        _USER32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        time.sleep(0.01)
    elif pyautogui is not None:
        pyautogui.moveTo(int(x), int(y), _pause=False)
        pyautogui.mouseDown(_pause=False)


def win32_mouse_up() -> None:
    """Nhả chuột trái tức thì bằng Win32 mouse_event."""
    if _USER32 is not None:
        _USER32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif pyautogui is not None:
        pyautogui.mouseUp(_pause=False)


def win32_left_click(x: int, y: int, screen_w: int = 1920, screen_h: int = 1080) -> None:
    """Click chuột trái đơn (Left Click) tại toạ độ (x, y) bằng sự kiện phần cứng Win32."""
    if _USER32 is not None:
        abs_x, abs_y = to_win32_abs(int(x), int(y), screen_w, screen_h)
        _USER32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        time.sleep(0.01)
        _USER32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        time.sleep(0.015)
        _USER32.mouse_event(MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
    elif pyautogui is not None:
        pyautogui.click(int(x), int(y), _pause=False)


def get_foreground_window_title() -> str:
    """Lấy tiêu đề cửa sổ đang Active để kiểm tra PowerPoint Slide Show."""
    if _USER32 is None:
        return ""
    hwnd = _USER32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = _USER32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    _USER32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_window_class_name(hwnd) -> str:
    """Lấy Win32 Window Class Name của cửa sổ theo HWND (vd: 'screenClass' cho Slide Show F5)."""
    if _USER32 is None or not hwnd:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(256)
        n = _USER32.GetClassNameW(wintypes.HWND(hwnd), buf, 256)
        return buf.value[:n] if n > 0 else ""
    except Exception:
        return ""


def is_powerpoint_foreground(own_window_prefix: str = "PowerPoint Hand Control") -> bool:
    """
    Kiểm tra xem cửa sổ đang Active của Windows có phải là Microsoft PowerPoint
    (cửa sổ soạn thảo hoặc màn hình trình chiếu Slide Show F5 / screenClass) hay không.
    """
    if _USER32 is not None:
        fg = _USER32.GetForegroundWindow()
        if fg and get_window_class_name(fg) == "screenClass":
            return True
    title = get_foreground_window_title()
    if not title:
        return False
    if own_window_prefix and title.startswith(own_window_prefix):
        return False
    t_low = title.lower()
    ppt_keywords = ("powerpoint", "slide show", "trình chiếu", ".pptx", ".ppsx", ".ppt")
    return any(k in t_low for k in ppt_keywords)


def find_powerpoint_window(own_window_prefix: str = "PowerPoint Hand Control") -> int:
    """
    Tìm HWND của cửa sổ Microsoft PowerPoint đang mở trên màn hình (ưu tiên cửa sổ Slide Show).
    Trả về 0 nếu không tìm thấy.
    """
    if _USER32 is None:
        return 0
    slideshow_hwnds = []
    editor_hwnds = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _enum_cb(hwnd: int, _lparam: int) -> bool:
        h_ptr = wintypes.HWND(hwnd)
        if not _USER32.IsWindowVisible(h_ptr):
            return True
        cls_name = get_window_class_name(h_ptr)
        length = _USER32.GetWindowTextLengthW(h_ptr)
        title = ""
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            _USER32.GetWindowTextW(h_ptr, buf, length + 1)
            title = buf.value
        if own_window_prefix and title.startswith(own_window_prefix):
            return True
        if cls_name == "screenClass":
            slideshow_hwnds.append(int(hwnd))
            return True
        if not title:
            return True
        t_low = title.lower()
        if "slide show" in t_low or "trình chiếu" in t_low:
            slideshow_hwnds.append(int(hwnd))
        elif any(k in t_low for k in ("powerpoint", ".pptx", ".ppsx", ".ppt")):
            editor_hwnds.append(int(hwnd))
        return True

    try:
        _USER32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
    except Exception:
        return 0

    if slideshow_hwnds:
        return slideshow_hwnds[0]
    if editor_hwnds:
        return editor_hwnds[0]
    return 0


def is_powerpoint_slideshow_active(own_window_prefix: str = "PowerPoint Hand Control") -> bool:
    """Kiểm tra PowerPoint Slide Show (F5 / class 'screenClass') có đang mở hay không."""
    if _USER32 is None:
        return False
    fg = _USER32.GetForegroundWindow()
    if fg and get_window_class_name(fg) == "screenClass":
        return True
    ppt_hwnd = find_powerpoint_window(own_window_prefix=own_window_prefix)
    if not ppt_hwnd:
        return False
    h_ptr = wintypes.HWND(ppt_hwnd)
    if get_window_class_name(h_ptr) == "screenClass":
        return True
    length = _USER32.GetWindowTextLengthW(h_ptr)
    if length > 0:
        buf = ctypes.create_unicode_buffer(length + 1)
        _USER32.GetWindowTextW(h_ptr, buf, length + 1)
        t_low = buf.value.lower()
        return ("slide show" in t_low) or ("trình chiếu" in t_low)
    return False


def ensure_powerpoint_foreground(own_window_prefix: str = "PowerPoint Hand Control") -> bool:
    """
    Nếu PowerPoint đang mở trên máy nhưng cửa sổ Camera OpenCV lỡ chiếm mất Focus,
    tự động trả Focus về cho cửa sổ PowerPoint để phím chuyển Slide / Bút vẽ luôn ăn 100%.
    """
    if is_powerpoint_foreground(own_window_prefix=own_window_prefix):
        return True
    ppt_hwnd = find_powerpoint_window(own_window_prefix=own_window_prefix)
    if ppt_hwnd != 0 and _USER32 is not None:
        try:
            if _USER32.IsIconic(ppt_hwnd):
                _USER32.ShowWindow(ppt_hwnd, 9)  # SW_RESTORE
            _USER32.SetForegroundWindow(ppt_hwnd)
            time.sleep(0.015)
            return True
        except Exception:
            return False
    return False


def play_action_beep(cls_id: int) -> None:
    """
    Phát âm báo 'Tít' ngắn ở luồng nền (không làm trễ vòng lặp Camera)
    khi kích hoạt lệnh chuyển Slide hoặc Zoom.
    """
    if os.name != "nt":
        return
    import threading

    def _beep_worker() -> None:
        try:
            import winsound
            if cls_id == 7:
                winsound.Beep(1280, 55)
            else:
                # Next Slide (1) / Zoom In (5): âm cao 1050Hz; Prev Slide (2) / Zoom Out (6): âm trầm 780Hz
                freq = 1050 if cls_id in (1, 5) else 780
                winsound.Beep(freq, 65)
        except Exception:
            pass

    threading.Thread(target=_beep_worker, daemon=True).start()


def play_dwell_beep(kind: str = "pen_on") -> None:
    """Âm báo siêu ngắn (~35ms) ở luồng nền báo hiệu chuyển đổi Laser <-> Bút vẽ."""
    if os.name != "nt":
        return

    def _beep_worker() -> None:
        try:
            import winsound
            if kind == "pen_on":
                winsound.Beep(1320, 35)
            elif kind == "laser_on":
                winsound.Beep(880, 30)
            else:
                winsound.Beep(1040, 35)
        except Exception:
            pass

    threading.Thread(target=_beep_worker, daemon=True).start()


def _get_pip_rect(screen_w: int, screen_h: int, corner: str = "bottom_right") -> Tuple[int, int, int, int]:
    """Tính toạ độ và kích thước chuẩn 480x270 của cửa sổ PiP theo góc màn hình."""
    pip_w, pip_h = 480, 270
    margin_x, margin_y = 20, 60
    c_key = (corner or "bottom_right").lower().strip()
    if c_key == "bottom_left":
        pos_x = margin_x
        pos_y = max(10, screen_h - pip_h - margin_y)
    elif c_key == "top_right":
        pos_x = max(10, screen_w - pip_w - margin_x)
        pos_y = 24
    elif c_key == "top_left":
        pos_x = margin_x
        pos_y = 24
    else:
        pos_x = max(10, screen_w - pip_w - margin_x)
        pos_y = max(10, screen_h - pip_h - margin_y)
    return pos_x, pos_y, pip_w, pip_h


def _get_opencv_root_hwnd(window_name: str):
    if _USER32 is None:
        return None
    hwnd = _USER32.FindWindowW(None, window_name)
    if not hwnd:
        return None
    root = _USER32.GetAncestor(hwnd, GA_ROOT)
    return root if root else hwnd


def _apply_pip_win32_styles(root_hwnd, pos_x: int, pos_y: int, pip_w: int, pip_h: int) -> None:
    if _USER32 is None or not root_hwnd:
        return
    if _USER32.IsIconic(root_hwnd):
        _USER32.ShowWindow(root_hwnd, 9)  # SW_RESTORE
    ex_style = int(_USER32.GetWindowLongW(root_hwnd, GWL_EXSTYLE))
    new_style = (
        (ex_style & ~WS_EX_APPWINDOW)
        | WS_EX_LAYERED
        | WS_EX_TRANSPARENT
        | WS_EX_TOPMOST
        | WS_EX_NOACTIVATE
        | WS_EX_TOOLWINDOW
    )
    _USER32.SetWindowLongW(root_hwnd, GWL_EXSTYLE, new_style)
    _USER32.SetLayeredWindowAttributes(root_hwnd, 0, 230, LWA_ALPHA)
    _USER32.SetWindowPos(
        root_hwnd,
        HWND_TOPMOST,
        pos_x,
        pos_y,
        pip_w,
        pip_h,
        SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_FRAMECHANGED,
    )


def set_opencv_window_pip_overlay(
    window_name: str,
    enable_pip: bool,
    screen_w: int,
    screen_h: int,
    corner: str = "bottom_right",
) -> None:
    """
    Cấu hình cửa sổ Camera:
    - enable_pip=True: Thu nhỏ 480x270 ở góc được chọn (`bottom_right`, `bottom_left`, `top_right`, `top_left`),
      luôn nổi trên PowerPoint Slide Show (TopMost) và xuyên chuột 100% (Click-Through).
    - enable_pip=False: Cửa sổ bình thường / Full màn hình.
    """
    if _USER32 is None:
        return
    root_hwnd = _get_opencv_root_hwnd(window_name)
    if not root_hwnd:
        return

    if enable_pip:
        pos_x, pos_y, pip_w, pip_h = _get_pip_rect(screen_w, screen_h, corner=corner)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, pip_w, pip_h)
            cv2.moveWindow(window_name, pos_x, pos_y)
        except Exception:
            pass
        _apply_pip_win32_styles(root_hwnd, pos_x, pos_y, pip_w, pip_h)
    else:
        ex_style = int(_USER32.GetWindowLongW(root_hwnd, GWL_EXSTYLE))
        new_style = (ex_style & ~(WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)) | WS_EX_APPWINDOW
        _USER32.SetWindowLongW(root_hwnd, GWL_EXSTYLE, new_style)
        _USER32.SetLayeredWindowAttributes(root_hwnd, 0, 255, LWA_ALPHA)
        _USER32.SetWindowPos(
            root_hwnd,
            HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        )


def maintain_opencv_pip_overlay(
    window_name: str,
    screen_w: int,
    screen_h: int,
    corner: str = "bottom_right",
) -> None:
    """
    Watchdog nhẹ (~0.02ms): đảm bảo cửa sổ PiP không bị PowerPoint Slide Show (F5)
    hoặc SetForegroundWindow che khuất mất trong quá trình thuyết trình.
    """
    if _USER32 is None:
        return
    root_hwnd = _get_opencv_root_hwnd(window_name)
    if not root_hwnd:
        return

    pos_x, pos_y, pip_w, pip_h = _get_pip_rect(screen_w, screen_h, corner=corner)
    ex_style = int(_USER32.GetWindowLongW(root_hwnd, GWL_EXSTYLE))
    required_bits = WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE
    needs_style = (ex_style & required_bits) != required_bits or _USER32.IsIconic(root_hwnd) or not _USER32.IsWindowVisible(root_hwnd)

    rect = wintypes.RECT()
    needs_geom = False
    if _USER32.GetWindowRect(root_hwnd, ctypes.byref(rect)):
        cur_w = rect.right - rect.left
        cur_h = rect.bottom - rect.top
        if abs(rect.left - pos_x) > 8 or abs(rect.top - pos_y) > 8 or abs(cur_w - pip_w) > 32 or abs(cur_h - pip_h) > 48:
            needs_geom = True

    covered_by_window = False
    if not (needs_style or needs_geom):
        prev_hwnd = _USER32.GetWindow(root_hwnd, GW_HWNDPREV)
        other_rc = wintypes.RECT()
        pip_r, pip_b = pos_x + pip_w, pos_y + pip_h
        steps = 0
        while prev_hwnd and steps < 24:
            if _USER32.IsWindowVisible(prev_hwnd) and _USER32.GetWindowTextLengthW(prev_hwnd) > 0:
                if _USER32.GetWindowRect(prev_hwnd, ctypes.byref(other_rc)):
                    ow = other_rc.right - other_rc.left
                    oh = other_rc.bottom - other_rc.top
                    overlaps_pip = (
                        ow >= 80
                        and oh >= 80
                        and other_rc.right > pos_x
                        and other_rc.left < pip_r
                        and other_rc.bottom > pos_y
                        and other_rc.top < pip_b
                    )
                    if overlaps_pip:
                        covered_by_window = True
                        break
            prev_hwnd = _USER32.GetWindow(prev_hwnd, GW_HWNDPREV)
            steps += 1

    if needs_style:
        _apply_pip_win32_styles(root_hwnd, pos_x, pos_y, pip_w, pip_h)
    elif needs_geom or covered_by_window:
        _USER32.SetWindowPos(
            root_hwnd,
            HWND_TOPMOST,
            pos_x,
            pos_y,
            pip_w,
            pip_h,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )


class OneEuroFilter2D:
    """
    Bộ lọc One Euro (Casiez 2012) cho toạ độ con trỏ 2D (pixel):
    - Tay đứng yên / di chậm  -> tần số cắt thấp (min_cutoff) -> lọc mạnh rung tay của MediaPipe.
    - Tay di nhanh            -> tần số cắt tăng theo tốc độ (beta * |v|) -> bám sát, không bị trễ.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev = np.zeros(2, dtype=np.float64)
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: Tuple[float, float], t: float) -> Tuple[float, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or self.t_prev is None:
            self.x_prev, self.t_prev = x_arr, t
            return float(x_arr[0]), float(x_arr[1])
        dt = max(t - self.t_prev, 1e-3)
        dx = (x_arr - self.x_prev) / dt
        a_d = self._alpha(np.full(2, self.d_cutoff), dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x_arr + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return float(x_hat[0]), float(x_hat[1])


class CursorRingOverlay:
    """
    Cửa sổ Win32 GDI siêu nhẹ (76x76 px), trong suốt & xuyên chuột 100% (Click-Through TopMost)
    hiển thị vòng tròn tiến độ trực tiếp quanh con trỏ trên PowerPoint Slide Show:
    - `dwell_arm` : Vòng cam nạp dần khi giữ yên ngón tay (chuyển Laser -> Bút).
    - `pen_ready` : Vòng xanh lá/lục lam báo hiệu Bút AI đã mở, sẵn sàng vẽ.
    - `ai_commit` : Vòng xanh dương đếm ngược ~1s trước khi AI nắn thành hình Vector.
    - `pen_finish`: Vòng cam nhỏ báo sắp nhả nét về lại Laser.
    Ẩn hoàn toàn khi đang di Laser bình thường để không che mất con trỏ Laser chính chủ.
    """

    def __init__(self, size_px: int = 76) -> None:
        self.size_px = max(48, int(size_px))
        self.half_px = self.size_px // 2
        self._lock = threading.Lock()
        self._x = 0
        self._y = 0
        self._mode = "hidden"
        self._progress = 0.0
        self._visible = False
        self._closed = False
        self._started = False
        self._hwnd = None
        self._thread: Optional[threading.Thread] = None
        self._ready_ev = threading.Event()
        self._wndproc_ref = None
        self._last_sent: Optional[Tuple[int, int, str, int, bool]] = None

    def _ensure_started(self) -> None:
        if os.name != "nt" or _USER32 is None or self._closed or self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._win32_thread_main, daemon=True)
        self._thread.start()
        self._ready_ev.wait(timeout=0.35)

    def _win32_thread_main(self) -> None:
        try:
            gdi32 = ctypes.windll.gdi32
            kernel32 = ctypes.windll.kernel32
            lresult_t = ctypes.c_ssize_t

            class _WNDCLASSEXW(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.UINT),
                    ("style", wintypes.UINT),
                    ("lpfnWndProc", ctypes.c_void_p),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HCURSOR),
                    ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR),
                    ("hIconSm", wintypes.HICON),
                ]

            class _PAINTSTRUCT(ctypes.Structure):
                _fields_ = [
                    ("hdc", wintypes.HDC),
                    ("fErase", wintypes.BOOL),
                    ("rcPaint", wintypes.RECT),
                    ("fRestore", wintypes.BOOL),
                    ("fIncUpdate", wintypes.BOOL),
                    ("rgbReserved", ctypes.c_byte * 32),
                ]

            WNDPROC = ctypes.WINFUNCTYPE(
                lresult_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            )

            kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            _USER32.DefWindowProcW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            _USER32.DefWindowProcW.restype = lresult_t
            _USER32.RegisterClassExW.argtypes = [ctypes.POINTER(_WNDCLASSEXW)]
            _USER32.RegisterClassExW.restype = wintypes.ATOM
            _USER32.CreateWindowExW.argtypes = [
                wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
            ]
            _USER32.CreateWindowExW.restype = wintypes.HWND
            _USER32.GetMessageW.argtypes = [
                ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT
            ]
            _USER32.GetMessageW.restype = wintypes.BOOL
            _USER32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
            _USER32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
            _USER32.PostMessageW.argtypes = [
                wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
            ]
            _USER32.PostQuitMessage.argtypes = [ctypes.c_int]
            _USER32.DestroyWindow.argtypes = [wintypes.HWND]
            _USER32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(_PAINTSTRUCT)]
            _USER32.BeginPaint.restype = wintypes.HDC
            _USER32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(_PAINTSTRUCT)]
            _USER32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT), wintypes.BOOL]
            _USER32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]

            gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
            gdi32.CreateCompatibleDC.restype = wintypes.HDC
            gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
            gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
            gdi32.DeleteDC.argtypes = [wintypes.HDC]
            gdi32.CreateSolidBrush.argtypes = [wintypes.DWORD]
            gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
            gdi32.CreatePen.argtypes = [ctypes.c_int, ctypes.c_int, wintypes.DWORD]
            gdi32.CreatePen.restype = wintypes.HPEN
            gdi32.GetStockObject.argtypes = [ctypes.c_int]
            gdi32.GetStockObject.restype = wintypes.HGDIOBJ
            gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
            gdi32.SelectObject.restype = wintypes.HGDIOBJ
            gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
            gdi32.Ellipse.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            gdi32.Polyline.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.POINT), ctypes.c_int]
            gdi32.BitBlt.argtypes = [
                wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
            ]

            WM_DESTROY = 0x0002
            WM_PAINT = 0x000F
            WM_CLOSE = 0x0010
            WM_ERASEBKGND = 0x0014
            WM_MOUSEACTIVATE = 0x0021
            WM_NCHITTEST = 0x0084
            WM_APP_SYNC = 0x8001
            WS_POPUP = 0x80000000
            SW_HIDE = 0
            LWA_COLORKEY = 0x00000001
            COLORKEY_RGB = 0x00030201  # RGB(1, 2, 3)

            sz = self.size_px
            cx = sz // 2
            cy = sz // 2
            r_ring = max(16, cx - 10)

            def _paint_ring(hdc_win: wintypes.HDC) -> None:
                with self._lock:
                    mode = self._mode
                    prog = self._progress
                    vis = self._visible

                mem_dc = gdi32.CreateCompatibleDC(hdc_win)
                bmp = gdi32.CreateCompatibleBitmap(hdc_win, sz, sz)
                old_bmp = gdi32.SelectObject(mem_dc, bmp)
                bg_brush = gdi32.CreateSolidBrush(COLORKEY_RGB)
                rc = wintypes.RECT(0, 0, sz, sz)
                _USER32.FillRect(mem_dc, ctypes.byref(rc), bg_brush)
                gdi32.DeleteObject(bg_brush)

                if vis and mode != "hidden":
                    hollow = gdi32.GetStockObject(5)  # HOLLOW_BRUSH
                    old_brush = gdi32.SelectObject(mem_dc, hollow)

                    if mode == "pen_ready":
                        pen_c = gdi32.CreatePen(0, 4, 0x0078EB00)
                        old_pen = gdi32.SelectObject(mem_dc, pen_c)
                        gdi32.Ellipse(mem_dc, cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring)
                        gdi32.SelectObject(mem_dc, old_pen)
                        gdi32.DeleteObject(pen_c)
                    else:
                        # Vòng nền xám tối mỏng + cung tròn tiến độ mượt
                        track_pen = gdi32.CreatePen(0, 3, 0x00463A34)
                        old_pen = gdi32.SelectObject(mem_dc, track_pen)
                        gdi32.Ellipse(mem_dc, cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring)
                        gdi32.SelectObject(mem_dc, old_pen)
                        gdi32.DeleteObject(track_pen)

                        clamped = max(0.02, min(1.0, float(prog)))
                        arc_col = (
                            0x00FFDC00 if mode == "ai_commit"
                            else (0x002878FF if mode == "pen_finish" else 0x0000A5FF)
                        )
                        arc_pen = gdi32.CreatePen(0, 5, arc_col)
                        old_arc_pen = gdi32.SelectObject(mem_dc, arc_pen)
                        n_pts = max(4, int(round(36 * clamped)))
                        pts = (wintypes.POINT * n_pts)()
                        start_rad = -math.pi * 0.5
                        span_rad = 2.0 * math.pi * clamped
                        for i in range(n_pts):
                            th = start_rad + span_rad * (i / float(n_pts - 1))
                            pts[i].x = int(round(cx + r_ring * math.cos(th)))
                            pts[i].y = int(round(cy + r_ring * math.sin(th)))
                        gdi32.Polyline(mem_dc, pts, n_pts)
                        gdi32.SelectObject(mem_dc, old_arc_pen)
                        gdi32.DeleteObject(arc_pen)

                    gdi32.SelectObject(mem_dc, old_brush)

                gdi32.BitBlt(hdc_win, 0, 0, sz, sz, mem_dc, 0, 0, 0x00CC0020)
                gdi32.SelectObject(mem_dc, old_bmp)
                gdi32.DeleteObject(bmp)
                gdi32.DeleteDC(mem_dc)

            def _wndproc(hwnd, msg, wparam, lparam):
                if msg == WM_NCHITTEST:
                    return -1  # HTTRANSPARENT
                if msg == WM_MOUSEACTIVATE:
                    return 3   # MA_NOACTIVATE
                if msg == WM_ERASEBKGND:
                    return 1
                if msg == WM_APP_SYNC:
                    with self._lock:
                        x, y = self._x, self._y
                        vis = self._visible
                    if vis:
                        _USER32.SetWindowPos(
                            hwnd,
                            HWND_TOPMOST,
                            x - self.half_px,
                            y - self.half_px,
                            sz,
                            sz,
                            SWP_NOACTIVATE | SWP_SHOWWINDOW,
                        )
                        _USER32.InvalidateRect(hwnd, None, False)
                    else:
                        _USER32.ShowWindow(hwnd, SW_HIDE)
                    return 0
                if msg == WM_PAINT:
                    ps = _PAINTSTRUCT()
                    hdc = _USER32.BeginPaint(hwnd, ctypes.byref(ps))
                    if hdc:
                        _paint_ring(hdc)
                        _USER32.EndPaint(hwnd, ctypes.byref(ps))
                    return 0
                if msg == WM_CLOSE:
                    _USER32.DestroyWindow(hwnd)
                    return 0
                if msg == WM_DESTROY:
                    _USER32.PostQuitMessage(0)
                    return 0
                return _USER32.DefWindowProcW(hwnd, msg, wparam, lparam)

            wndproc_cb = WNDPROC(_wndproc)
            self._wndproc_ref = wndproc_cb
            hinst = kernel32.GetModuleHandleW(None)
            cls_name = f"PPT_HandCtrl_Ring_{id(self)}"

            wc = _WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
            wc.style = 0
            wc.lpfnWndProc = ctypes.cast(wndproc_cb, ctypes.c_void_p)
            wc.hInstance = hinst
            wc.lpszClassName = cls_name
            _USER32.RegisterClassExW(ctypes.byref(wc))

            ex_style = (
                WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            )
            hwnd = _USER32.CreateWindowExW(
                ex_style, cls_name, "", WS_POPUP, 0, 0, sz, sz, None, None, hinst, None
            )
            if not hwnd:
                self._ready_ev.set()
                return
            _USER32.SetLayeredWindowAttributes(hwnd, COLORKEY_RGB, 0, LWA_COLORKEY)
            self._hwnd = hwnd
            self._ready_ev.set()

            msg = wintypes.MSG()
            while _USER32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                _USER32.TranslateMessage(ctypes.byref(msg))
                _USER32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            self._ready_ev.set()

    def update(
        self,
        scr_x: int,
        scr_y: int,
        mode: str = "hidden",
        progress: float = 0.0,
        visible: bool = False,
    ) -> None:
        """Cập nhật vị trí và trạng thái vòng tròn tiến độ quanh con trỏ."""
        if self._closed or os.name != "nt" or _USER32 is None:
            return
        if not visible and not self._started:
            return
        self._ensure_started()
        if not self._hwnd:
            return

        state_key = (int(scr_x), int(scr_y), str(mode), int(round(float(progress) * 100.0)), bool(visible))
        if state_key == self._last_sent:
            return
        self._last_sent = state_key

        with self._lock:
            self._x = int(scr_x)
            self._y = int(scr_y)
            self._mode = str(mode)
            self._progress = float(progress)
            self._visible = bool(visible)
        try:
            _USER32.PostMessageW(self._hwnd, 0x8001, 0, 0)
        except Exception:
            pass

    def hide(self) -> None:
        self.update(0, 0, mode="hidden", progress=0.0, visible=False)

    def close(self) -> None:
        self._closed = True
        if self._hwnd and _USER32 is not None:
            try:
                _USER32.PostMessageW(self._hwnd, 0x0010, 0, 0)  # WM_CLOSE
            except Exception:
                pass
            self._hwnd = None

