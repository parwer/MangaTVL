import ctypes
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

from PIL import Image, ImageDraw

APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "MangaTVL"
LOG = APP_DIR / "companion.log"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_NAME = "MangaTVL Companion"  # the installer's "start with Windows" task writes the same value
PORT = int(os.getenv("MTVL_PORT", "8765"))


def icon_image(size: int = 64, ready: bool = True) -> Image.Image:
    """Tray / exe icon: a speech balloon, green when the models are loaded."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size / 64
    fill = (46, 160, 67, 255) if ready else (200, 120, 0, 255)
    d.ellipse([4 * s, 6 * s, 60 * s, 46 * s], fill=fill)
    d.polygon([(16 * s, 40 * s), (12 * s, 60 * s), (30 * s, 44 * s)], fill=fill)
    d.text((20 * s, 12 * s), "TH", fill="white", font_size=int(20 * s))
    return im


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def message(text: str):
    ctypes.windll.user32.MessageBoxW(None, text, "MangaTVL Companion", 0x40)


def autostart() -> bool:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, RUN_NAME)
            return True
    except OSError:
        return False


def set_autostart(on: bool):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        if on:
            winreg.SetValueEx(k, RUN_NAME, 0, winreg.REG_SZ, f'"{sys.executable}"')
        else:
            try:
                winreg.DeleteValue(k, RUN_NAME)
            except OSError:
                pass


def main():
    """Windowless companion: the server runs on a thread, a tray icon shows its state."""
    import pystray

    APP_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log  # no console in the windowed exe (None would break uvicorn's logging)
    if port_in_use(PORT):
        message("MangaTVL Companion เปิดอยู่แล้ว (ดูไอคอนที่มุมขวาล่างของจอ)")
        return

    import uvicorn
    from apps.companion import server

    state = {"text": "กำลังโหลดโมเดล…"}
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="info"))

    def run():
        try:
            server.pipe()  # load the models before the first page arrives
            state["text"] = "พร้อมใช้งาน"
            icon.icon, icon.title = icon_image(ready=True), "MangaTVL Companion: พร้อมใช้งาน"
        except Exception as e:  # still serve: health tells the extension, the log has the details
            state["text"] = f"โหลดโมเดลไม่สำเร็จ: {e}"[:120]
            print(f"model load failed: {e!r}", flush=True)
        icon.update_menu()
        srv.run()

    def quit_app(icon_, _):
        srv.should_exit = True
        icon_.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _: state["text"], None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("เปิดพร้อม Windows", lambda _i, _m: set_autostart(not autostart()),
                         checked=lambda _: autostart(), visible=getattr(sys, "frozen", False)),
        pystray.MenuItem("เปิดไฟล์ log", lambda _i, _m: subprocess.Popen(["notepad", str(LOG)])),
        pystray.MenuItem("ปิดโปรแกรม", quit_app),
    )
    icon = pystray.Icon("MangaTVL", icon_image(ready=False), "MangaTVL Companion: กำลังโหลดโมเดล…", menu)
    threading.Thread(target=run, daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()
