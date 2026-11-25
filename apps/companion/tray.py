import ctypes
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from PIL import Image, ImageDraw

APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "MangaTVL"
LOG = APP_DIR / "companion.log"
PORT = int(os.getenv("MTVL_PORT", "8765"))
RELEASES_URL = "https://github.com/parwer/MangaTVL/releases/latest"
COLORS = {"brand": (226, 51, 74, 255), "ready": (46, 160, 67, 255), "loading": (214, 138, 0, 255),
          "error": (200, 40, 40, 255)}


def icon_image(size: int = 64, state: str = "brand") -> Image.Image:
    """App icon: a speech balloon with two lines of text, colored by state (brand, ready, loading, error)."""
    big = size * 4  # drawn large, then scaled down for smooth edges
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = big / 64
    fill = COLORS.get(state, COLORS["brand"])
    d.rounded_rectangle([4 * s, 6 * s, 60 * s, 46 * s], radius=14 * s, fill=fill)
    d.polygon([(14 * s, 40 * s), (10 * s, 60 * s), (30 * s, 45 * s)], fill=fill)
    white = (255, 255, 255, 255)
    d.rounded_rectangle([14 * s, 17 * s, 50 * s, 23 * s], radius=3 * s, fill=white)
    d.rounded_rectangle([14 * s, 29 * s, 40 * s, 35 * s], radius=3 * s, fill=white)
    return im.resize((size, size), Image.LANCZOS)


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def message(text: str):
    ctypes.windll.user32.MessageBoxW(None, text, "MangaTVL Companion", 0x40)


def main():
    """Windowless companion: the server answers at once, the models load beside it, a tray icon shows the state."""
    import pystray

    APP_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log  # no console in the windowed exe (None would break uvicorn's logging)
    if port_in_use(PORT):
        message("MangaTVL Companion is already running.\n\nLook for its icon in the system tray "
                "(bottom-right corner of the screen, you may need to click ^ to see it).")
        return

    import uvicorn
    from apps.companion import server

    state = {"text": "Starting: loading models…"}
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="info"))
    server.LOAD["loading"] = True  # health says "starting" until the models are in

    def load():
        try:
            server.pipe()  # load the models before the first page arrives
            state["text"] = "Ready to translate"
            icon.icon, icon.title = icon_image(state="ready"), "MangaTVL Companion: ready"
        except Exception as e:  # still serve: health tells the extension, the log has the details
            server.LOAD["error"] = str(e)[:200]
            state["text"] = f"Couldn't load the models: {e}"[:120]
            icon.icon, icon.title = icon_image(state="error"), "MangaTVL Companion: couldn't load the models"
            print(f"model load failed: {e!r}", flush=True)
        finally:
            server.LOAD["loading"] = False
        try:
            icon.update_menu()
        except Exception:  # the tray may not be up yet
            pass

    def quit_app(icon_, _):
        srv.should_exit = True
        icon_.stop()

    menu = pystray.Menu(
        pystray.MenuItem(lambda _: state["text"], None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Get the latest version", lambda _i, _m: webbrowser.open(RELEASES_URL)),
        pystray.MenuItem("Open log file", lambda _i, _m: subprocess.Popen(["notepad", str(LOG)])),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit MangaTVL Companion", quit_app),
    )
    icon = pystray.Icon("MangaTVL", icon_image(state="loading"), "MangaTVL Companion: starting…", menu)
    def shown(icon_):
        icon_.visible = True
        try:  # the app has no window: say where it went
            icon_.notify("Running minimized in the system tray.", "MangaTVL Companion")
        except Exception as e:  # notifications can be turned off; the icon is still there
            print(f"start notification failed: {e!r}", flush=True)

    threading.Thread(target=srv.run, daemon=True).start()
    threading.Thread(target=load, daemon=True).start()
    icon.run(setup=shown)


if __name__ == "__main__":
    main()
