import sys
from pathlib import Path

import PyInstaller.__main__

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "packaging" / "build" / "app"
DIST = REPO / "packaging" / "dist"
WORK = REPO / "packaging" / "build" / "pyi"


def main():
    """PyInstaller --onedir build of the staged torch-free companion (tray app, no console)."""
    launcher = APP / "mangatvl_companion.py"
    launcher.write_text("import multiprocessing\nfrom apps.companion.tray import main\n"
                        "if __name__ == '__main__':\n    multiprocessing.freeze_support()\n    main()\n",
                        encoding="utf-8")
    sys.path.insert(0, str(APP))
    from apps.companion.tray import icon_image
    ico = WORK / "mangatvl.ico"
    ico.parent.mkdir(parents=True, exist_ok=True)
    icon_image(256).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
    PyInstaller.__main__.run([
        str(launcher), "--name", "MangaTVL-Companion", "--onedir", "--noconfirm", "--windowed",
        "--icon", str(ico), "--hidden-import", "pystray._win32",
        "--distpath", str(DIST), "--workpath", str(WORK), "--specpath", str(WORK), "--paths", str(APP),
        # the server finds models via Path(__file__).parents[2] -> keep the repo layout inside _internal
        "--add-data", f"{APP / 'manga_translator' / 'assets'};manga_translator/assets",
        "--collect-all", "rapidocr", "--collect-all", "pythainlp", "--collect-all", "uharfbuzz",
        "--collect-all", "freetype", "--collect-binaries", "onnxruntime",
        "--collect-submodules", "manga_translator", "--collect-submodules", "apps",
        "--exclude-module", "torch", "--exclude-module", "ultralytics", "--exclude-module", "easyocr",
        "--exclude-module", "matplotlib", "--exclude-module", "tkinter",
    ])
    exe = DIST / "MangaTVL-Companion" / "MangaTVL-Companion.exe"
    size = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file()) / 2 ** 20
    print(f"built {exe} ({size:.0f} MB on disk)")


if __name__ == "__main__":
    main()
