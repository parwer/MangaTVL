import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ISCC = [Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe"]


def main():
    """Wrap packaging/dist/MangaTVL-Companion (from build_exe.py) into one Setup .exe (Inno Setup 6)."""
    iscc = next((p for p in ISCC if p.exists()), None)
    if iscc is None:
        raise SystemExit("Inno Setup 6 not found: winget install JRSoftware.InnoSetup")
    server = (REPO / "apps" / "companion" / "server.py").read_text(encoding="utf-8")
    version = re.search(r'^VERSION = "([^"]+)"', server, re.M).group(1)
    subprocess.run([str(iscc), f"/DAppVersion={version}", str(REPO / "packaging" / "installer.iss")], check=True)
    out = REPO / "packaging" / "dist" / f"MangaTVL-Companion-Setup-{version}.exe"
    print(f"built {out} ({out.stat().st_size / 2 ** 20:.0f} MB)")


if __name__ == "__main__":
    main()
