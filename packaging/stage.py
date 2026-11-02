import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "packaging" / "build" / "app"
CODE = ["pipeline.py", "manga_translator", "apps/__init__.py", "apps/companion", "apps/extension"]
MODELS = ["comictextdetector.pt.onnx", "lama_ft2_512.onnx", "best_diplom.onnx", "bubble_freetext_yolo26s.onnx",
          "baberu"]
SKIP = {"__pycache__", "models"}  # assets/models is copied selectively (ONNX only)


def place(src: Path, dst: Path):
    """Copy a file or tree; big model files are hard-linked when possible."""
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file() and not SKIP & set(f.relative_to(src).parts):
                place(f, dst / f.relative_to(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if src.stat().st_size > 2 ** 20 and src.suffix in (".onnx", ".bin", ".json"):
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    """Stage the torch-free companion app into packaging/build/app (same layout as the repo)."""
    if "--clean" in sys.argv and OUT.exists():
        shutil.rmtree(OUT)
    for c in CODE:
        place(REPO / c, OUT / c)
    models = REPO / "manga_translator" / "assets" / "models"
    for m in MODELS:
        place(models / m, OUT / "manga_translator" / "assets" / "models" / m)
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"staged {len(files)} files, {sum(f.stat().st_size for f in files) / 2 ** 20:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
