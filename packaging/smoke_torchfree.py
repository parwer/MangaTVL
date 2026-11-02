import asyncio
import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "packaging" / "build" / "app"


async def main(pages):
    """Run the staged app's pipeline on pages without torch; writes packaging/build/smoke/<page>.png."""
    for mod in ("torch", "ultralytics", "easyocr", "manga_ocr"):
        assert importlib.util.find_spec(mod) is None, f"{mod} is installed -- not a torch-free env"
    sys.path.insert(0, str(APP))
    from dotenv import dotenv_values
    from PIL import Image
    import apps.companion.server as srv
    assert str(APP) in srv.__file__, "not the staged copy"
    key = dotenv_values(REPO / ".env").get("OPENROUTER_API_KEY")
    t = time.time()
    pipe = srv.pipe()
    print(f"load {time.time() - t:.1f}s")
    out = REPO / "packaging" / "build" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    chapter: dict = {}
    for p in pages:
        t = time.time()
        page = pipe.analyze(Image.open(p).convert("RGB"))
        if page.ocr and key:
            results = await pipe.translate_page(page, srv.translator_for("openrouter", key, None), chapter=chapter)
            pipe.finish(page, results, "thai").save(out / f"{Path(p).stem}.png")
        print(Path(p).stem, f"{len(page.ocr)} regions, reverted {[i for i, _ in page.reverted]}, "
              f"{time.time() - t:.1f}s", page.timings, flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
