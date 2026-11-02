import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "manga_translator" / "assets" / "models"


def timed(name, fn, n):
    """Run fn n times and print each wall time (the first includes graph compilation)."""
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append(round(time.perf_counter() - t, 3))
    print(f"{name:42s} {out}", flush=True)


def main():
    """Per-model DirectML timings: fixed vs changing input shapes (does DML recompile per shape?)."""
    import onnxruntime as ort
    sys.path.insert(0, str(REPO))
    print("providers:", ort.get_available_providers())
    rng = np.random.default_rng(0)
    for dev in ("dml", "cpu"):
        from manga_translator.masking import TextMasker
        tm = TextMasker(device=dev)
        page = rng.integers(0, 255, (1400, 1000, 3), dtype=np.uint8)
        timed(f"CTD 1024 fixed [{dev}]", lambda: tm.predict(page), 3)

        from manga_translator.inpainting.lama_inpainter import LamaInpainter
        lama = LamaInpainter(device=dev, onnx_path=MODELS / "lama_ft2_512.onnx")
        crop = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        hole = np.zeros((300, 400), np.uint8)
        hole[100:200, 100:300] = 255
        timed(f"LaMa 512 fixed [{dev}]", lambda: lama.lama(crop, hole), 3)

        from manga_translator.detection.onnx_detection import ONNXDetection
        det = ONNXDetection(MODELS / "best_diplom.onnx", device=dev)
        shapes = iter([(1400, 1000), (1300, 900), (1500, 1100), (1400, 1000)])
        timed(f"YOLO dynamic, 4 page sizes [{dev}]",
              lambda: det.detect(rng.integers(0, 255, (*next(shapes), 3), dtype=np.uint8)), 4)

    from rapidocr import RapidOCR
    for dml in (True, False):
        eng = RapidOCR(params={"EngineConfig.onnxruntime.use_dml": dml, "Global.log_level": "error",
                               "Global.use_cls": False})
        sizes = iter([(60, 200), (80, 240), (120, 300), (60, 200), (90, 180), (150, 420)])
        timed(f"RapidOCR crops, 6 sizes [dml={dml}]",
              lambda: eng(rng.integers(0, 255, (*next(sizes), 3), dtype=np.uint8)), 6)


if __name__ == "__main__":
    main()
