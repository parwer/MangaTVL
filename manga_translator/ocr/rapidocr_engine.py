import cv2
import numpy as np
from PIL import Image

from ..schemas.interface import DetectionResult, OCRResult
from .ocr_engine import OCREngine

LANGS = {"en": "EN", "th": "TH", "ko": "KOREAN", "ch": "CH", "ch_sim": "CH", "zh": "CH", "ru": "CYRILLIC", "latin": "LATIN"}


def reading_order(boxes, texts):
    """Lines top to bottom (same row = centres within half a line), left to right."""
    items = sorted(zip(boxes, texts), key=lambda bt: (bt[0][1] + bt[0][3]) / 2)
    rows = []
    for b, t in items:
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if rows and abs(cy - rows[-1][0]) < 0.5 * max(h, rows[-1][1]):
            rows[-1][2].append((b, t))
        else:
            rows.append([cy, h, [(b, t)]])
    out = []
    for _, _, r in rows:
        out += sorted(r, key=lambda bt: bt[0][0])
    return out


def tighten(arr, b, thr=60):
    """Shrink a (loose) PP-OCR line box to the pixels contrasting with its border."""
    # loose boxes pulled balloon outlines into the erase mask -> no flat fill -> LaMa -> leftovers
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
    if x2 - x1 < 4 or y2 - y1 < 4:
        return [x1, y1, x2, y2]
    g = arr[y1:y2, x1:x2].astype(np.int16).mean(2)
    bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))
    ys, xs = np.nonzero(np.abs(g - bg) > thr)
    if len(xs) < 5:
        return [x1, y1, x2, y2]
    return [x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1]


class RapidOCREngine(OCREngine):
    """RapidOCR (PP-OCRv5 mobile, onnxruntime) per detection; no torch needed."""

    def __init__(self, language: str = "en", device: str = "cpu", **_):
        super().__init__(backend="rapidocr")
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        self.eng = RapidOCR(params={
            "EngineConfig.onnxruntime.use_dml": device == "dml",
            "Rec.lang_type": getattr(LangRec, LANGS.get(language, "EN")),
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Global.log_level": "error",
            "Global.use_cls": False,  # the angle classifier flipped short words; balloon text is upright
            "Det.limit_type": "max", "Det.limit_side_len": 960,  # default upscaled small crops: 4x slower
        })

    def ocr(self, crop: Image.Image):
        """Text and line boxes (crop coords) of one crop."""
        arr = np.asarray(crop.convert("RGB"))
        if min(arr.shape[:2]) < 8:
            return "", []
        # the detector misses text touching the crop edge: pad with the border colour
        pad = int(0.25 * max(8, min(arr.shape[:2])))
        bg = np.median(np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]]), 0)
        padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[int(v) for v in bg])
        s = 1.0
        if min(padded.shape[:2]) < 64:  # small balloons: the det model expects ~32 px text
            s = 64 / min(padded.shape[:2])
            padded = cv2.resize(padded, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        r = self.eng(padded)
        if r is None or r.boxes is None or not len(r.boxes):
            return "", []
        boxes = [[int(p[:, 0].min() / s) - pad, int(p[:, 1].min() / s) - pad,
                  int(p[:, 0].max() / s) - pad, int(p[:, 1].max() / s) - pad] for p in np.asarray(r.boxes)]
        lines = reading_order([tighten(arr, b) for b in boxes], list(r.txts))
        return " ".join(t for _, t in lines if t).strip(), [b for b, _ in lines]

    def get_ocr(self, image: Image.Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        out = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            crop = self.clean_poly(image.crop((x1, y1, x2, y2)), det.segmentation, offset=(x1, y1))
            text, boxes = self.ocr(crop)
            out.append(OCRResult(text=text, boxes=[[a + x1, b + y1, c + x1, d + y1] for a, b, c, d in boxes],
                                 detection_result=det))
        return out
