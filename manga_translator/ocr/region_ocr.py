"""OCR for overlapping regions without reading the same words twice.

A free_text region that overlaps a balloon (an aside written across its outline, or free
text found by the 2-class detector) would otherwise be read by both regions and translated
twice. Its letters are hidden (painted white) from the balloons' OCR; free text is read on
the untouched page. Afterwards regions with no readable text are dropped, as is free text
whose OCR has fewer than two letters (art read as "1" -- the model then echoes it, or
confuses it with the numbered tags, and a stray digit gets lettered).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..schemas.interface import DetectionResult, OCRResult


def _overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def grow_to_glyphs(regions: list[DetectionResult], glyph: np.ndarray, share: float = 0.3, pad: int = 3) -> np.ndarray:
    """Grow each region's box to the glyph components lying mostly (>= ``share``) inside it --
    detector boxes are often a few px tighter than the lettering -- updating ``bbox`` in place.
    Returns the pixel mask of those glyphs (to hide)."""
    H, W = glyph.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats((glyph > 0).astype(np.uint8), 8)
    out = np.zeros((H, W), np.uint8)
    for r in regions:
        a, b, c, e = r.bbox
        box = np.zeros((H, W), bool)
        box[max(0, b - pad):e + pad, max(0, a - pad):c + pad] = True
        ids, cnt = np.unique(lab[box], return_counts=True)
        for i, k in zip(ids, cnt):
            if i and k >= share * stats[i, cv2.CC_STAT_AREA]:
                box |= lab == i
        ys, xs = np.nonzero(box)
        r.bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        out[box] = 255
    return cv2.dilate(out, np.ones((3, 3), np.uint8))


def read_regions(engine, image: Image.Image, dets: list[DetectionResult], core: np.ndarray | None = None) -> list[OCRResult]:
    """OCR every detection with ``engine`` (an OCREngine), hiding overlapping free text from the
    balloons. ``core``: confident glyph pixels (letters), used to hide exactly the free text's
    letters. Returns the kept OCR results in detection order."""
    if not dets:
        return []
    hide = [d for d in dets if d.form_type == "free_text"
            and any(_overlap(d.bbox, b.bbox) for b in dets if b.form_type != "free_text")]
    if hide and core is not None:
        arr = np.asarray(image.convert("RGB")).copy()
        # letters, not dilated glyph blobs (those bridge the aside and the dialogue)
        arr[grow_to_glyphs(hide, cv2.dilate(core, np.ones((3, 3), np.uint8))) > 0] = 255
        masked = Image.fromarray(arr)
        ocr = [None] * len(dets)
        for idx, im in (([i for i, d in enumerate(dets) if d.form_type != "free_text"], masked),
                        ([i for i, d in enumerate(dets) if d.form_type == "free_text"], image)):
            if idx:
                for i, r in zip(idx, engine.get_ocr(im, [dets[i] for i in idx])):
                    ocr[i] = r
    else:
        ocr = engine.get_ocr(image, dets)
    return [o for o in ocr if o is not None and o.text.strip()
            and (o.detection_result.form_type != "free_text" or sum(ch.isalpha() for ch in o.text) >= 2)]
