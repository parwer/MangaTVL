"""Save per-page image artefacts for visual audit (only when --save-images).

For each page writes into ``dest_dir/``:
    original.png      the source page
    inpainted.png     after text removal (stage 4 output)
    rendered.png      after translated text drawn (stage 5 output)
    text_mask.png     pixels that changed inpainted -> rendered (the drawn glyphs)
    overlay.png       original with each bubble's polygon (green) + bbox (red) drawn
"""

import numpy as np
import cv2
from PIL import Image

from manga_translator.utils.common import pil2cv, cv2pil


def _save(img, path):
    if img is None:
        return
    if isinstance(img, np.ndarray):
        img = cv2pil(img) if img.ndim == 3 else Image.fromarray(img)
    img.save(path)


def polygon_overlay(cap) -> Image.Image:
    """Original page with each detection's polygon (green) and bbox (red)."""
    cv_img = pil2cv(cap.original).copy()
    for det in cap.detection:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        poly = getattr(det, "segmentation", None)
        if poly:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(cv_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return cv2pil(cv_img)


def save_page(cap, dest_dir, text_mask=None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    _save(cap.original, dest_dir / "original.png")
    _save(cap.inpainted, dest_dir / "inpainted.png")
    _save(cap.rendered, dest_dir / "rendered.png")
    if text_mask is not None:
        _save((text_mask > 0).astype(np.uint8) * 255, dest_dir / "text_mask.png")
    _save(polygon_overlay(cap), dest_dir / "overlay.png")
