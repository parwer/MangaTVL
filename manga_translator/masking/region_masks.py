"""Per-region glyph masks: which pixels of the page-level glyph mask belong to which
detected region (balloon / free text).

A region's mask is the glyph mask inside its polygon (+7 px) or box (+6 px), plus two
second opinions that catch lettering CTD under-segments: off-colour pixels near the text
inside a flat balloon, and high-contrast ink inside the OCR line boxes. Finally every pixel
is owned by exactly one region (the smallest detection containing it), so overlapping
balloons never erase or avoid each other's text.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas.interface import DetectionResult, OCRResult


def _poly(seg, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(seg, np.int32).reshape(-1, 1, 2)], 255)
    return m


def region_glyph_masks(bgr: np.ndarray, glyph: np.ndarray, items: list[OCRResult]) -> list[np.ndarray]:
    """uint8 0/255 mask per OCR item (same order)."""
    H, W = glyph.shape
    out = []
    for o in items:
        d = o.detection_result
        if d.segmentation:
            lim = cv2.dilate(_poly(d.segmentation, (H, W)), np.ones((7, 7), np.uint8))
        else:
            x1, y1, x2, y2 = d.bbox
            lim = np.zeros((H, W), np.uint8)
            lim[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6] = 255
        rm = glyph & lim
        if not rm.any() and o.boxes:  # CTD missed it -> the OCR boxes
            for a, b, c, e in o.boxes:
                cv2.rectangle(rm, (a - 2, b - 2), (c + 2, e + 2), 255, -1)
            rm &= lim
        if d.segmentation:
            rm |= bubble_residue(bgr, d.segmentation, rm, o.boxes)
        if d.form_type == "bubble":
            rm |= ink_in_boxes(bgr, o.boxes) & lim
        out.append(rm)
    return assign_owners(out, [o.detection_result for o in items], (H, W))


def bubble_residue(bgr, polygon, rm, boxes) -> np.ndarray:
    """Lettering CTD under-segments inside a flat balloon (ellipsis dots, punctuation, thin
    strokes): pixels in the balloon's inner area that differ from the balloon colour AND lie
    near the known text."""
    H, W = rm.shape
    pts = np.asarray(polygon, np.int32).reshape(-1, 1, 2)
    poly = _poly(polygon, (H, W))
    _, _, w, h = cv2.boundingRect(pts)
    e = max(4, int(0.05 * np.hypot(w, h)))
    inner = cv2.erode(poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    px = bgr[inner > 0].astype(np.float32)
    if len(px) < 50:
        return np.zeros_like(rm)
    bgc = np.median(px, 0)
    ring = inner & ~cv2.dilate(rm, np.ones((5, 5), np.uint8))
    ring_px = bgr[ring > 0].astype(np.float32)
    if not len(ring_px) or ring_px.std(0).max() > 12:  # not a flat balloon -> leave it to CTD
        return np.zeros_like(rm)
    diff = ((np.abs(bgr.astype(np.float32) - bgc).max(2) > 40) * 255).astype(np.uint8) & inner
    near = rm.copy()
    for a, b, c, d in boxes or []:
        cv2.rectangle(near, (a, b), (c, d), 255, -1)
    near = cv2.dilate(near, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return cv2.dilate(diff & near, np.ones((5, 5), np.uint8)) & inner


def ink_in_boxes(bgr, boxes, pad: int = 3) -> np.ndarray:
    """Inside each OCR line box, pixels contrasting strongly with the box's own background
    are ink. Catches words CTD misses (dithered / gradient balloons, unusual fonts)."""
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    out = np.zeros((H, W), np.uint8)
    for a, b, c, d in boxes or []:
        x1, y1, x2, y2 = max(0, a - pad), max(0, b - pad), min(W, c + pad), min(H, d + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        g = gray[y1:y2, x1:x2]
        bg = int(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
        ink = (np.abs(g - bg) > 70).astype(np.uint8) * 255
        if ink.mean() > 255 * 0.6:  # mostly "ink" -> art, not text on a balloon
            continue
        out[y1:y2, x1:x2] |= cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return out


def assign_owners(masks: list[np.ndarray], dets: list[DetectionResult], shape) -> list[np.ndarray]:
    """A pixel claimed by several regions goes to the smallest detection (polygon, else box)
    that contains it."""
    if len(masks) < 2:
        return masks
    H, W = shape
    owner = np.full((H, W), -1, np.int32)
    best = np.full((H, W), np.inf, np.float32)
    for i, d in enumerate(dets):
        if d.segmentation:
            shp = _poly(d.segmentation, (H, W))
        else:
            a, b, c, e = d.bbox
            shp = np.zeros((H, W), np.uint8)
            shp[max(0, b):e, max(0, a):c] = 255
        area = float(np.count_nonzero(shp)) or float("inf")
        take = (shp > 0) & (area < best)
        owner[take], best[take] = i, area
    claims = np.zeros((H, W), np.int32)
    for m in masks:
        claims += m > 0
    out = []
    for i, m in enumerate(masks):
        contested = (m > 0) & (claims > 1)
        keep = (m > 0) & ~(contested & (owner != i) & (owner >= 0))
        out.append((keep * 255).astype(np.uint8))
    return out
