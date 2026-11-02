"""Region repairs from the glyph mask: text the balloon / free-text detectors missed or cut.

The detectors work on boxes; the text masker (``masking.TextMasker``) sees letters. Using
its glyph mask and text blocks:

* ``straddling_blocks``: an aside written across a balloon outline (half inside, half
  outside the polygon) becomes its own free_text region instead of being half erased.
* ``complete_free_text``: a free_text box covering only part of a text line is grown to the
  whole line; overlapping free_text boxes are merged.
* ``missed_blocks``: text blocks no detection covers (captions on dark panels, text over art).
* ``orphan_glyphs``: lettering segmented by the masker but in no text block and no detection
  (stylised SFX), clustered into free_text candidates (the OCR letter filter drops art).

``refine_regions`` applies all four in the order the pipeline uses.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..schemas.interface import DetectionResult


def _cover(a, b) -> float:
    """Share of box a covered by box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h) / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def refine_regions(dets: list[DetectionResult], blocks, glyph: np.ndarray, core: np.ndarray,
                   gray: np.ndarray) -> list[DetectionResult]:
    """All repairs. ``blocks``: masker text blocks [x1, y1, x2, y2, score]; ``glyph``: refined
    glyph mask; ``core``: confident glyph pixels; ``gray``: the page in grayscale."""
    dets = list(dets) + straddling_blocks(dets, blocks, glyph, core, gray)
    complete_free_text(dets, blocks, glyph)
    dets += missed_blocks(dets, blocks, glyph)
    return dets + orphan_glyphs(dets, core)


# ------------------------------------------------------------------ straddling asides
def _barrier(gray, ink, s, c, dark: int = 90) -> bool:
    """True if a dark non-letter stroke (a balloon outline) separates two letter boxes on the
    same line, i.e. they belong to different speakers."""
    if gray is None:
        return False
    xa, xb = (s[2], c[0]) if s[2] <= c[0] else (c[2], s[0])
    if xb - xa < 1:
        return False
    ya, yb = max(s[1], c[1]), min(s[3], c[3])
    if yb <= ya:
        return False
    gap = (gray[ya:yb, xa:xb] < dark) & ~ink[ya:yb, xa:xb]
    return bool(gap.any(0).sum() >= 1 and gap.sum() >= 0.5 * (yb - ya))


def _aside_box(g, inside, gray=None, full: float = 0.9):
    """The aside part of a text block that straddles a balloon outline, in block coords.

    The masker often merges a balloon's dialogue with an aside written just outside into one
    block. Per letter component: mostly outside or crossing the outline -> aside (seed); fully
    inside -> dialogue, unless it sits on the same text line right next to a *crossing* seed
    (the inside half of an aside written across the edge). Returns (x1, y1, x2, y2) or None."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= 4:
            comps.append([x, y, x + w, y + h, float(inside[lab == i].mean())])
    seeds = [c for c in comps if c[4] < full]
    if not seeds:
        return None
    # crossing = really written across the edge (>= 30 % inside); a few % inside is just a loose
    # polygon grazing an outside aside -> no chaining
    crossing = [c for c in seeds if c[4] >= 0.3]
    grow = True
    while grow and crossing:
        grow = False
        for c in comps:
            if c in seeds or c[4] < full:
                continue
            for s in crossing:
                h = max(s[3] - s[1], c[3] - c[1])
                vov = min(s[3], c[3]) - max(s[1], c[1])
                gap = max(c[0] - s[2], s[0] - c[2], 0)
                if vov > 0.5 * min(s[3] - s[1], c[3] - c[1]) and gap < 0.8 * h and not _barrier(gray, g, s, c):
                    seeds.append(c)
                    crossing.append(c)
                    grow = True
                    break
    a = np.array(seeds)
    return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 2].max()), int(a[:, 3].max())


def straddling_blocks(dets, blocks, glyph, core=None, gray=None, lo: float = 0.15, hi: float = 0.85):
    """Text blocks partly inside, partly outside a balloon *polygon* -> free_text regions for
    their aside part (erased by the inpainter, which also redraws the outline)."""
    H, W = glyph.shape
    polys = []
    for d in dets:
        if d.form_type == "free_text" or not d.segmentation:
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.asarray(d.segmentation, np.int32).reshape(-1, 1, 2)], 255)
        polys.append(cv2.erode(m, np.ones((5, 5), np.uint8)))  # the outline stroke counts as outside
    if not polys:
        return []
    free = [d.bbox for d in dets if d.form_type == "free_text"]
    out = []
    for x1, y1, x2, y2, *_ in blocks:
        g = glyph[y1:y2, x1:x2] > 0
        tot = int(g.sum())
        if tot < 80:
            continue
        if any(min(x2, f[2]) - max(x1, f[0]) > 0.5 * (x2 - x1) and min(y2, f[3]) - max(y1, f[1]) > 0.5 * (y2 - y1)
               for f in free):
            continue  # already a free_text detection
        for m in polys:
            inside = int((g & (m[y1:y2, x1:x2] > 0)).sum()) / tot
            if lo < inside < hi:
                # split on confident ink (letters), not the dilated glyph mask whose blobs bridge
                # the aside and the dialogue
                ink = (core[y1:y2, x1:x2] > 0) if core is not None else g
                ink = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                box = _aside_box(ink, m[y1:y2, x1:x2] > 0, gray[y1:y2, x1:x2] if gray is not None else None)
                if box:
                    a, b, c, e = box
                    out.append(DetectionResult(bbox=[int(x1 + a), int(y1 + b), int(x1 + c), int(y1 + e)],
                                               form_type="free_text"))
                break
    return out


# ------------------------------------------------------------------ partial free text
def complete_free_text(dets, blocks, glyph, share: float = 0.3) -> None:
    """Grow each free_text box to the glyphs of every text block it covers >= ``share`` of
    (unless the block is inside a balloon), then merge overlapping free_text boxes. In place."""
    free = [d for d in dets if d.form_type == "free_text"]
    if not free:
        return
    balloons = [d.bbox for d in dets if d.form_type != "free_text"]
    for x1, y1, x2, y2, *_ in blocks:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if any(_cover((x1, y1, x2, y2), b) > 0.3 for b in balloons):
            continue
        g = glyph[y1:y2, x1:x2] > 0
        tot = int(g.sum())
        if tot < 80:
            continue
        ys, xs = np.nonzero(g)
        gb = [x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1]
        for d in free:
            a, b, c, e = d.bbox
            part = int(g[max(0, b - y1):max(0, e - y1), max(0, a - x1):max(0, c - x1)].sum())
            nb = [min(a, gb[0]), min(b, gb[1]), max(c, gb[2]), max(e, gb[3])]
            # only complete a line the box already mostly holds: a block can span two neighbouring
            # balloons the detector missed, and growing a box to > 2x would swallow the other one
            grow = (nb[2] - nb[0]) * (nb[3] - nb[1]) / max(1, (c - a) * (e - b))
            if part >= share * tot and grow <= 2.0 and nb != list(d.bbox):
                d.bbox = nb
                d.segmentation = None  # the old polygon would clip the grown region's mask
    merged = True
    while merged:
        merged = False
        for i, d in enumerate(free):
            for f in free[i + 1:]:
                if max(_cover(d.bbox, f.bbox), _cover(f.bbox, d.bbox)) > 0.3:
                    d.bbox = [min(d.bbox[0], f.bbox[0]), min(d.bbox[1], f.bbox[1]),
                              max(d.bbox[2], f.bbox[2]), max(d.bbox[3], f.bbox[3])]
                    d.segmentation = None
                    free.remove(f)
                    dets.remove(f)
                    merged = True
                    break
            if merged:
                break


# ------------------------------------------------------------------ missed text
def missed_blocks(dets, blocks, glyph, cover: float = 0.75) -> list[DetectionResult]:
    """Text blocks the detectors missed -> free_text regions. Glyphs of existing detections are
    subtracted first and the box shrunk to the remaining ink, so a block spanning several
    balloons doesn't duplicate them."""
    taken = np.zeros_like(glyph)
    for d in dets:
        a, b, c, e = d.bbox
        taken[max(0, b - 4):e + 4, max(0, a - 4):c + 4] = 255
    extra = []
    for x1, y1, x2, y2, *_ in blocks:
        sub = glyph[y1:y2, x1:x2] & ~taken[y1:y2, x1:x2]
        tot = np.count_nonzero(glyph[y1:y2, x1:x2])
        if tot == 0 or np.count_nonzero(sub) < (1 - cover) * tot or np.count_nonzero(sub) < 80:
            continue
        ys, xs = np.nonzero(sub)
        bx1, by1, bx2, by2 = x1 + xs.min(), y1 + ys.min(), x1 + xs.max(), y1 + ys.max()
        if bx2 - bx1 > 12 and by2 - by1 > 12:
            extra.append(DetectionResult(bbox=[int(bx1), int(by1), int(bx2), int(by2)], form_type="free_text"))
            taken[by1:by2, bx1:bx2] = 255
    return extra


def orphan_glyphs(dets, core, gap: int = 15, min_px: int = 150) -> list[DetectionResult]:
    """Confident glyph pixels outside every detection, clustered into free_text candidates."""
    H, W = core.shape
    taken = np.zeros((H, W), np.uint8)
    for d in dets:
        a, b, c, e = d.bbox
        taken[max(0, b - 6):e + 6, max(0, a - 6):c + 6] = 255
    free = (core > 0) & (taken == 0)
    if free.sum() < min_px:
        return []
    grp = cv2.dilate(free.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grp, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        sel = free[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)
        if int(sel.sum()) < min_px or w < 12 or h < 12 or w * h > 0.05 * H * W:
            continue
        ys, xs = np.nonzero(sel)
        out.append(DetectionResult(bbox=[int(x + xs.min()), int(y + ys.min()), int(x + xs.max()) + 1,
                                         int(y + ys.max()) + 1], form_type="free_text"))
    return out
