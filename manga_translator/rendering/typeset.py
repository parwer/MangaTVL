from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from ..utils.lang import canonical_name
from .shaper import get_font, stroke_mask
from .tokenize import tokenize

NO_BREAK_BEFORE = set("!?.,…~)]}」』、。！？ๆฯ\"'”’-—:;")


def _font(path, size: int):
    return get_font(str(path), int(size))


def break_units(text: str, lang: str) -> list[str]:
    """Break units for line breaking: a line may end between two units, never inside one."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    toks = tokenize(text, lang)
    if " " not in "".join(toks) and " " in text:  # space-delimited split: put the separators back
        toks = [t + " " for t in toks[:-1]] + toks[-1:]
    out: list[str] = []
    for t in toks:
        if not t:
            continue
        if out and (t[0] in NO_BREAK_BEFORE or t.isspace()):
            out[-1] += t
        else:
            out.append(t)
    return out


@dataclass
class Style:
    fill: tuple[int, int, int]
    stroke: tuple[int, int, int] | None
    stroke_frac: float  # stroke width as a fraction of font size


# ---------------------------------------------------------------- analysis
def detect_style(orig_bgr: np.ndarray, glyph_mask: np.ndarray, clean_bgr: np.ndarray, area: np.ndarray) -> Style:
    """Estimate original text colour / outline and whether the new text needs one."""
    core = cv2.erode(glyph_mask, np.ones((3, 3), np.uint8))
    if np.count_nonzero(core) < 30:
        core = glyph_mask
    px = orig_bgr[core > 0].reshape(-1, 3).astype(np.float32)
    bg_px = clean_bgr[area > 0].reshape(-1, 3).astype(np.float32)
    bg = np.median(bg_px, 0) if len(bg_px) else np.array([255, 255, 255], np.float32)
    bg_busy = len(bg_px) > 0 and float(bg_px.std(0).mean()) > 18
    lum = lambda c: 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
    fill = np.array([0, 0, 0], np.float32) if lum(bg) > 128 else np.array([255, 255, 255], np.float32)
    outline = None
    if len(px) >= 30:
        _, lbl, cen = cv2.kmeans(px, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
        d = [np.linalg.norm(c - bg) for c in cen]
        far, near = (0, 1) if d[0] > d[1] else (1, 0)
        if d[far] > 60:
            fill = cen[far]
        # A colour lying *between* fill and background (on the fill->bg line) is just
        # anti-aliasing / the paper inside the glyph box, not an outline: without this
        # check plain black-on-white scans got a grey halo around every letter.
        v = bg - cen[far]
        t = float(np.dot(cen[near] - cen[far], v) / max(float(np.dot(v, v)), 1e-6))
        perp = float(np.linalg.norm(cen[near] - (cen[far] + t * v)))
        blend = 0.0 < t < 1.0 and perp < 40 and not bg_busy
        if (not blend and np.linalg.norm(cen[near] - bg) > 60
                and np.linalg.norm(cen[near] - cen[far]) > 80):
            outline = cen[near]  # original had a visible outline (typical SFX / text over art)
    stroke = None
    if outline is not None:
        stroke = outline
    elif bg_busy:
        stroke = np.array([255, 255, 255], np.float32) if lum(fill) < 128 else np.array([0, 0, 0], np.float32)
    to_rgb = lambda c: tuple(int(v) for v in c[::-1])
    return Style(fill=to_rgb(fill), stroke=to_rgb(stroke) if stroke is not None else None,
                 stroke_frac=0.12 if outline is not None else 0.09)


def balloon_area(clean_bgr: np.ndarray, bbox, polygon, glyph_mask: np.ndarray, margin_scale: float = 1.0) -> np.ndarray:
    """Writable area mask for one region (uint8 0/255), computed on a crop around it."""
    H, W = clean_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    xs, ys = [x1 - pad, x2 + pad], [y1 - pad, y2 + pad]
    if polygon is not None and len(polygon) >= 3:
        pts = np.asarray(polygon, np.int32).reshape(-1, 2)
        xs += [int(pts[:, 0].min()), int(pts[:, 0].max())]
        ys += [int(pts[:, 1].min()), int(pts[:, 1].max())]
    e = 48  # > every morphology radius used below, so the crop gives the same pixels as the full page
    cx1, cy1, cx2, cy2 = max(0, min(xs) - e), max(0, min(ys) - e), min(W, max(xs) + e), min(H, max(ys) + e)
    poly = None if polygon is None else (np.asarray(polygon, np.int32).reshape(-1, 2) - [cx1, cy1]).tolist()
    sub = _balloon_area_full(clean_bgr[cy1:cy2, cx1:cx2], [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], poly,
                             None if glyph_mask is None else glyph_mask[cy1:cy2, cx1:cx2], margin_scale)
    out = np.zeros((H, W), np.uint8)
    out[cy1:cy2, cx1:cx2] = sub
    return out


def _balloon_area_full(clean_bgr, bbox, polygon, glyph_mask, margin_scale=1.0):
    H, W = clean_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    area = np.zeros((H, W), np.uint8)
    if polygon is not None and len(polygon) >= 3:
        cv2.fillPoly(area, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 255)
    else:
        cv2.rectangle(area, (x1, y1), (x2, y2), 255, -1)
    # Refine with the actual flat-colour balloon interior: pixels close to the
    # balloon colour, connected to where the text was. Removes polygon overshoot
    # onto the balloon border / tail and catches balloons YOLO under-segmented.
    gm = glyph_mask & area if glyph_mask is not None else None
    seed_mask = gm if gm is not None and gm.any() else area
    ring = cv2.dilate(seed_mask, np.ones((7, 7), np.uint8)) & ~seed_mask & area
    if ring.any():
        bgc = np.median(clean_bgr[ring > 0].astype(np.float32), 0)
        close = (np.abs(clean_bgr.astype(np.float32) - bgc).max(2) < 28).astype(np.uint8) * 255
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        lim = np.zeros_like(area)
        lim[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
        close &= lim
        n, lab = cv2.connectedComponents(close, 8)
        if n > 1:
            hits = np.bincount(lab[(seed_mask > 0) | (ring > 0)].ravel(), minlength=n)
            hits[0] = 0
            k = int(hits.argmax())
            cand = ((lab == k) * 255).astype(np.uint8)
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            cand = _fill_holes(cand)
            if cand.sum() > 0.25 * area.sum() or polygon is None:
                area = cand if polygon is None else cv2.bitwise_and(cand, cv2.dilate(area, np.ones((5, 5), np.uint8)))
    a = int(np.count_nonzero(area))
    margin = int(np.clip(0.06 * np.sqrt(a) * margin_scale, 2, 18))
    return cv2.erode(area, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))


def _fill_holes(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    return m | ~ff


# ---------------------------------------------------------------- layout
def _row_halfwidth(area: np.ndarray, cx: int, y0: int, y1: int) -> int:
    """Max symmetric half-width around cx valid for every row in [y0, y1)."""
    H, W = area.shape
    if y0 < 0 or y1 > H or y1 <= y0:
        return 0
    band = area[y0:y1] > 0
    if not band[:, cx].all():
        return 0
    allrows = band.all(0)
    left = cx - 1
    while left >= 0 and allrows[left]:
        left -= 1
    right = cx + 1
    while right < W and allrows[right]:
        right += 1
    return min(cx - left - 1, right - cx - 1)


def _break(tokens, avail, measure):
    """DP: split tokens into len(avail) lines; line i width <= avail[i]; minimise sum((1 - w_i/avail_i)^2)."""
    n, k = len(tokens), len(avail)
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[-1] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for li in range(1, k + 1):
        cap = avail[li - 1]
        for j in range(1, n + 1):
            for i in range(j - 1, -1, -1):
                if best[li - 1][i] == INF:
                    continue
                wdt = measure("".join(tokens[i:j]).strip())
                if wdt > cap:
                    break  # longer slices only get wider
                cost = best[li - 1][i] + (1 - wdt / cap) ** 2
                if cost < best[li][j]:
                    best[li][j], back[li][j] = cost, i
    if best[k][n] == INF:
        return None
    lines, j = [], n
    for li in range(k, 0, -1):
        i = back[li][j]
        lines.append("".join(tokens[i:j]).strip())
        j = i
    return lines[::-1]


def layout(text, lang, area, center, font_path, max_size, min_size=11):
    """Largest font size + line split that fits the area."""
    tokens = break_units(text, lang)
    if not tokens:
        return None
    cx, cy = center
    ys, xs = np.nonzero(area)
    if len(xs) == 0:
        return None
    for size in range(max_size, min_size - 1, -1 if max_size - min_size < 24 else -2):
        f = _font(str(font_path), size)
        # Thai stacks marks above (่ ้ ี้) and below (ุ ู) the line box -> extra leading
        line_h = int(f.line_height * (1.22 if canonical_name(lang) == "thai" else 0.95))
        meas = f.measure
        max_lines = max(1, (ys.max() - ys.min()) // line_h)
        for k in range(1, min(max_lines, len(tokens)) + 1):
            top = int(cy - k * line_h / 2)
            inset = int(line_h * 0.18)  # marks/leading may touch the edge; glyph bodies may not
            avail = [2 * _row_halfwidth(area, cx, top + i * line_h + inset, top + (i + 1) * line_h - inset) for i in range(k)]
            if min(avail) <= size:
                continue
            lines = _break(tokens, avail, meas)
            if lines:
                return size, lines, line_h, top
    return None


# ---------------------------------------------------------------- render
def render_region(img: Image.Image, text: str, lang: str, area: np.ndarray, center, style: Style,
                  font_path, max_size=None, min_size: int = 11, ss: int = 3, dry_run: bool = False):
    """Typeset one region (dry_run: only return the best-fit font size)."""
    font_path = Path(font_path)
    ys, xs = np.nonzero(area)
    if len(xs) == 0:
        return False
    max_size = max_size or int(np.clip((ys.max() - ys.min()) * 0.45, 14, 80))
    # snap the anchor inside the area (nearest high-clearance point)
    cx, cy = int(center[0]), int(center[1])
    if area[np.clip(cy, 0, area.shape[0] - 1), np.clip(cx, 0, area.shape[1] - 1)] == 0:
        dt = cv2.distanceTransform(area, cv2.DIST_L2, 3)
        cy, cx = np.unravel_index(int(dt.argmax()), dt.shape)
    res = layout(text, lang, area, (cx, cy), font_path, max_size, min_size)
    if res is None:  # retry from the area's most open point (DT peak)
        dt = cv2.distanceTransform(area, cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(dt.argmax()), dt.shape)
        if (px, py) != (cx, cy):
            res = layout(text, lang, area, (int(px), int(py)), font_path, max_size, min_size)
            if res is not None:
                cx, cy = int(px), int(py)
    if res is None:  # last resort: smaller sizes (the area is not grown: text stays inside the balloon)
        res = layout(text, lang, area, (cx, cy), font_path, max_size, min(min_size, 8))
        if res is None:
            return None if dry_run else False
    size, lines, line_h, top = res
    if dry_run:
        return size
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = size
    W, H = (x1 - x0 + 2 * pad), (y1 - y0 + 2 * pad)
    f = _font(str(font_path), size * ss)
    fill_a = np.zeros((H * ss, W * ss), np.float32)
    for i, line in enumerate(lines):
        cov, base = f.render_line(line)
        lw = f.measure(line)
        pad_l = (cov.shape[1] - lw) / 2  # render_line pads both sides equally
        lx = int(round((cx - x0 + pad) * ss - lw / 2 - pad_l))
        ly = int(round((top + i * line_h - y0 + pad) * ss + f.ascender - base))
        ya, xa = max(0, ly), max(0, lx)
        yb, xb = min(H * ss, ly + cov.shape[0]), min(W * ss, lx + cov.shape[1])
        if yb > ya and xb > xa:
            fill_a[ya:yb, xa:xb] = np.maximum(fill_a[ya:yb, xa:xb], cov[ya - ly:yb - ly, xa - lx:xb - lx])
    rgba = np.zeros((H * ss, W * ss, 4), np.float32)
    if style.stroke:
        st_a = stroke_mask(fill_a, size * style.stroke_frac * ss)
        rgba[..., :3] = np.array(style.stroke, np.float32)
        rgba[..., 3] = st_a
    a = fill_a[..., None]
    rgba[..., :3] = rgba[..., :3] * (1 - a) + np.array(style.fill, np.float32) * a
    rgba[..., 3] = np.maximum(rgba[..., 3], fill_a)
    rgba[..., 3] *= 255
    layer = Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")
    layer = layer.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.35))
    img.alpha_composite(layer, (int(x0 - pad), int(y0 - pad))) if img.mode == "RGBA" else img.paste(layer, (int(x0 - pad), int(y0 - pad)), layer)
    # footprint of what was drawn (page coords) so later regions can avoid it
    drawn = np.zeros(area.shape, np.uint8)
    la = (np.asarray(layer)[..., 3] > 8).astype(np.uint8) * 255
    ox, oy = int(x0 - pad), int(y0 - pad)
    ya, xa = max(0, oy), max(0, ox)
    yb, xb = min(area.shape[0], oy + la.shape[0]), min(area.shape[1], ox + la.shape[1])
    drawn[ya:yb, xa:xb] = la[ya - oy:yb - oy, xa - ox:xb - ox]
    return drawn
