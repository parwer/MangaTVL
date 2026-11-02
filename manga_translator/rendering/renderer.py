from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from ..schemas.interface import TranslationResult
from ..utils.common import cv2pil, pil2cv
from .fonts import bold_font_path, font_path as resolve_font_path
from .typeset import balloon_area, detect_style, render_region


def _poly_mask(polygon, shape, grow: int = 0) -> np.ndarray | None:
    if not polygon or len(polygon) < 3:
        return None
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 255)
    if grow > 0:
        m = cv2.dilate(m, np.ones((grow, grow), np.uint8))
    elif grow < 0:
        m = cv2.erode(m, np.ones((-grow, -grow), np.uint8))
    return m


def _others(masks, keep):
    """f(i) -> union of the masks of the regions in ``keep`` other than i (counts computed once)."""
    claims = np.zeros(masks[0].shape, np.int16) if masks else None
    for j in keep:
        claims += masks[j] > 0
    keep = set(keep)

    def union_except(i):
        own = (masks[i] > 0) if i in keep else 0
        return ((claims - own) > 0).astype(np.uint8) * 255
    return union_except


def _extent(mask):
    """(x0, y0, x1, y1) of the non-zero pixels, or None."""
    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, x + w - 1, y + h - 1) if w else None


def estimate_glyph_masks(original_bgr, clean_bgr, items, thresh: int = 40) -> list[np.ndarray]:
    """Original lettering per region ~= pixels the inpainter changed, inside the region."""
    H, W = clean_bgr.shape[:2]
    changed = ((cv2.absdiff(original_bgr, clean_bgr).max(2) > thresh) * 255).astype(np.uint8)
    out = []
    for it in items:
        det = it.ocr_result.detection_result
        lim = _poly_mask(det.segmentation, (H, W), grow=7)
        if lim is None:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            lim = np.zeros((H, W), np.uint8)
            lim[max(0, y1):y2, max(0, x1):x2] = 255
        out.append(changed & lim)
    return out


def _orig_size(gm, text) -> float | None:
    """Rough original lettering size: sqrt(ink area per character)."""
    n = len("".join((text or "").split()))
    if gm is None or not gm.any() or n == 0:
        return None
    return float(np.sqrt(np.count_nonzero(gm) / n))


class TextRenderer:
    def __init__(self,
                 font_path=None,      # fallback TTF when the font registry has no match
                 max_font_size=80,
                 min_font_size=11,
                 lang="thai",         # default target language (overridable per render() call)
                 text_scale=1.0,      # >1 = allow bigger text, <1 = smaller (scales the size caps)
                 supersample=3,       # glyphs are drawn at this scale and downscaled for smooth edges
                 ):
        self.font_path = font_path
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.lang = lang
        self.text_scale = text_scale
        self.supersample = max(1, int(supersample))
        self.last_layout: list[dict] = []  # per input of the last render() call (eval / debugging)

    @staticmethod
    def _clamp_text_scale(value):
        try:
            return float(np.clip(float(value), 0.5, 2.0))
        except (TypeError, ValueError):
            return 1.0

    def render(self, image, inputs: list[TranslationResult], lang=None, font=None, text_scale=None, *,
               original=None, glyph_masks: list[np.ndarray] | None = None, keep_words=()) -> Image.Image:
        """Letter ``inputs`` onto ``image`` (the cleaned page; PIL or BGR array)."""
        self.last_layout = [{"font_size": None, "drawn": False} for _ in (inputs or [])]
        if not inputs:
            return cv2pil(image) if isinstance(image, np.ndarray) else image
        lang = lang or self.lang
        fp = resolve_font_path(font, lang=lang, fallback=self.font_path)
        bold = bold_font_path(font, lang=lang)
        scale = self._clamp_text_scale(text_scale if text_scale is not None else self.text_scale)
        clean = image if isinstance(image, np.ndarray) else pil2cv(image)
        orig = original if original is None or isinstance(original, np.ndarray) else pil2cv(original)
        H, W = clean.shape[:2]
        texts = [(it.translated_text or "").strip() for it in inputs]
        if glyph_masks is None:
            glyph_masks = (estimate_glyph_masks(orig, clean, inputs) if orig is not None
                           else [np.zeros((H, W), np.uint8) for _ in inputs])
        gms = [g.copy() for g in glyph_masks]
        dets = [it.ocr_result.detection_result for it in inputs]
        keep = [i for i, t in enumerate(texts) if t]
        out = cv2pil(clean).convert("RGBA")
        ss = self.supersample
        max_cap = int(self.max_font_size * scale)
        min_size = self.min_font_size
        polys = {i: _poly_mask(dets[i].segmentation, (H, W), grow=-3) for i in keep}
        union_except = _others(gms, keep) if gms else None

        def build_area(i, ms=1.0):
            d, gm = dets[i], gms[i]
            area = balloon_area(clean, d.bbox, d.segmentation, gm, margin_scale=ms)
            g, a = _extent(gm), _extent(area)
            if g:  # the original lettering's footprint is always writable space
                if not a or (a[2] - a[0]) < 0.8 * (g[2] - g[0]) or (a[3] - a[1]) < 0.8 * (g[3] - g[1]):
                    pts = cv2.findNonZero(cv2.dilate(gm, np.ones((7, 7), np.uint8)))
                    cv2.fillConvexPoly(area, cv2.convexHull(pts), 255)
            block = cv2.dilate(union_except(i), np.ones((11, 11), np.uint8))
            if gm.any():  # own footprint stays writable even where it touches a neighbour's
                own = np.zeros_like(gm)
                cv2.fillConvexPoly(own, cv2.convexHull(cv2.findNonZero(gm)), 255)
                block &= ~own
            area &= ~block
            if polys[i] is not None:  # never outside the balloon
                area &= polys[i]
            return area

        def fit(i, area, center, style, max_size):
            return render_region(out, texts[i], lang, area, center, style, font_path=fp,
                                 max_size=max_size, min_size=min_size, ss=ss, dry_run=True, keep_words=keep_words)

        plans, fits = {}, []
        for i in keep:
            area = build_area(i)
            if not area.any():
                continue
            gm = gms[i]
            src = gm if gm.any() else area
            x0, y0, x1, y1 = _extent(src)
            ys, xs = np.nonzero(src[y0:y1 + 1, x0:x1 + 1])
            center = (int(np.median(xs + x0)), int((ys.min() + ys.max()) / 2 + y0))
            style = detect_style(orig if orig is not None else clean, gm, clean, area)
            f = fit(i, area, center, style, max_cap)
            ref = float(np.median(fits)) if len(fits) >= 3 else None
            if f is None or (ref and f < 0.75 * ref):  # cramped: thinner inner margin
                a2 = build_area(i, ms=0.4)
                f2 = fit(i, a2, center, style, max_cap)
                if f2 is not None and (f is None or f2 > f):
                    area, f = a2, f2
            if f is not None:
                fits.append(f)
            plans[i] = [area, center, style, f, _orig_size(gm, inputs[i].ocr_result.text)]

        bfits = [p[3] for i, p in plans.items() if p[3] and dets[i].form_type != "free_text"]
        if bfits:
            self._grow_free_text(plans, dets, union_except, clean, float(np.median(bfits)),
                                 lambda i, a, c, s: fit(i, a, c, s, max_cap))

        # one dialogue size per page; emphasis relative to the original lettering size
        cap = int(np.median(bfits) * 1.12) if bfits else None
        ghs = [p[4] for p in plans.values() if p[4]]
        med_gh = float(np.median(ghs)) if ghs else None
        occupied = np.zeros((H, W), np.uint8)
        for i in sorted(plans, key=lambda k: int(np.count_nonzero(plans[k][0]))):  # small first
            area, center, style, f, gh = plans[i]
            area = area & ~cv2.dilate(occupied, np.ones((9, 9), np.uint8))
            ratio = float(gh / med_gh) if (gh and med_gh) else 1.0
            c = int(cap * float(np.clip(ratio, 1.0, 2.5))) if cap else None
            limit = min(x for x in (f, c, max_cap) if x) if f else max_cap
            face = bold if (bold and ratio >= 1.4) else fp
            drawn = render_region(out, texts[i], lang, area, center, style, font_path=face,
                                  max_size=limit, min_size=min_size, ss=ss, keep_words=keep_words)
            if drawn is False and face != fp:  # the bold face may not fit at the measured size
                face = fp
                drawn = render_region(out, texts[i], lang, area, center, style, font_path=fp,
                                      max_size=limit, min_size=min_size, ss=ss, keep_words=keep_words)
            ok = isinstance(drawn, np.ndarray)
            if ok:
                occupied |= drawn
            self.last_layout[i] = {"font_size": limit if ok else None, "fit": f, "cap": c,
                                   "drawn": ok, "bold": ok and face != fp}
        return out.convert("RGB")

    @staticmethod
    def _grow_free_text(plans, dets, union_except, clean, med, fit):
        """Free text squeezed far below dialogue size grows into the free space around it."""
        H, W = clean.shape[:2]
        balloons = np.zeros((H, W), np.uint8)
        for d in dets:
            if d.form_type != "free_text" and d.segmentation:
                cv2.fillPoly(balloons, [np.asarray(d.segmentation, np.int32).reshape(-1, 1, 2)], 255)
        balloons = cv2.dilate(balloons, np.ones((9, 9), np.uint8))
        # solid line art: strong contrast against the local background (light or dark panels);
        # the small median removes screentone dots so a tone background stays writable
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        g5 = cv2.medianBlur(gray, 5).astype(np.int16)
        local = cv2.medianBlur(gray, 31).astype(np.int16)
        balloons |= cv2.dilate(((np.abs(g5 - local) > 60) * 255).astype(np.uint8), np.ones((9, 9), np.uint8))
        for i, p in plans.items():
            area, center, style, f, gh = p
            if dets[i].form_type != "free_text" or (f and f >= 0.7 * med):
                continue
            x1, y1, x2, y2 = (int(v) for v in dets[i].bbox)
            gx, gy = int(0.35 * (x2 - x1)) + 8, int(0.25 * (y2 - y1)) + 8
            grown = np.zeros_like(area)
            grown[max(0, y1 - gy):y2 + gy, max(0, x1 - gx):x2 + gx] = 255
            grown &= ~balloons & ~cv2.dilate(union_except(i), np.ones((11, 11), np.uint8))
            grown |= area
            f3 = fit(i, grown, center, style)
            if f3 is not None and (f is None or f3 > f):
                f3 = min(f3, max(f or 0, int(0.8 * med)))  # an aside stays smaller than dialogue
                if style.stroke is None:  # now next to tone / line art: a paper-coloured halo
                    halo = (255, 255, 255) if sum(style.fill) < 380 else (0, 0, 0)
                    style = replace(style, stroke=halo, stroke_frac=0.09)
                plans[i] = [grown, center, style, f3, gh]
