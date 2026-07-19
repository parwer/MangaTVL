"""Reference-free metrics for pipeline stages 3 / 4 / 5.

Every function takes an ``eval.harness.Capture`` and returns a dict with two
keys: ``per`` (one record per bubble) and ``summary`` (page-level aggregate).
Nothing here needs ground-truth labels.
"""

import numpy as np
import cv2

from manga_translator.utils.common import pil2cv

# ---- Stage 5 structural fit reuses the renderer's *real* logic -------------
from manga_translator.rendering.renderer import (
    _stroke_for_font,
    ELLIPSIS,
    ABSOLUTE_MIN_FONT_SIZE,
)
from manga_translator.rendering.extract_text_box import extract_text_box
from manga_translator.rendering.fonts import font_path as resolve_font_path
from PIL import ImageDraw


# ===========================================================================
# helpers
# ===========================================================================

def _gray(pil_img) -> np.ndarray:
    return cv2.cvtColor(pil2cv(pil_img), cv2.COLOR_BGR2GRAY)


def _poly_mask(poly, shape, erode_px: int = 0) -> np.ndarray | None:
    """Filled (optionally eroded) mask for one polygon. Returns None if degenerate."""
    if not poly:
        return None
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        mask = cv2.erode(mask, k)
    return mask


def _poly_diag(poly) -> float:
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return float(np.hypot(w, h))


def _thai_ratio(text: str) -> float:
    """Fraction of non-space characters that are in the Thai Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    thai = sum(1 for c in chars if "฀" <= c <= "๿")
    return thai / len(chars)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# ===========================================================================
# Stage 3 — Translation (reference-free)
# ===========================================================================

def translation_metrics(cap) -> dict:
    """Signals: fallback rate (translator returned the original text), whether
    the output is actually in the target script, and length blow-up. These
    don't judge *meaning* (that needs a reference or an LLM judge) but they
    reliably catch the common failure modes: API failure -> fallback, and
    'model echoed source / refused' -> low target-script ratio."""
    per = []
    for i, tr in enumerate(cap.translation):
        src = (tr.ocr_result.text or "").strip()
        out = (tr.translated_text or "").strip()
        if not src:
            continue  # empty OCR — nothing to translate, don't penalise
        fallback = out == src                     # translator fell back to original
        thai = _thai_ratio(out)
        per.append({
            "bubble_index": i,
            "source": src,
            "translated": out,
            "fallback": fallback,
            "target_script_ratio": round(thai, 3),
            "untranslated": (not fallback) and thai < 0.2 and len(out) > 0,
            "empty_output": len(out) == 0,
            "len_ratio": round(len(out) / max(1, len(src)), 3),
        })

    n = len(per)
    summary = {
        "bubbles": n,
        "fallback_rate": round(sum(p["fallback"] for p in per) / n, 3) if n else None,
        "untranslated_rate": round(sum(p["untranslated"] for p in per) / n, 3) if n else None,
        "empty_output_rate": round(sum(p["empty_output"] for p in per) / n, 3) if n else None,
        "mean_target_script_ratio": _mean([p["target_script_ratio"] for p in per]),
        "mean_len_ratio": _mean([p["len_ratio"] for p in per]),
    }
    return {"per": per, "summary": summary}


# ===========================================================================
# Stage 4 — Inpainting (polygon-interior residual)
# ===========================================================================

def inpainting_metrics(cap, erode_frac: float = 0.06, ink_delta: int = 45) -> dict:
    """Did inpainting actually clean the inside of each bubble?

    Measured inside the segmentation polygon (eroded inward to exclude the dark
    bubble outline). A clean interior is uniform, so:
      * interior_std_after   low  -> flat / clean fill
      * std_reduction        high -> a lot of text variance was removed
      * ink_residual_after   low  -> few dark (text-like) pixels remain

    Caveat: the dark-ink ratio assumes dark text on a light bubble; for dark or
    busy backgrounds rely on ``std_reduction`` (background-colour agnostic).
    Bubbles with no polygon (detect-only) are skipped and reported as coverage.
    """
    orig = _gray(cap.original)
    inp = _gray(cap.inpainted) if cap.inpainted is not None else None
    per = []
    no_poly = 0
    if inp is None:
        return {"per": [], "summary": {"bubbles": 0, "note": "no inpainted image"}}

    for i, det in enumerate(cap.detection):
        poly = getattr(det, "segmentation", None)
        if not poly:
            no_poly += 1
            continue
        erode_px = max(2, int(_poly_diag(poly) * erode_frac))
        mask = _poly_mask(poly, orig.shape, erode_px=erode_px)
        if mask is None or not mask.any():
            no_poly += 1
            continue
        sel = mask > 0
        before = orig[sel].astype(np.float64)
        after = inp[sel].astype(np.float64)
        std_b, std_a = float(before.std()), float(after.std())
        bg = float(np.median(after))
        ink_b = float(np.mean(before < (np.median(before) - ink_delta)))
        ink_a = float(np.mean(after < (bg - ink_delta)))
        per.append({
            "bubble_index": i,
            "interior_std_before": round(std_b, 2),
            "interior_std_after": round(std_a, 2),
            "std_reduction": round((std_b - std_a) / std_b, 3) if std_b > 1e-6 else None,
            "ink_residual_before": round(ink_b, 4),
            "ink_residual_after": round(ink_a, 4),
            "ink_removed": round((ink_b - ink_a) / ink_b, 3) if ink_b > 1e-6 else None,
        })

    n = len(per)
    summary = {
        "bubbles": n,
        "bubbles_without_polygon": no_poly,
        "mean_interior_std_after": _mean([p["interior_std_after"] for p in per]),
        "mean_std_reduction": _mean([p["std_reduction"] for p in per]),
        "mean_ink_residual_after": _mean([p["ink_residual_after"] for p in per]),
        "mean_ink_removed": _mean([p["ink_removed"] for p in per]),
    }
    return {"per": per, "summary": summary}


# ===========================================================================
# Stage 5 — Rendering
# ===========================================================================

def _text_mask(cap, diff_thresh: int = 30) -> np.ndarray | None:
    """Pixels that changed between the inpainted page and the rendered page =
    the drawn glyphs + stroke."""
    if cap.inpainted is None or cap.rendered is None:
        return None
    a = _gray(cap.inpainted).astype(np.int16)
    b = _gray(cap.rendered).astype(np.int16)
    return (np.abs(a - b) > diff_thresh).astype(np.uint8)


def rendering_spill_metrics(cap, dilate_px: int = 2, diff_thresh: int = 30) -> dict:
    """Does the rendered text stay inside the bubble's actual shape?

    Localises each bubble's drawn text by its detection bbox, then splits those
    pixels by the segmentation polygon (dilated by a small tolerance):
      * spill_ratio = text pixels outside the polygon / total text pixels
      * fill_ratio  = text area inside the polygon / polygon area
    """
    tmask = _text_mask(cap, diff_thresh=diff_thresh)
    if tmask is None:
        return {"per": [], "summary": {"bubbles": 0, "note": "missing images"}}
    h, w = tmask.shape
    per = []
    no_poly = 0
    for i, det in enumerate(cap.detection):
        poly = getattr(det, "segmentation", None)
        if not poly:
            no_poly += 1
            continue
        pmask = _poly_mask(poly, (h, w))
        if pmask is None:
            no_poly += 1
            continue
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            pmask_tol = cv2.dilate(pmask, k)
        else:
            pmask_tol = pmask
        # restrict to this bubble's bbox so neighbouring bubbles' text isn't counted
        bx = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(bx, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)
        local = (tmask > 0) & (bx > 0)
        total = int(local.sum())
        if total == 0:
            continue  # nothing rendered for this bubble (e.g. empty translation)
        inside = int((local & (pmask_tol > 0)).sum())
        poly_area = int((pmask > 0).sum())
        spill = (total - inside) / total
        per.append({
            "bubble_index": i,
            "text_pixels": total,
            "spill_ratio": round(spill, 3),
            "spills": bool(spill > 0.05),     # text noticeably crossing the bubble edge
            "fill_ratio": round(inside / poly_area, 3) if poly_area else None,
        })

    n = len(per)
    summary = {
        "bubbles": n,
        "bubbles_without_polygon": no_poly,
        "mean_spill_ratio": _mean([p["spill_ratio"] for p in per]),
        "spill_rate": round(sum(p["spill_ratio"] > 0.05 for p in per) / n, 3) if n else None,
        "mean_fill_ratio": _mean([p["fill_ratio"] for p in per]),
    }
    return {"per": per, "summary": summary}


def rendering_fit_metrics(cap, renderer, lang: str) -> dict:
    """Structural fit, reusing the renderer's own fit search per bubble:
    the chosen font size, and whether the text had to be truncated with an
    ellipsis at the absolute-minimum size (the renderer's last-resort fallback,
    i.e. the text genuinely didn't fit)."""
    if cap.rendered is None:
        return {"per": [], "summary": {"bubbles": 0}}
    img = cap.rendered
    w, h = img.size
    draw = ImageDraw.Draw(img)  # measurement only — we never draw here
    fp = resolve_font_path(None, lang=lang, fallback=renderer.font_path)
    scale = renderer._clamp_text_scale(renderer.text_scale)
    pad = _stroke_for_font(renderer.max_font_size) + 2

    per = []
    for i, item in enumerate(cap.translation):
        text = item.translated_text if item.translated_text else item.ocr_result.text
        if not (text or "").strip():
            continue
        det = item.ocr_result.detection_result
        box = renderer._compute_box(det, (h, w), pad, scale) or tuple(
            int(v) for v in extract_text_box(img, item)
        )
        prepared, _spaced = renderer._prepare_text(text, lang)
        wrap_text, fs = renderer.wrap_extraction(draw, prepared, box, det.bbox, fp)
        truncated = (ELLIPSIS in wrap_text) and fs <= ABSOLUTE_MIN_FONT_SIZE
        per.append({
            "bubble_index": i,
            "font_size": fs,
            "lines": wrap_text.count("\n") + 1,
            "truncated": bool(truncated),
        })

    n = len(per)
    summary = {
        "bubbles": n,
        "truncate_rate": round(sum(p["truncated"] for p in per) / n, 3) if n else None,
        "mean_font_size": _mean([p["font_size"] for p in per]),
        "min_font_size": min((p["font_size"] for p in per), default=None),
        "mean_lines": _mean([p["lines"] for p in per]),
    }
    return {"per": per, "summary": summary}
