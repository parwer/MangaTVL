import textwrap
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from ..utils.common import cv2pil, inscribed_rect, inset_bbox

from ..schemas.interface import TranslationResult
from .extract_text_box import extract_text_box
from .tokenize import tokenize, is_space_delimited
from .fonts import font_path as resolve_font_path
import functools
import re

DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "THSarabunNew.ttf"

ABSOLUTE_MIN_FONT_SIZE = 10
ELLIPSIS = "…"
# Private-use placeholder standing in for original spaces while wrapping spaceless
# scripts (Thai/CJK), so they survive the strip-token-boundary-spaces step.
SPACE_SENTINEL = ""


def _stroke_for_font(font_size):
    return max(1, font_size // 12)


@functools.lru_cache(maxsize=128)
def _load_font(path, size):
    """Cached TTF loader — the fit search reloads the same (font, size) many
    times, so caching avoids re-reading the file on every step. Falls back to
    Pillow's default font when the path can't be loaded."""
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except Exception:
        print(f"Failed to load font from {path}. Using default font.")
        return ImageFont.load_default()


class TextRenderer:
    def __init__(self,
                 font_path=DEFAULT_FONT_PATH,
                 tokenizer=None,   # optional override: callable(text, lang) -> list[str]; None = language-aware default
                 max_font_size=100,
                 min_font_size=18,
                 lang="thai",      # default target language for line breaking (overridable per render() call)
                 text_scale=1.2,   # inflate the fit area (LIR -> bbox) so text fills more of the bubble; >1 = bigger
                 supersample=2,    # draw text at this scale then downscale (LANCZOS) for crisper edges; 1 = off
                 display_mode="horizontal" # "vertical" or "horizontal" or "mixed"
                 ):
        self.font_path = font_path
        self.tokenizer = tokenizer
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.lang = lang
        self.text_scale = text_scale
        self.supersample = max(1, int(supersample))
        self.display_mode = display_mode

    def render(self, image, inputs: list[TranslationResult], lang=None, font=None, text_scale=None):
        if inputs is None or len(inputs) == 0:
            print("No inputs provided for rendering. Returning original image.")
            return image

        lang = lang or self.lang  # target language drives how lines are broken
        # Pick the font: explicit `font` key -> per-language default -> this
        # renderer's font_path. Resolved once per call and reused for every bubble.
        font_path = resolve_font_path(font, lang=lang, fallback=self.font_path)
        scale = self._clamp_text_scale(text_scale if text_scale is not None else self.text_scale)
        ss = self.supersample
        image_copy = cv2pil(image.copy())
        w, h = image_copy.size
        # `measure_draw` measures text in original coords (for the fit search);
        # `text_draw` is where glyphs land. With supersampling the glyphs are drawn
        # on a transparent overlay at ss× and downscaled, so artwork stays untouched
        # and only the (anti-aliased) text gets the crisper high-res treatment.
        measure_draw = ImageDraw.Draw(image_copy)
        if ss > 1:
            overlay = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(overlay)
        else:
            overlay = None
            text_draw = measure_draw
        # Inset by the max possible stroke so the largest font's stroke can't poke out of the bubble.
        pad = _stroke_for_font(self.max_font_size) + 2

        for item in inputs:
            text = item.translated_text if item.translated_text else item.ocr_result.text
            det = item.ocr_result.detection_result
            box = self._compute_box(det, (h, w), pad, scale) or tuple(int(v) for v in extract_text_box(image_copy, item))
            self._render_single(measure_draw, text_draw, text, box, det.bbox, lang, font_path, ss)

        if overlay is not None:
            overlay = overlay.resize((w, h), Image.LANCZOS)
            image_copy.paste(overlay, (0, 0), overlay)

        return image_copy

    @staticmethod
    def _clamp_text_scale(value):
        try:
            return min(2.0, max(0.5, float(value)))
        except (TypeError, ValueError):
            return 1.0

    def _compute_box(self, det, image_shape, pad, text_scale=1.0):
        """Render area priority: polygon LIR → det.bbox inset → None (caller falls back).
        When ``text_scale`` > 1 the LIR is grown toward the detection bbox so text
        fills more of the bubble (the strict LIR leaves a lot of a round bubble empty)."""
        poly = getattr(det, "segmentation", None)
        box = None
        if poly:
            box = inscribed_rect(poly, image_shape, padding=pad)
        if box is None and det.bbox:
            box = inset_bbox(det.bbox, pad)
        if box is None:
            return None
        if text_scale and text_scale != 1.0 and det.bbox:
            box = self._inflate_box(box, det.bbox, image_shape, pad, text_scale)
        return box

    def _inflate_box(self, box, bbox, image_shape, pad, scale):
        """Grow ``box`` by ``scale`` around its centre, capped to the detection
        bbox (inset by ``pad``) and the image — so text gets bigger but never
        spills past the bubble's bounding box or the page edge."""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hw, hh = (x2 - x1) / 2.0 * scale, (y2 - y1) / 2.0 * scale
        nx1, ny1, nx2, ny2 = cx - hw, cy - hh, cx + hw, cy + hh

        cap = inset_bbox(bbox, pad) or tuple(bbox)
        cx1, cy1, cx2, cy2 = cap
        h, w = image_shape[:2]
        nx1 = int(max(nx1, cx1, 0))
        ny1 = int(max(ny1, cy1, 0))
        nx2 = int(min(nx2, cx2, w))
        ny2 = int(min(ny2, cy2, h))
        if nx2 <= nx1 or ny2 <= ny1:
            return box
        return (nx1, ny1, nx2, ny2)

    def _render_single(self, measure_draw, text_draw, text, box, det_box, lang, font_path, ss=1):
        prepared, spaced = self._prepare_text(text, lang)

        # Fit search runs in original coords using `measure_draw`.
        wrap_text, font_size = self.wrap_extraction(measure_draw, prepared, box, det_box, font_path)
        wrap_text = self._finalize_text(wrap_text, spaced)

        # Draw at ss× on `text_draw`: load the font and stroke at device scale,
        # measure at that scale, then centre within the box scaled to device space.
        dfont = _load_font(font_path, max(1, int(round(font_size * ss))))
        stroke = max(1, int(round(_stroke_for_font(font_size) * ss)))
        try:
            text_bbox = text_draw.multiline_textbbox((0, 0), wrap_text, font=dfont, align='center', stroke_width=stroke)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            text_width, text_height = text_draw.multiline_textsize(wrap_text, font=dfont)

        x1, y1, x2, y2 = box
        box_width = (x2 - x1) * ss
        box_height = (y2 - y1) * ss

        x = x1 * ss + (box_width - text_width) / 2
        y = y1 * ss + (box_height - text_height) / 2

        text_draw.multiline_text((x, y), wrap_text, font=dfont, fill=(0, 0, 0), align='center',
                                 stroke_width=stroke, stroke_fill=(255, 255, 255))
        return


    def wrap_extraction(self, draw, tokenized_text, box, det_box, font_path):
        # `tokenized_text` already has break points marked by spaces (see _prepare_text).
        target_width = box[2] - box[0]
        target_height = box[3] - box[1]

        low = self.min_font_size
        high = self.max_font_size
        best_font_size = low
        best_wrap_text = None

        while low <= high:
            mid = (low + high) // 2
            fits, wrap_text = self._fits_in_box(draw, mid, tokenized_text, target_width, target_height, det_box, font_path)
            if fits:
                best_font_size = mid
                best_wrap_text = wrap_text
                low = mid + 1
            else:
                high = mid - 1

        if best_wrap_text is not None:
            return best_wrap_text, best_font_size

        # Fallback 1: shrink below min_font_size before giving up.
        for fs in range(self.min_font_size - 1, ABSOLUTE_MIN_FONT_SIZE - 1, -1):
            fits, wrap_text = self._fits_in_box(draw, fs, tokenized_text, target_width, target_height, det_box, font_path)
            if fits:
                return wrap_text, fs

        # Fallback 2: truncate with ellipsis at absolute min — guarantees no overflow.
        truncated = self._truncate_to_fit(draw, tokenized_text, target_width, target_height,
                                          ABSOLUTE_MIN_FONT_SIZE, det_box, font_path)
        return truncated, ABSOLUTE_MIN_FONT_SIZE


    def _tokenize(self, text, lang):
        """Break-unit tokens for `text` in target `lang`. Uses a caller-supplied
        tokenizer override if set, else the language-aware default."""
        if self.tokenizer is not None:
            return self.tokenizer(text, lang)
        return tokenize(text, lang)

    def _prepare_text(self, text, lang):
        """Turn `text` into a string whose spaces mark allowed line-break points,
        ready for textwrap. Returns (prepared, spaced):
        - spaced=True (Latin/Korean/...): spaces are real word separators; tokens
          are the words, joined by spaces and kept as-is on output.
        - spaced=False (Thai/CJK): original spaces are protected with a sentinel,
          tokens (words/chars) are joined by spaces for wrapping, and those
          token-boundary spaces are stripped on output (originals restored)."""
        if is_space_delimited(lang):
            return " ".join(self._tokenize(text, lang)), True
        protected = text.replace(" ", SPACE_SENTINEL)
        return " ".join(self._tokenize(protected, lang)), False

    def _finalize_text(self, wrap_text, spaced):
        """Undo the break-point spacing from _prepare_text after wrapping."""
        if spaced:
            return wrap_text
        return wrap_text.replace(" ", "").replace(SPACE_SENTINEL, " ")

    def _fits_in_box(self, draw, font_size, tokenized_text, target_width, target_height, det_box, font_path):
        font = _load_font(font_path, font_size)

        stroke = _stroke_for_font(font_size)
        wrap_width = len(tokenized_text)

        while wrap_width > 0:
            wrap_lines = textwrap.wrap(tokenized_text, width=wrap_width, break_long_words=False)
            wrap_text = "\n".join(wrap_lines)

            try:
                text_bbox = draw.multiline_textbbox((0, 0), wrap_text, font=font, align='center', stroke_width=stroke)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_width, text_height = draw.multiline_textsize(wrap_text, font=font)

            if text_width <= target_width and text_height <= target_height:
                return True, wrap_text

            wrap_width -= 1

        return False, None

    def _truncate_to_fit(self, draw, tokenized_text, target_width, target_height, font_size, det_box, font_path):
        tokens = [t for t in tokenized_text.split(' ') if t]
        if not tokens:
            return ELLIPSIS

        lo, hi = 1, len(tokens)
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(tokens[:mid]) + " " + ELLIPSIS
            fits, wrap_text = self._fits_in_box(draw, font_size, candidate, target_width, target_height, det_box, font_path)
            if fits:
                best = wrap_text
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best is not None else ELLIPSIS
