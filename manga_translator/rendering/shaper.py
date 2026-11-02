from __future__ import annotations

import functools
import threading

import cv2
import freetype
import numpy as np
import uharfbuzz as hb


class ShapedFont:
    def __init__(self, path: str, size: int):
        self.path, self.size = str(path), int(size)
        blob = hb.Blob.from_file_path(self.path)
        self.hb_font = hb.Font(hb.Face(blob))
        self.hb_font.scale = (self.size * 64, self.size * 64)
        self.ft = freetype.Face(self.path)
        self.ft.set_char_size(self.size * 64)
        m = self.ft.size
        self.ascender = m.ascender / 64.0
        self.descender = -m.descender / 64.0  # positive
        self.line_height = self.ascender + self.descender

    def shape(self, text: str):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {"kern": True, "liga": True})
        return buf.glyph_infos, buf.glyph_positions

    @functools.lru_cache(maxsize=4096)
    def measure(self, text: str) -> float:
        _, pos = self.shape(text)
        return sum(p.x_advance for p in pos) / 64.0

    def getlength(self, text: str) -> float:  # Pillow-compatible name
        return self.measure(text)

    def render_line(self, text: str) -> tuple[np.ndarray, float]:
        """Coverage mask (float32 0..1) for one line; returns (mask, baseline_y)."""
        infos, pos = self.shape(text)
        pad = int(self.size * 0.5)
        width = int(sum(p.x_advance for p in pos) / 64.0) + 2 * pad
        height = int(self.line_height) + 2 * pad
        canvas = np.zeros((height, width), np.float32)
        base = pad + self.ascender
        pen = float(pad)
        for info, p in zip(infos, pos):
            self.ft.load_glyph(info.codepoint, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_TARGET_LIGHT)
            g = self.ft.glyph
            g.render(freetype.FT_RENDER_MODE_NORMAL)
            bm = g.bitmap
            if bm.width and bm.rows:
                a = np.array(bm.buffer, np.uint8).reshape(bm.rows, bm.pitch)[:, : bm.width].astype(np.float32) / 255
                x = int(round(pen + p.x_offset / 64.0 + g.bitmap_left))
                y = int(round(base - p.y_offset / 64.0 - g.bitmap_top))
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(width, x + bm.width), min(height, y + bm.rows)
                if x1 > x0 and y1 > y0:
                    sub = a[y0 - y: y1 - y, x0 - x: x1 - x]
                    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)
            pen += p.x_advance / 64.0
        return canvas, base


@functools.lru_cache(maxsize=1024)
def _font(path: str, size: int, thread: int) -> ShapedFont:
    return ShapedFont(path, size)


def get_font(path: str, size: int) -> ShapedFont:
    """A cached font for this thread (a FreeType face renders into one shared glyph slot)."""
    return _font(path, size, threading.get_ident())


def stroke_mask(fill: np.ndarray, radius: float) -> np.ndarray:
    """Round outline: dilate the coverage by `radius` px (sub-pixel via blur)."""
    if radius <= 0:
        return fill
    r = int(np.ceil(radius))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(fill, k)
