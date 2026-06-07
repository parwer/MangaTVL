"""Tests for ocr/ocr_engine.OCREngine.clean_poly — erasing everything outside
the bubble polygon to white before OCR."""
import numpy as np
from PIL import Image

from manga_translator.ocr.ocr_engine import OCREngine


def _black(w=10, h=10):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def test_clean_poly_fills_outside_white_keeps_inside():
    engine = OCREngine()
    img = _black(10, 10)
    poly = [[2, 2], [8, 2], [8, 8], [2, 8]]  # inner square, offset (0,0)
    out = np.array(engine.clean_poly(img, poly, offset=(0, 0)))
    # outside the polygon -> white
    assert tuple(out[0, 0]) == (255, 255, 255)
    # inside the polygon -> unchanged (black)
    assert tuple(out[5, 5]) == (0, 0, 0)


def test_clean_poly_none_returns_unchanged():
    engine = OCREngine()
    img = _black(10, 10)
    out = engine.clean_poly(img, None)
    assert np.array_equal(np.array(out), np.array(img))


def test_clean_poly_applies_offset():
    engine = OCREngine()
    img = _black(10, 10)
    # polygon given in full-image coords; crop offset shifts it into local frame
    poly = [[12, 12], [18, 12], [18, 18], [12, 18]]
    out = np.array(engine.clean_poly(img, poly, offset=(10, 10)))
    assert tuple(out[0, 0]) == (255, 255, 255)
    assert tuple(out[5, 5]) == (0, 0, 0)
