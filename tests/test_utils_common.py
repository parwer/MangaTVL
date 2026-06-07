"""Tests for manga_translator/utils/common.py — geometry + image helpers."""
import numpy as np
from PIL import Image

from manga_translator.utils import common


def test_xyxy_xywh_roundtrip():
    box = (10, 20, 50, 80)
    x, y, w, h = common.xyxy2xywh(box)
    assert (w, h) == (40, 60)
    assert common.xywh2xyxy((x, y, w, h)) == (10, 20, 50, 80)


def test_combine_bbox_union():
    boxes = [[10, 10, 20, 20], [15, 5, 30, 18], [0, 12, 8, 25]]
    assert list(common.combine_bbox(boxes)) == [0, 5, 30, 25]


def test_combine_bbox_empty():
    assert list(common.combine_bbox([])) == [0, 0, 0, 0]


def test_inset_bbox_normal():
    assert common.inset_bbox([10, 10, 50, 50], 5) == (15, 15, 45, 45)


def test_inset_bbox_degenerate_returns_none():
    assert common.inset_bbox([10, 10, 14, 14], 5) is None


def test_inscribed_rect_square():
    # A clean axis-aligned square polygon -> LIR ~ the square itself.
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    rect = common.inscribed_rect(poly, (100, 100), padding=0)
    assert rect is not None
    x1, y1, x2, y2 = rect
    assert 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 100
    area = (x2 - x1) * (y2 - y1)
    assert area >= 0.8 * 80 * 80  # at least 80% of the square


def test_inscribed_rect_too_few_points():
    assert common.inscribed_rect([[10, 10], [20, 20]], (100, 100)) is None


def test_inscribed_rect_none_input():
    assert common.inscribed_rect(None, (100, 100)) is None


def test_refine_unit_value_type_list_passthrough():
    assert common.refine_unit_value_type([[1, 2, 3, 4]]) == [[1, 2, 3, 4]]


def test_refine_unit_value_type_from_string():
    assert common.refine_unit_value_type("[[1, 2], [3, 4]]") == [[1, 2], [3, 4]]


def test_img_pattern():
    assert common.img_pattern(Image.new("RGB", (4, 4))) == "pil"
    assert common.img_pattern(np.zeros((4, 4, 3), dtype=np.uint8)) == "cv2"
    assert common.img_pattern("data:image/jpeg;base64,abc") == "base64"
    assert common.img_pattern("http://example.com/x.png") == "url"


def test_pil2cv_cv2pil_roundtrip():
    pil = Image.new("RGB", (8, 6), (255, 0, 0))  # red
    cv = common.pil2cv(pil)
    assert cv.shape == (6, 8, 3)
    # OpenCV is BGR -> red becomes (0,0,255)
    assert tuple(int(c) for c in cv[0, 0]) == (0, 0, 255)
    back = common.cv2pil(cv)
    assert back.size == (8, 6)
    assert back.getpixel((0, 0)) == (255, 0, 0)


def test_base64_roundtrip_preserves_size():
    pil = Image.new("RGB", (32, 16), (120, 200, 80))
    uri = common.convert_img_to_base64(pil)
    assert uri.startswith("data:image/jpeg;base64,")
    decoded = common.convert_base64_to_img(uri)
    assert decoded.size == (32, 16)


def test_resize_image_downscales_keeping_ratio():
    big = Image.new("RGB", (1000, 500))
    out = common.resize_image(big, max_size=100)
    assert max(out.size) == 100
    assert out.size == (100, 50)


def test_resize_image_keeps_small_image():
    small = Image.new("RGB", (40, 30))
    out = common.resize_image(small, max_size=100)
    assert out.size == (40, 30)
