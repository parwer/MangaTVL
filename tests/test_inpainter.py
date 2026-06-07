"""Tests for inpainting/inpainter.InpainterBase — box expansion, box-based
masking, and PanelCleaner-style safe polygon mask selection."""
import numpy as np

from manga_translator.inpainting.inpainter import InpainterBase
from manga_translator.schemas.interface import OCRResult, DetectionResult


class DummyInpainter(InpainterBase):
    def _inpaint(self, image, masks, **kwargs):
        return image


def _ocr(boxes, segmentation=None):
    return OCRResult(
        text="x",
        boxes=boxes,
        detection_result=DetectionResult(bbox=[0, 0, 1, 1], form_type="bubble",
                                         segmentation=segmentation),
    )


def test_expand_box_grows_and_clips():
    di = DummyInpainter()
    assert di.expand_box([5, 5, 10, 10], (100, 100), expand_margin=2) == (3, 3, 12, 12)
    # clipped at image edges (0 and max)
    assert di.expand_box([1, 1, 99, 99], (100, 100), expand_margin=5) == (0, 0, 100, 100)


def test_masking_draws_boxes():
    di = DummyInpainter()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = di._masking(img, [_ocr([[10, 10, 20, 20]])], expand_margin=0)
    assert mask.shape == (50, 50)
    assert mask[15, 15] == 255   # inside the box
    assert mask[0, 0] == 0       # outside


def test_pick_safe_polygon_mask_uniform_interior_passes():
    di = DummyInpainter()
    gray = np.full((100, 100), 255, dtype=np.uint8)  # uniform -> border std 0
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    mask = di._pick_safe_polygon_mask(gray, poly)
    assert mask is not None
    assert mask.shape == (100, 100)
    assert mask.any()


def test_pick_safe_polygon_mask_noisy_border_rejected():
    di = DummyInpainter()
    rng = np.random.RandomState(0)
    gray = rng.randint(0, 256, size=(100, 100)).astype(np.uint8)  # high variance
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    # every candidate's border std >> max_std=25 -> None ("too risky, skip")
    assert di._pick_safe_polygon_mask(gray, poly) is None


def test_pick_safe_polygon_mask_stays_within_polygon_bbox():
    di = DummyInpainter()
    gray = np.full((100, 100), 255, dtype=np.uint8)
    poly = [[10, 10], [90, 10], [90, 90], [10, 90]]
    mask = di._pick_safe_polygon_mask(gray, poly)
    assert mask is not None
    # the chosen mask must not leak outside the polygon's bounding box
    ys, xs = np.where(mask > 0)
    assert xs.min() >= 10 and xs.max() <= 90
    assert ys.min() >= 10 and ys.max() <= 90
