import cv2
import numpy as np
import pytest
import torch

from manga_translator.inpainting.lama.arch import LamaGenerator, load_lama
from manga_translator.inpainting.lama_inpainter import LamaInpainter, default_model
from manga_translator.inpainting.text_eraser import TextEraser
from manga_translator.masking import refine_mask, region_glyph_masks
from manga_translator.masking.textmask import decode_blocks
from manga_translator.schemas.interface import DetectionResult, OCRResult


def ellipse(cx, cy, ax, ay, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[int(cx + ax * np.cos(a)), int(cy + ay * np.sin(a))] for a in t]


def balloon_page(poly, tone=180, text=True):
    img = np.full((300, 400, 3), tone, np.uint8)
    pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    if text:
        cv2.putText(img, "HELLO", (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def text_mask(img):
    """Stand-in for CTD + refine: dark pixels inside the balloon's inner area."""
    return ((img.max(2) < 100) * 255).astype(np.uint8)


# ---------------------------------------------------------------- LaMa architecture
def test_lama_generator_shape_and_state_dict_layout():
    g = LamaGenerator().eval()
    sd = g.state_dict()
    assert len(sd) == 989                                     # same tensors as Big-LaMa checkpoints
    assert sd["model.5.conv1.ffc.convg2g.fu.conv_layer.weight"].shape == (384, 384, 1, 1)
    assert sd["model.4.ffc.convl2g.weight"].shape == (384, 256, 3, 3)
    assert sd["model.24.weight"].shape == (512, 256, 3, 3)
    with torch.no_grad():
        y = g(torch.rand(1, 3, 64, 96), torch.zeros(1, 1, 64, 96))
    assert y.shape == (1, 3, 64, 96) and 0 <= float(y.min()) and float(y.max()) <= 1


@pytest.mark.skipif(not default_model().exists(), reason="LaMa checkpoint not installed")
def test_lama_checkpoint_loads_strict_and_fills_hole():
    g = load_lama(default_model(), "cpu")
    img = torch.full((1, 3, 64, 64), 0.8)
    m = torch.zeros(1, 1, 64, 64); m[..., 24:40, 24:40] = 1
    with torch.no_grad():
        y = g(img, m)
    assert float((y[..., 24:40, 24:40] - 0.8).abs().mean()) < 0.1   # a flat page stays flat


# ---------------------------------------------------------------- flat fill / LaMa routing
def bare_inpainter(lama=None):
    inp = LamaInpainter.__new__(LamaInpainter)
    inp.device, inp.flat_std, inp.stats, inp.ort, inp.net = "cpu", 6.0, {"flat": 0, "lama": 0}, None, None
    inp.lama = lama or (lambda bgr, mask: np.full_like(bgr, 7))
    return inp


def test_flat_surround_is_filled_with_its_colour_without_lama():
    img = np.full((100, 100, 3), 230, np.uint8)
    img[40:60, 40:60] = 0
    mask = np.zeros((100, 100), np.uint8); mask[38:62, 38:62] = 255
    inp = bare_inpainter()
    out = inp.erase(img, mask)
    assert (out == 230).all() and inp.stats == {"flat": 1, "lama": 0}


def test_busy_surround_goes_to_lama_and_only_the_hole_changes():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), np.uint8); mask[40:60, 40:60] = 255
    inp = bare_inpainter()
    out = inp.erase(img, mask)
    assert inp.stats["lama"] == 1
    assert (out[mask > 0] == 7).all() and (out[mask == 0] == img[mask == 0]).all()


def test_fast_erase_uses_telea_instead_of_lama():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), np.uint8); mask[40:60, 40:60] = 255
    inp = bare_inpainter(lama=lambda bgr, mask: (_ for _ in ()).throw(AssertionError("LaMa in fast mode")))
    out = inp.erase(img, mask, fast=True)
    assert inp.stats["telea"] == 1 and inp.stats["lama"] == 0
    assert (out[mask == 0] == img[mask == 0]).all() and not (out[mask > 0] == img[mask > 0]).all()


# ---------------------------------------------------------------- masks
def test_refine_mask_grows_to_cover_edges():
    prob = np.zeros((60, 60), np.float32); prob[20:40, 28:32] = 0.9
    m = refine_mask(np.zeros((60, 60, 3), np.uint8), prob)
    assert m[30, 30] == 255 and m[30, 25] == 255 and m[30, 34] == 255   # wider than the core
    assert not m[5, 5]


def test_decode_blocks_scales_and_suppresses():
    rows = np.array([[100, 100, 40, 20, 0.9, 0.9, 0.1],
                     [101, 100, 40, 20, 0.8, 0.9, 0.1],       # duplicate -> NMS
                     [300, 300, 40, 20, 0.2, 0.9, 0.1]], np.float32)  # low score
    out = decode_blocks(rows, 0.5, 1000, 1000)
    assert len(out) == 1 and out[0][:4] == [160, 180, 240, 220]


def test_region_masks_are_owned_by_the_smallest_region():
    big, small = ellipse(200, 150, 150, 110), ellipse(200, 150, 70, 40)
    img = balloon_page(big)
    glyph = text_mask(img)
    glyph[:, :] &= cv2.erode(cv2.fillPoly(np.zeros((300, 400), np.uint8), [np.asarray(big, np.int32)], 255),
                             np.ones((9, 9), np.uint8))
    items = [OCRResult(text="HELLO", boxes=[[140, 130, 260, 165]],
                       detection_result=DetectionResult(bbox=[50, 40, 350, 260], segmentation=big, form_type="bubble")),
             OCRResult(text="HELLO", boxes=[[140, 130, 260, 165]],
                       detection_result=DetectionResult(bbox=[130, 110, 270, 190], segmentation=small, form_type="bubble"))]
    a, b = region_glyph_masks(img, glyph, items)
    assert b.any() and not (a & b).any()                     # text inside both goes to the small one
    assert (b > 0).sum() >= (glyph > 0).sum() * 0.9


# ---------------------------------------------------------------- eraser + guard
class StubMasker:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, bgr):
        return self.fn(bgr)


def test_eraser_cleans_a_flat_balloon():
    poly = ellipse(200, 150, 150, 110)
    img = balloon_page(poly)
    inner = cv2.erode(cv2.fillPoly(np.zeros((300, 400), np.uint8), [np.asarray(poly, np.int32)], 255),
                      np.ones((15, 15), np.uint8))
    rm = cv2.dilate(text_mask(img) & inner, np.ones((5, 5), np.uint8))
    masker = StubMasker(lambda bgr: ((bgr.max(2) < 100) & (inner > 0)).astype(np.float32))
    res = TextEraser(masker, bare_inpainter()).erase(img, [rm], [DetectionResult(bbox=[50, 40, 350, 260],
                                                                                segmentation=poly, form_type="bubble")])
    assert res.reverted == []
    assert (res.clean[inner > 0] >= 250).all()               # balloon is white again
    assert (res.clean[:20] == img[:20]).all()                # outside untouched


def test_guard_restores_a_region_whose_text_survives():
    rng = np.random.default_rng(1)
    img = rng.integers(100, 200, (300, 400, 3), dtype=np.uint8)   # busy art, no balloon
    cv2.putText(img, "BOOM", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
    rm = cv2.dilate(((img.max(2) < 60) * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    keep_text = lambda bgr, mask: bgr.copy()                     # an inpainter that fails to erase
    masker = StubMasker(lambda bgr: (bgr.max(2) < 60).astype(np.float32))
    res = TextEraser(masker, bare_inpainter(keep_text)).erase(img, [rm], [DetectionResult(bbox=[90, 100, 330, 180],
                                                                                         form_type="free_text")])
    assert [i for i, _ in res.reverted] == [0]
    assert (res.clean == img).all()                          # original pixels back


def test_no_check_erases_once_without_running_the_mask_model_again():
    # CPU path: each re-check runs the mask model over the whole page (about 23 s on a CPU)
    rng = np.random.default_rng(1)
    img = rng.integers(100, 200, (300, 400, 3), dtype=np.uint8)
    cv2.putText(img, "BOOM", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
    rm = cv2.dilate(((img.max(2) < 60) * 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    calls = []
    masker = StubMasker(lambda bgr: calls.append(1) or (bgr.max(2) < 60).astype(np.float32))
    det = [DetectionResult(bbox=[90, 100, 330, 180], form_type="free_text")]
    res = TextEraser(masker, bare_inpainter(lambda bgr, mask: bgr.copy())).erase(img, [rm], det, check=False)
    assert calls == [] and res.reverted == []                # no re-check, so nothing reverted either


@pytest.mark.parametrize("k,pad", [(4, 1), (2, 0)])
def test_conv_transpose_as_phase_convs_gives_the_same_output(k, pad):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
    from ctd_subpixel import phase_convs
    torch.manual_seed(0)
    x, w = torch.randn(1, 5, 6, 7), torch.randn(5, 3, k, k)
    ref = torch.nn.functional.conv_transpose2d(x, w, stride=2, padding=pad)
    out = torch.zeros_like(ref)
    for a, b, wc, (top, left, bottom, right) in phase_convs(w.numpy(), k, pad):
        xp = torch.nn.functional.pad(x, (left, right, top, bottom))
        out[:, :, a::2, b::2] = torch.nn.functional.conv2d(xp, torch.from_numpy(wc))
    assert torch.allclose(out, ref, atol=1e-5)
