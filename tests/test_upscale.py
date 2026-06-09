"""Tests for rendering.upscale — LANCZOS path + graceful Real-ESRGAN fallback."""
from PIL import Image

from manga_translator.rendering.upscale import upscale_image, MAX_FACTOR


def _img(w=40, h=30):
    return Image.new("RGB", (w, h), (123, 200, 50))


def test_factor_one_or_less_is_noop():
    im = _img()
    assert upscale_image(im, 1.0) is im
    assert upscale_image(im, 0.5) is im
    assert upscale_image(im, None) is im


def test_lanczos_doubles_size():
    out = upscale_image(_img(40, 30), 2.0, method="lanczos")
    assert out.size == (80, 60)


def test_factor_is_clamped():
    out = upscale_image(_img(10, 10), 99, method="lanczos")
    assert out.size == (int(10 * MAX_FACTOR), int(10 * MAX_FACTOR))


def test_non_integer_factor():
    out = upscale_image(_img(40, 30), 1.5, method="lanczos")
    assert out.size == (60, 45)


def test_bad_factor_is_noop():
    im = _img()
    assert upscale_image(im, "not-a-number") is im


def test_realesrgan_falls_back_to_lanczos_when_unavailable():
    # realesrgan/basicsr aren't installed in the test env, so this must not raise
    # and must still produce the LANCZOS-upscaled size.
    out = upscale_image(_img(40, 30), 2.0, method="realesrgan")
    assert out.size == (80, 60)
