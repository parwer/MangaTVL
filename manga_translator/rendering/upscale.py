"""Optional output upscaling, applied to the final rendered page before it is
returned.

Two backends, chosen per request:
- ``lanczos`` (default): plain high-quality resize. Fast, no extra deps. Makes
  the page (and our crisp rendered text) bigger; it does not deblur the source
  artwork, just enlarges it.
- ``realesrgan``: AI super-resolution (anime-tuned Real-ESRGAN) that actually
  sharpens/deblurs the artwork. Heavy and slow on CPU, and depends on
  ``realesrgan``/``basicsr`` which aren't installed by default — so it is
  imported lazily and **falls back to LANCZOS** if unavailable or it errors.
"""
import functools

from PIL import Image

MAX_FACTOR = 4.0
ANIME_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)


def upscale_image(image: Image.Image, factor=1.0, method="lanczos", device="cpu") -> Image.Image:
    """Return ``image`` upscaled by ``factor`` using ``method``.

    ``factor`` <= 1 (or falsy) is a no-op; values are clamped to ``MAX_FACTOR``.
    ``realesrgan`` falls back to LANCZOS when the backend can't be loaded/run.
    """
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return image
    if factor <= 1.0:
        return image
    factor = min(factor, MAX_FACTOR)

    if method == "realesrgan":
        out = _realesrgan(image, factor, device)
        if out is not None:
            return out
        print("[upscale] Real-ESRGAN unavailable; falling back to LANCZOS")

    w, h = image.size
    return image.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)


@functools.lru_cache(maxsize=2)
def _realesrgan_upsampler(device):
    """Build & cache a RealESRGANer (anime 6-block x4 model). Lazy heavy import."""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    use_cuda = device in ("gpu", "cuda")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4, model_path=ANIME_MODEL_URL, model=model,
        tile=512, tile_pad=10, pre_pad=0,         # tile to bound memory on big pages
        half=use_cuda, device="cuda" if use_cuda else "cpu",
    )


def _realesrgan(image, factor, device):
    try:
        import numpy as np
        up = _realesrgan_upsampler(device)
        bgr = np.array(image.convert("RGB"))[:, :, ::-1]  # PIL RGB -> cv BGR
        out, _ = up.enhance(bgr, outscale=factor)
        return Image.fromarray(out[:, :, ::-1])           # BGR -> RGB
    except Exception as e:
        print(f"[upscale] Real-ESRGAN failed: {e}")
        return None
