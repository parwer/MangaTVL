"""Text removal with LaMa (51M-param Fourier-convolution inpainter, manga fine-tune),
with a free flat-colour fill where the surround is one colour.

* Inside plain balloons the background around the glyphs is one colour, so filling the
  glyph mask with the median colour of the ring around it is pixel-perfect and free.
* Over art / screentone / gradients LaMa runs on a crop around each text cluster with
  context, so its FFT convolutions can continue halftone patterns and lines.

Two backends: PyTorch (``model_path``, a Big-LaMa checkpoint; bf16 on CUDA) or onnxruntime
(``onnx_path``, a fixed 512x512 export; larger crops run as overlapping native-scale tiles).
Default model: ``assets/models/lama_ft2.pt`` (tone-preserving fine-tune) when present, else
``lama_large_512px.ckpt``.

Use ``erase(bgr, mask)`` with a pixel glyph mask (masking.TextMasker). As an
``InpainterBase`` it also works in the classic pipeline (polygon fill + box masks).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..utils.common import cv2pil, pil2cv
from .inpainter import InpainterBase

MODELS = Path(__file__).resolve().parent.parent / "assets" / "models"


def default_model() -> Path:
    ft = MODELS / "lama_ft2.pt"
    return ft if ft.exists() else MODELS / "lama_large_512px.ckpt"


class LamaInpainter(InpainterBase):
    def __init__(self, device="cpu", model_path=None, onnx_path=None, flat_std: float = 6.0,
                 show_log: bool = False):
        super().__init__(device=device, show_log=show_log)
        self.device = "cuda" if device == 0 else str(device or "cpu")
        self.flat_std = flat_std
        self.stats = {"flat": 0, "lama": 0}
        self.ort = self.net = None
        if onnx_path:
            import onnxruntime as ort
            ort.set_default_logger_severity(3)
            want = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}.get(self.device.split(":")[0])
            provs = ([want] if want in ort.get_available_providers() else []) + ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            if provs[0] == "DmlExecutionProvider":
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.ort = ort.InferenceSession(str(onnx_path), so, providers=provs)
        else:
            from .lama import load_lama
            self.net = load_lama(model_path or default_model(), self.device)

    # ------------------------------------------------------------ InpainterBase hook
    def _inpaint(self, image, mask, **kwargs):
        return cv2pil(self.erase(pil2cv(image), mask))

    # ------------------------------------------------------------ glyph-mask erase
    def erase(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """bgr HxWx3 uint8, mask HxW uint8 (255 = erase). Returns a new image."""
        out = bgr.copy()
        if not mask.any():
            return out
        H, W = mask.shape
        # nearby glyphs form one job (a balloon, an SFX)
        grow = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(grow, 8)
        for i in range(1, n):
            x, y, w, h, _ = stats[i]
            cm = np.zeros_like(mask)
            cm[y:y + h, x:x + w] = mask[y:y + h, x:x + w] & ((lab[y:y + h, x:x + w] == i) * 255).astype(np.uint8)
            ring = cv2.dilate(cm, np.ones((9, 9), np.uint8)) & ~cm
            ring_px = bgr[ring > 0].astype(np.float32)
            if len(ring_px):
                color = np.median(ring_px, 0)
                dev = np.abs(ring_px - color).max(1)
                # robust flatness: a few outline / neighbour pixels in the ring must not push a
                # plain balloon to LaMa (-> grey smudges)
                if ring_px.std(0).max() < self.flat_std or (np.mean(dev < 14) >= 0.85 and color.mean() > 60):
                    out[cm > 0] = color.astype(np.uint8)
                    self.stats["flat"] += 1
                    continue
            mg = int(max(64, 0.6 * max(w, h)))  # context margin ~ cluster size
            x1, y1, x2, y2 = max(0, x - mg), max(0, y - mg), min(W, x + w + mg), min(H, y + h + mg)
            res = self.lama(out[y1:y2, x1:x2], cm[y1:y2, x1:x2])
            sub = cm[y1:y2, x1:x2] > 0
            out[y1:y2, x1:x2][sub] = res[sub]
            self.stats["lama"] += 1
        return out

    def lama(self, bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMa on one crop (no flat-fill shortcut)."""
        if self.ort is not None:
            h, w = bgr.shape[:2]
            return self._onnx_tiled(bgr, mask) if max(h, w) > 512 else self._onnx_one(bgr, mask)
        return self._torch(bgr, mask)

    def _torch(self, bgr, mask, max_side: int = 1024):
        import torch
        h, w = bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        img, m = bgr, mask
        if s < 1:
            img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ph, pw = (-img.shape[0]) % 8, (-img.shape[1]) % 8
        img_p = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        m_p = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        x = torch.from_numpy(img_p[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(self.device)
        mt = torch.from_numpy((m_p > 127).astype(np.float32))[None, None].to(self.device)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):  # fp16 darkens LaMa; bf16 is safe
                    y = self.net(x, mt)
            else:
                y = self.net(x, mt)
        y = (y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[:, :, ::-1]
        y = y[: img.shape[0], : img.shape[1]]
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC) if s < 1 else y

    def _onnx_one(self, bgr, mask, S: int = 512):
        """Fit a crop into the fixed SxS input: crops <= S are reflect-padded at native scale
        (LaMa is scale-sensitive: upscaling a small crop breaks lines up); larger ones are
        downscaled and scaled back."""
        h, w = bgr.shape[:2]
        sc = min(1.0, S / max(h, w))
        img, m = bgr, mask
        if sc < 1:
            img = cv2.resize(bgr, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
            m = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ih, iw = img.shape[:2]
        border = cv2.BORDER_REFLECT_101 if min(ih, iw) > 1 else cv2.BORDER_REPLICATE
        img = cv2.copyMakeBorder(img, 0, S - ih, 0, S - iw, border)
        m = cv2.copyMakeBorder(m, 0, S - ih, 0, S - iw, cv2.BORDER_CONSTANT, value=0)
        x = (img[:, :, ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None].copy()
        y = self.ort.run(None, {"image": x, "mask": (m > 127).astype(np.float32)[None, None]})[0]
        y = (np.clip(y[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)[:, :, ::-1][:ih, :iw]
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC) if sc < 1 else y

    def _onnx_tiled(self, bgr, mask, S: int = 512, margin: int = 96):
        """Native-scale SxS tiles over the mask instead of downscaling the crop (downscaled fills
        were blurry and left letter strokes). Tiles run in order so each sees the previous
        fills as context; each writes only its inner part (minus ``margin`` except at borders)."""
        h, w = bgr.shape[:2]
        out = bgr.copy()
        todo = mask > 127
        step = S - 2 * margin

        def starts(n):
            v = list(range(0, max(1, n - S) + 1, step))
            if n > S and (n - S) % step:
                v.append(n - S)
            return sorted(set(max(0, a) for a in v))

        for y in starts(h):
            for x in starts(w):
                y2, x2 = min(h, y + S), min(w, x + S)
                inner = np.zeros((h, w), bool)
                inner[y if y == 0 else y + margin:y2 if y2 == h else y2 - margin,
                      x if x == 0 else x + margin:x2 if x2 == w else x2 - margin] = True
                sel = todo & inner
                if not sel.any():
                    continue
                m = (todo * 255).astype(np.uint8)  # the whole remaining hole is hidden from the model
                res = self._onnx_one(out[y:y2, x:x2], m[y:y2, x:x2], S)
                sub = sel[y:y2, x:x2]
                out[y:y2, x:x2][sub] = res[sub]
                todo &= ~sel
        if todo.any():  # safety net
            res = self._onnx_one(out, (todo * 255).astype(np.uint8), S)
            out[todo] = res[todo]
        return out
