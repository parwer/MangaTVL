from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "assets" / "models" / "comictextdetector.pt.onnx"


class TextMasker:
    """Glyph probability map and text blocks from comic-text-detector (ONNX)."""

    def __init__(self, model_path=DEFAULT_MODEL, size: int = 1024, device: str = "cpu"):
        self.size = size
        self.blocks: list[list] = []
        self.torch_net = None
        device = "cuda" if device == 0 else str(device or "cpu")
        if device.startswith("cuda"):
            try:
                import onnx
                from onnx2torch import convert
                self.torch_net = convert(onnx.load(str(model_path))).to(device).eval()
                self.device = device
                return
            except ImportError:
                pass
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        want = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}.get(device.split(":")[0])
        provs = ([want] if want in ort.get_available_providers() else []) + ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        if provs[0] == "DmlExecutionProvider":  # DirectML requirements
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(model_path), so, providers=provs)
        self.inp = self.sess.get_inputs()[0].name

    def _run(self, x: np.ndarray):
        if self.torch_net is None:
            return self.sess.run(None, {self.inp: x})
        import torch
        with torch.no_grad():
            outs = self.torch_net(torch.from_numpy(x).to(self.device))
        return [o.float().cpu().numpy() for o in outs]

    def _forward_square(self, bgr: np.ndarray):
        """Letterbox (pad bottom/right) to size x size, run; prob map at input res + blocks."""
        h, w = bgr.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = self._run(np.ascontiguousarray(x))
        # outputs: text blocks (YOLOv5 head, [..., 7]), seg mask (1 ch), line map (2 ch)
        seg = next(o for o in outs if o.ndim == 4 and o.shape[1] == 1)
        blk = next((o for o in outs if o.ndim == 3 and o.shape[-1] == 7), None)
        blocks = decode_blocks(blk[0], s, w, h) if blk is not None else []
        return cv2.resize(seg[0, 0, :nh, :nw], (w, h), interpolation=cv2.INTER_LINEAR), blocks

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Glyph probability map (HxW, 0..1); fills ``self.blocks``."""
        h, w = bgr.shape[:2]
        if h <= w * 1.6:
            prob, self.blocks = self._forward_square(bgr)
            return prob
        tile, step = int(w * 1.4), int(w * 1.1)
        acc = np.zeros((h, w), np.float32)
        self.blocks = []
        y = 0
        while True:
            y2 = min(h, y + tile)
            p, blocks = self._forward_square(bgr[y:y2])
            acc[y:y2] = np.maximum(acc[y:y2], p)
            self.blocks += [[a, b + y, c, d + y, sc] for a, b, c, d, sc in blocks]
            if y2 >= h:
                break
            y += step
        return acc


def decode_blocks(b: np.ndarray, s: float, w: int, h: int, conf: float = 0.4, nms: float = 0.35) -> list[list]:
    """Decode CTD text-block rows into [x1, y1, x2, y2, score] in page pixels."""
    score = b[:, 4] * b[:, 5:].max(1)
    b, score = b[score > conf], score[score > conf]
    if not len(b):
        return []
    xywh = [[float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)] for cx, cy, bw, bh in b[:, :4]]
    keep = cv2.dnn.NMSBoxes(xywh, score.astype(float).tolist(), conf, nms)
    out = []
    for i in np.array(keep).reshape(-1):
        x, y, bw, bh = xywh[i]
        out.append([max(0, int(x / s)), max(0, int(y / s)), min(w, int((x + bw) / s)), min(h, int((y + bh) / s)),
                    float(score[i])])
    return out


def refine_mask(bgr: np.ndarray, prob: np.ndarray, thresh: float = 0.3) -> np.ndarray:
    """Binary glyph mask (uint8 0/255) grown to cover anti-aliased edges and outline strokes."""
    m = (prob > thresh).astype(np.uint8) * 255
    if not m.any():
        return m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        comp = lab[y:y + h, x:x + w] == i
        stroke = float(dist[y:y + h, x:x + w][comp].max())  # ~ half the stroke width
        r = int(np.clip(round(stroke * 1.2 + 2), 2, 9))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        pad = r + 1
        y1, y2, x1, x2 = max(0, y - pad), min(m.shape[0], y + h + pad), max(0, x - pad), min(m.shape[1], x + w + pad)
        sub = (lab[y1:y2, x1:x2] == i).astype(np.uint8) * 255
        out[y1:y2, x1:x2] |= cv2.dilate(sub, k)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
