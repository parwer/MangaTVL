from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .detector import DetectorBase
from ..schemas.interface import DetectionResult


_PROVIDERS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}


def _session(path, device=None):
    """onnxruntime session on the provider for ``device`` (cpu by default)."""
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    want = "cuda" if device == 0 or str(device).startswith("cuda") else str(device or "cpu")
    provs = [_PROVIDERS[want]] if want in _PROVIDERS and _PROVIDERS[want] in ort.get_available_providers() else []
    provs.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    if provs[0] == "DmlExecutionProvider":  # DirectML requirements
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=provs)


def letterbox(rgb: np.ndarray, size, stride: int = 32, dynamic: bool = False):
    """Resize keeping aspect + pad with 114."""
    h, w = rgb.shape[:2]
    H, W = size
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = int(round((H - nh) / 2 - 0.1)), int(round((W - nw) / 2 - 0.1))
    if dynamic:
        dh, dw = ((H - nh) % stride) / 2, ((W - nw) % stride) / 2
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        H, W = nh + top + int(round(dh + 0.1)), nw + left + int(round(dw + 0.1))
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, top, left


def decode_segment(pred: np.ndarray, protos: np.ndarray, nc: int, conf: float, iou: float,
                   input_hw, ratio: float, top: int, left: int, image_hw, max_det: int = 300):
    """Decode an Ultralytics seg export into (box, class, score, polygon) in page pixels."""
    p = pred.T                                        # [N, 4+nc+nm]
    scores = p[:, 4:4 + nc]
    cls = scores.argmax(1)
    score = scores[np.arange(len(p)), cls]
    keep = score > conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if not len(p):
        return []
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    off = cls[:, None].astype(np.float32) * 7680.0   # per-class NMS: classes never suppress each other
    b = boxes + off
    idx = cv2.dnn.NMSBoxes([[float(a), float(c), float(e - a), float(f - c)] for a, c, e, f in b],
                           score.astype(float).tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)
    idx = idx[np.argsort(-score[idx])][:max_det]
    coeffs, boxes, cls, score = p[idx, 4 + nc:], boxes[idx], cls[idx], score[idx]
    nm, mh, mw = protos.shape
    masks = (coeffs @ protos.reshape(nm, -1)).reshape(-1, mh, mw).astype(np.float32)  # logits
    IH, IW = input_hw
    h, w = image_hw
    yy, xx = np.ogrid[0:IH, 0:IW]
    out = []
    for m, bx, c, s in zip(masks, boxes, cls, score):
        # as ultralytics process_mask(upsample=True) + Masks.xy: upsample the logits to the network
        # input, threshold, crop to the box, contour there, then map the points back (thresholding
        # at page resolution gave ~4 % smaller polygons)
        m = cv2.resize(m, (IW, IH), interpolation=cv2.INTER_LINEAR) > 0.0
        m &= (xx >= bx[0]) & (xx < bx[2]) & (yy >= bx[1]) & (yy < bx[3])
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = None
        if cs:
            pts = max(cs, key=len).reshape(-1, 2).astype(np.float32)
            pts[:, 0] = np.clip((pts[:, 0] - left) / ratio, 0, w)
            pts[:, 1] = np.clip((pts[:, 1] - top) / ratio, 0, h)
            poly = pts.tolist()
        x1, x2 = (float(np.clip((v - left) / ratio, 0, w)) for v in (bx[0], bx[2]))
        y1, y2 = (float(np.clip((v - top) / ratio, 0, h)) for v in (bx[1], bx[3]))
        out.append(((x1, y1, x2, y2), int(c), float(s), poly))
    return out


class ONNXDetection(DetectorBase):
    """Same ``detect()`` output as ``YoloDetection`` for an ONNX export of the same model."""

    def __init__(self, model_path, cls_names=None, conf: float = 0.25, iou: float = 0.5, device=None, **_):
        super().__init__(cls_names=cls_names)
        self.session = _session(model_path, device)
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.nc = len(ast.literal_eval(meta.get("names", "{0: 'bubble'}")))
        self.size = ast.literal_eval(meta.get("imgsz", "[640, 640]"))  # [h, w]
        self.stride = int(meta.get("stride", 32))
        self.dynamic = any(isinstance(d, str) or d is None for d in self.session.get_inputs()[0].shape[2:])
        self.conf, self.iou = conf, iou

    def detect(self, image: Image.Image | np.ndarray) -> list[DetectionResult]:
        # like ultralytics: a PIL image is RGB, a numpy array is OpenCV BGR
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        canvas, r, top, left = letterbox(rgb, self.size, self.stride, self.dynamic)
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        pred, protos = self.session.run(None, {self.input_name: x})
        dets = decode_segment(pred[0], protos[0], self.nc, self.conf, self.iou, canvas.shape[:2],
                              r, top, left, rgb.shape[:2])
        return self.build_results([d[0] for d in dets], [d[1] for d in dets], [d[3] for d in dets])
