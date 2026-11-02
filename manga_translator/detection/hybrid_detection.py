from __future__ import annotations

from pathlib import Path

from .detector import DetectorBase
from ..schemas.interface import DetectionResult

MODELS_DIR = Path(__file__).resolve().parent.parent / "assets" / "models"
DEFAULT_FREE_TEXT_MODEL = MODELS_DIR / "bubble_freetext_yolo26s.pt"


def _inside_share(f, b) -> float:
    """Share of box f covered by box b."""
    w = min(f[2], b[2]) - max(f[0], b[0])
    h = min(f[3], b[3]) - max(f[1], b[1])
    return max(0, w) * max(0, h) / max(1, (f[2] - f[0]) * (f[3] - f[1]))


class HybridDetection(DetectorBase):
    def __init__(self, balloon_detector: DetectorBase, free_text_detector: DetectorBase,
                 max_inside: float = 0.5):
        super().__init__()
        self.balloons = balloon_detector
        self.free_text = free_text_detector
        self.max_inside = max_inside

    def detect(self, image) -> list[DetectionResult]:
        dets = self.balloons.detect(image)
        boxes = [d.bbox for d in dets if d.form_type != "free_text"]
        extra = [f for f in self.free_text.detect(image) if f.form_type == "free_text"
                 and all(_inside_share(f.bbox, b) < self.max_inside for b in boxes)]
        return dets + extra


def load_detector(model_path=None, device=None, **kw) -> DetectorBase:
    """ONNXDetection for an .onnx file, else YoloDetection."""
    if model_path is not None and str(model_path).endswith(".onnx"):
        from .onnx_detection import ONNXDetection
        return ONNXDetection(model_path, device=device, **kw)
    from .yolo_detection import YoloDetection
    if model_path is None:
        return YoloDetection(device=device, **kw)
    return YoloDetection(model_path=model_path, device=device, task="segment", **kw)


def build_detector(balloon_model=None, free_text_model=None, device=None) -> DetectorBase:
    """Balloon detector, made hybrid when a free-text model is given."""
    det = load_detector(balloon_model, device)
    if free_text_model == "default":
        free_text_model = DEFAULT_FREE_TEXT_MODEL
    if not free_text_model:
        return det
    return HybridDetection(det, load_detector(free_text_model, device))
