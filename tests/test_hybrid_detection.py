import numpy as np

from manga_translator.detection import HybridDetection, build_detector
from manga_translator.detection.detector import DetectorBase
from manga_translator.detection.onnx_detection import decode_segment, letterbox
from manga_translator.schemas.interface import DetectionResult


class Fixed(DetectorBase):
    def __init__(self, dets):
        super().__init__()
        self.dets = dets

    def detect(self, image):
        return list(self.dets)


def d(box, form):
    return DetectionResult(bbox=box, form_type=form)


def test_hybrid_keeps_balloons_and_free_text_outside_them():
    balloons = Fixed([d([100, 100, 300, 300], "bubble")])
    p3 = Fixed([d([110, 110, 290, 200], "bubble"),          # its own balloons are ignored
                d([120, 120, 280, 280], "free_text"),        # inside a balloon -> balloon's dialogue
                d([250, 250, 400, 320], "free_text"),        # straddles the edge (<50 % inside) -> kept
                d([500, 50, 600, 120], "free_text")])        # outside -> kept
    out = HybridDetection(balloons, p3).detect(None)
    assert [(o.form_type, o.bbox) for o in out] == [
        ("bubble", [100, 100, 300, 300]), ("free_text", [250, 250, 400, 320]), ("free_text", [500, 50, 600, 120])]


def test_hybrid_balloon_model_free_text_is_kept():
    out = HybridDetection(Fixed([d([0, 0, 10, 10], "free_text")]), Fixed([])).detect(None)
    assert len(out) == 1


def test_build_detector_plain_without_free_text_model(monkeypatch):
    import manga_translator.detection.hybrid_detection as hd
    monkeypatch.setattr(hd, "load_detector", lambda path=None, device=None, **k: Fixed([]))
    assert not isinstance(build_detector(), HybridDetection)
    assert isinstance(build_detector(free_text_model="x.pt"), HybridDetection)


def test_letterbox_dynamic_pads_to_stride_only():
    canvas, r, top, left = letterbox(np.zeros((1000, 700, 3), np.uint8), (640, 640), 32, dynamic=True)
    assert canvas.shape[0] == 640 and canvas.shape[1] % 32 == 0 and canvas.shape[1] < 640
    canvas, r, top, left = letterbox(np.zeros((1000, 700, 3), np.uint8), (640, 640), 32, dynamic=False)
    assert canvas.shape[:2] == (640, 640) and left > 0


def test_decode_segment_maps_box_and_polygon_back():
    # one anchor: box centred (320, 320) 200x100 on a 640x640 input, class 1, mask = whole proto plane
    nc, nm = 2, 4
    pred = np.zeros((4 + nc + nm, 3), np.float32)
    pred[:, 0] = [320, 320, 200, 100, 0.1, 0.9, 1, 0, 0, 0]
    pred[:, 1] = [100, 100, 20, 20, 0.05, 0.05, 1, 0, 0, 0]       # below conf -> dropped
    pred[:, 2] = [322, 321, 198, 100, 0.1, 0.8, 1, 0, 0, 0]       # duplicate -> NMS
    protos = np.ones((nm, 160, 160), np.float32)
    # original 1280x1280 page -> ratio 0.5, no padding
    out = decode_segment(pred, protos, nc, 0.25, 0.5, (640, 640), 0.5, 0, 0, (1280, 1280))
    assert len(out) == 1
    (x1, y1, x2, y2), cls, score, poly = out[0]
    assert cls == 1 and abs(score - 0.9) < 1e-6
    assert (round(x1), round(y1), round(x2), round(y2)) == (440, 540, 840, 740)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert 438 <= min(xs) and max(xs) <= 842 and 538 <= min(ys) and max(ys) <= 742


def test_decode_segment_empty():
    pred = np.zeros((10, 5), np.float32)
    assert decode_segment(pred, np.zeros((4, 8, 8), np.float32), 2, 0.25, 0.5, (64, 64), 1, 0, 0, (64, 64)) == []
