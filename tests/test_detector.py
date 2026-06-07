"""Tests for detection/detector.DetectorBase.build_result / build_results —
converting ultralytics-style output into DetectionResult schema objects."""
from manga_translator.detection.detector import DetectorBase


class DummyDetector(DetectorBase):
    def detect(self, image):
        return []


def test_build_result_bbox_to_int_and_form_type():
    d = DummyDetector()
    r = d.build_result([1.4, 2.6, 10.2, 20.8], 0)
    assert r.bbox == [1, 2, 10, 20]
    assert r.form_type == "bubble"  # default cls_names {0: bubble}
    assert r.segmentation is None


def test_build_result_free_text_class():
    d = DummyDetector()
    r = d.build_result([0, 0, 5, 5], 1)
    assert r.form_type == "free_text"


def test_build_result_unknown_class():
    d = DummyDetector()
    r = d.build_result([0, 0, 5, 5], 7)
    assert r.form_type == "unknown"


def test_build_result_polygon_to_int():
    d = DummyDetector()
    r = d.build_result([0, 0, 10, 10], 0, polygon=[[1.4, 2.6], [3.2, 4.8]])
    assert r.segmentation == [[1, 3], [3, 5]]


def test_build_result_custom_cls_names():
    d = DummyDetector(cls_names={0: "speech"})
    assert d.build_result([0, 0, 1, 1], 0).form_type == "speech"


def test_build_results_zips_boxes_classes_polygons():
    d = DummyDetector()
    boxes = [[0, 0, 1, 1], [2, 2, 3, 3]]
    classes = [0, 1]
    polys = [[[0, 0], [1, 1]], None]
    out = d.build_results(boxes, classes, polys)
    assert len(out) == 2
    assert out[0].form_type == "bubble" and out[0].segmentation == [[0, 0], [1, 1]]
    assert out[1].form_type == "free_text" and out[1].segmentation is None


def test_build_results_without_polygons():
    d = DummyDetector()
    out = d.build_results([[0, 0, 1, 1]], [0])
    assert len(out) == 1 and out[0].segmentation is None
