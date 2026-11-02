from .detector import DetectorBase
from .hybrid_detection import HybridDetection, build_detector, load_detector
from .onnx_detection import ONNXDetection


def __getattr__(name):  # ultralytics (torch) only when YoloDetection is used
    if name == "YoloDetection":
        from .yolo_detection import YoloDetection
        return YoloDetection
    raise AttributeError(name)
