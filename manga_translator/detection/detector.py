from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from ..schemas.interface import DetectionResult


class DetectorBase(ABC):
    """Common interface for detection backends.

    Subclasses (ONNX, ultralytics YOLO, ...) implement ``detect`` and return
    ``list[DetectionResult]``. ``build_results`` converts an ultralytics-style
    output (xyxy boxes + class ids + optional polygon masks from
    ``result.masks.xy``) into the schema in one call, so concrete subclasses
    can stay thin.
    """

    DEFAULT_CLS_NAMES = {0: "bubble", 1: "free_text"}

    def __init__(self, cls_names: Optional[dict[int, str]] = None):
        self.cls_names = dict(cls_names) if cls_names is not None else dict(self.DEFAULT_CLS_NAMES)

    @abstractmethod
    def detect(self, image: Image.Image | np.ndarray) -> list[DetectionResult]:
        raise NotImplementedError

    def class_name(self, class_id) -> str:
        return self.cls_names.get(int(class_id), "unknown")

    def build_result(
        self,
        bbox_xyxy: Sequence[float],
        class_id,
        polygon: Optional[Sequence[Sequence[float]]] = None,
    ) -> DetectionResult:
        x1, y1, x2, y2 = bbox_xyxy
        segmentation: Optional[list[list[int]]] = None
        if polygon is not None and len(polygon) > 0:
            segmentation = [[int(round(float(x))), int(round(float(y)))] for x, y in polygon]
        return DetectionResult(
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            form_type=self.class_name(class_id),
            segmentation=segmentation,
        )

    def build_results(
        self,
        boxes_xyxy,
        class_ids,
        polygons=None,
    ) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for i in range(len(boxes_xyxy)):
            poly = polygons[i] if polygons is not None and i < len(polygons) else None
            results.append(self.build_result(boxes_xyxy[i], class_ids[i], poly))
        return results
