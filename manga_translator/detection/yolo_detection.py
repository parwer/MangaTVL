from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO

from .detector import DetectorBase
from ..schemas.interface import DetectionResult


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "models" / "best_diplom.pt"


class YoloDetection(DetectorBase):
    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        cls_names: Optional[dict[int, str]] = None,
        conf: float = 0.25,
        iou: float = 0.5,
        device: Optional[str] = None,
        task: str = "detect"
    ):
        super().__init__(cls_names=cls_names)
        self.model = YOLO(str(model_path), task=task)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image: Image.Image | np.ndarray) -> list[DetectionResult]:
        result = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else None
        return self.build_results(boxes, cls_ids, polygons)
