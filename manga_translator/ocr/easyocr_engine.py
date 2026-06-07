from .ocr_engine import OCREngine
from PIL import Image
from ..schemas.interface import OCRResult, DetectionResult
from ..utils.common import pil2cv
import easyocr
import numpy as np


class EasyOCREngine(OCREngine):
    """EasyOCR backend that mirrors PaddleOCREngine's per-bubble pipeline:
    crop each detection bbox -> clean outside the polygon -> OCR the crop ->
    map boxes back to full-image coordinates. Drop-in alternative to
    PaddleOCREngine for comparing recognition quality."""

    def __init__(self, language: str, device: str = "cpu"):
        super().__init__(backend="easyocr")

        self.gpu = device in ("gpu", "cuda")
        self.ocr_model = easyocr.Reader([language], gpu=self.gpu)

        print(f"EasyOCR initialized with language: {language} on {'GPU' if self.gpu else 'CPU'}")

    def ocr(self, cropped_images):
        cv_image = pil2cv(cropped_images)

        # detail=1 -> (poly, text, confidence); paragraph=True groups lines in
        # the (already isolated) bubble into reading order, like PaddleOCR's join.
        result = self.ocr_model.readtext(cv_image, detail=1, paragraph=True)

        texts = []
        boxes = []
        for res in result:
            poly, text = res[0], res[1]
            texts.append(text)
            boxes.append(self.poly_to_xyxy(poly))

        text = " ".join(t for t in texts if t)
        return text, boxes

    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        ocr_results = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            poly = det.segmentation
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = self.clean_poly(cropped_img, poly, offset=(x1, y1))
            text, boxes = self.ocr(cropped_img)

            mapped_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                mapped_box = [xmin + x1, ymin + y1, xmax + x1, ymax + y1]
                mapped_boxes.append(mapped_box)

            ocr_result = OCRResult(
                text=text,
                boxes=mapped_boxes,
                detection_result=det
            )
            ocr_results.append(ocr_result)

        return ocr_results

    def poly_to_xyxy(self, poly):
        """
        Accepts:
        - poly as list/tuple of 4 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        - or poly as flat 8-length list/array [x1,y1,x2,y2,...]
        - or already [xmin,ymin,xmax,ymax] -> returns as-is
        Returns:
        [xmin, ymin, xmax, ymax] as ints (to match OCRResult.boxes schema)
        """
        arr = np.array(poly, dtype=float)

        if arr.ndim == 2 and arr.shape == (4, 2):
            xs, ys = arr[:, 0], arr[:, 1]
            vals = [xs.min(), ys.min(), xs.max(), ys.max()]
        elif arr.ndim == 1 and arr.size == 8:
            xs, ys = arr[0::2], arr[1::2]
            vals = [xs.min(), ys.min(), xs.max(), ys.max()]
        elif arr.ndim == 1 and arr.size == 4:
            vals = [arr[0], arr[1], arr[2], arr[3]]
        else:
            raise ValueError(f"Unsupported poly format: shape={arr.shape}")

        return [int(round(v)) for v in vals]
