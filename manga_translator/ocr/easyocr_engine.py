from .bubble_ocr_matcher import match_ocr_to_bubbles
from .ocr_engine import OCREngine
from PIL import Image
from ..schemas.interface import OCRResult, DetectionResult
from ..utils.common import pil2cv
import easyocr
import numpy as np

"""
No Longer Used - This file is a placeholder for potential future EasyOCR-based OCR implementation.
"""

class EasyOCREngine(OCREngine):
    def __init__(self, language: str, device: str = "cpu"):
        super().__init__(backend="easyocr")

        self.gpu = True if device == "gpu" else False
        self.ocr_model = easyocr.Reader([language], gpu=self.gpu)
    
        print(f"EasyOCR initialized with language: {language} on {'GPU' if self.gpu else 'CPU'}")
    
    def get_ocr(self, image, detection_results):
        result = self.ocr_model.readtext(pil2cv(image), detail=1, paragraph=True)
        result = [(res[1], self.poly_to_xyxy(res[0])) for res in result]
        matched = match_ocr_to_bubbles(detection_results, result)
        return matched
    
    def poly_to_xyxy(self, poly):
        """
        Accepts:
        - poly as list/tuple of 4 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        - or poly as flat 8-length list/array [x1,y1,x2,y2,...]
        - or already [xmin,ymin,xmax,ymax] -> returns as-is
        Returns:
        [xmin, ymin, xmax, ymax] as floats
        """
        # make numpy array for robust handling
        arr = np.array(poly, dtype=float)

        # case: shape (4,2)
        if arr.ndim == 2 and arr.shape == (4,2):
            xs = arr[:,0]
            ys = arr[:,1]
            return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

        # case: flat length 8
        if arr.ndim == 1 and arr.size == 8:
            xs = arr[0::2]
            ys = arr[1::2]
            return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

        # case: already xyxy
        if arr.ndim == 1 and arr.size == 4:
            return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]

        raise ValueError(f"Unsupported poly format: shape={arr.shape}")
