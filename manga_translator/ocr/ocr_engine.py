from ..schemas.interface import OCRResult, DetectionResult
from PIL import Image
import cv2
import numpy as np

class OCREngine:
    def __init__(self, backend: str = "paddleocr"):
        self.backend = backend

    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        # Dummy implementation of OCR
        print(f"Performing OCR using backend: {self.backend}")
        return []  # Return an empty list for now

    def clean_poly(self, cropped_image: Image.Image, poly, offset=(0, 0), fill=(255, 255, 255)) -> Image.Image:
        """Erase everything outside ``poly`` so OCR only sees the bubble interior.

        The rectangular crop usually pulls in slivers of neighbouring bubbles or
        artwork; masking to the segmentation polygon removes that noise. When
        ``poly`` is ``None`` (detect-only model) the image is returned unchanged.

        Args:
            cropped_image: the PIL crop (already cut to the detection bbox).
            poly: polygon points ``[[x, y], ...]`` in full-image coordinates.
            offset: the crop's top-left ``(x1, y1)`` used to translate ``poly``
                into the crop's local frame.
            fill: replacement colour for pixels outside the polygon.

        Returns:
            A new PIL image with the outside region filled.
        """
        if poly is None or len(poly) == 0:
            return cropped_image

        ox, oy = offset
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2) - np.array([ox, oy], dtype=np.float32)
        pts = np.round(pts).astype(np.int32)

        arr = np.array(cropped_image)
        h, w = arr.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        if arr.ndim == 2:  # grayscale
            fill_value = fill[0] if isinstance(fill, (tuple, list)) else fill
        else:  # RGB / RGBA — pad or trim fill to the channel count
            ch = arr.shape[2]
            base = list(fill) if isinstance(fill, (tuple, list)) else [fill] * 3
            fill_value = tuple((base + [255] * ch)[:ch])

        arr[mask == 0] = fill_value
        return Image.fromarray(arr)
    