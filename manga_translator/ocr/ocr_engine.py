from ..schemas.interface import OCRResult, DetectionResult
from PIL import Image

class OCREngine:
    def __init__(self, backend: str = "paddleocr"):
        self.backend = backend
    
    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        # Dummy implementation of OCR
        print(f"Performing OCR using backend: {self.backend}")
        return []  # Return an empty list for now