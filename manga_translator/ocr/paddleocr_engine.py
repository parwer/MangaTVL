from .ocr_engine import OCREngine
from PIL import Image
from ..schemas.interface import OCRResult, DetectionResult
from ..utils.common import pil2cv
from paddleocr import PaddleOCR
# import paddle

class PaddleOCREngine(OCREngine):
    def __init__(self, language: str):
        super().__init__(backend="paddleocr")

        self.ocr_model = PaddleOCR(use_angle_cls=True, lang=language,)

        print(f"PaddleOCR initialized with language: {language}")

    def ocr(self, cropped_images) -> OCRResult:
        cv_image = pil2cv(cropped_images)

        result = self.ocr_model.predict(cv_image)
        # text = " ".join([line[1][0] for line in result])
        # boxes = [line[0] for line in result]
        
        return result
        # return OCRResult(text=text, bbox=boxes)

    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        ocr_results = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            cropped_img = image.crop((x1, y1, x2, y2))
            ocr_result = self.ocr(cropped_img)
            # ocr_result = OCRResult(text=ocr_result.text, bbox=ocr_result.bbox)
            ocr_results.append(ocr_result)
        return ocr_results