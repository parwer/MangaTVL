from .ocr_engine import OCREngine
from PIL import Image
from ..schemas.interface import OCRResult, DetectionResult
from ..utils.common import pil2cv
import paddle
from paddleocr import PaddleOCR
from multiprocessing import Manager, Process

class PaddleOCREngine(OCREngine):
    def __init__(self, language: str):
        super().__init__(backend="paddleocr")

        self.ocr_model = PaddleOCR(use_angle_cls=True, lang=language,)
        self.device = "GPU" if paddle.is_compiled_with_cuda() else "CPU"

        print(f"PaddleOCR initialized with language: {language} on {self.device}")

    def ocr(self, cropped_images) -> OCRResult:
        cv_image = pil2cv(cropped_images)

        result = self.ocr_model.predict(cv_image)

        texts = []
        boxes = []
        for item in result:
            texts.extend(item["rec_texts"])
            boxes.extend(item["rec_boxes"])
        
        text = " ".join([i for i in texts])
        return text, boxes

    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        ocr_results = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            cropped_img = image.crop((x1, y1, x2, y2))
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