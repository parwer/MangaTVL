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

        # enable_mkldnn=False: Paddle 3.3's PIR executor crashes inside the
        # oneDNN/MKLDNN CPU backend during text detection
        # (NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... onednn).
        # Disabling MKLDNN falls back to the plain CPU kernels.
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang=language, enable_mkldnn=False)
        self.device = "GPU" if paddle.is_compiled_with_cuda() else "CPU"

        print(f"PaddleOCR initialized with language: {language} on {self.device}")

    def ocr(self, cropped_images) -> OCRResult:
        cv_image = pil2cv(cropped_images)

        result = self.ocr_model.predict(cv_image)

        pairs = []
        for item in result:
            for t, b in zip(item["rec_texts"], item["rec_boxes"]):
                pairs.append((t, b))

        # Reading order: top-to-bottom (y_center), then left-to-right within a line (x_center).
        pairs.sort(key=lambda tb: ((tb[1][1] + tb[1][3]) / 2, (tb[1][0] + tb[1][2]) / 2))

        texts = [t for t, _ in pairs]
        boxes = [b for _, b in pairs]
        text = " ".join(texts)
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