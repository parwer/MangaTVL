from .ocr_engine import OCREngine
from PIL import Image
from ..schemas.interface import OCRResult, DetectionResult


class MangaOCREngine(OCREngine):
    """manga-ocr backend for Japanese (incl. vertical / tategaki text).

    manga-ocr (kha-white/manga-ocr) is trained specifically on Japanese manga
    text and takes a cropped text region, returning the recognized string
    directly — it has no internal text detector — so it slots into the same
    per-bubble crop pipeline as EasyOCREngine / PaddleOCREngine. It produces no
    per-line boxes, so the bubble bbox is used as the single box.

    The heavy ``manga_ocr`` import (transformers + a ~400MB model) lives inside
    ``__init__`` so it is only imported and the model only downloaded when this
    engine is actually constructed, i.e. when a Japanese page is requested.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(backend="mangaocr")

        from manga_ocr import MangaOcr  # lazy: heavy deps + model download
        force_cpu = device not in ("gpu", "cuda")
        self.ocr_model = MangaOcr(force_cpu=force_cpu)

        print(f"manga-ocr initialized on {'CPU' if force_cpu else 'GPU'}")

    def get_ocr(self, image: Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        ocr_results = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            poly = det.segmentation
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img = self.clean_poly(cropped_img, poly, offset=(x1, y1))

            # manga-ocr reads the whole crop as one text region and returns a string.
            text = self.ocr_model(cropped_img)

            ocr_result = OCRResult(
                text=text or "",
                boxes=[[x1, y1, x2, y2]],  # no per-line boxes; use the bubble bbox
                detection_result=det,
            )
            ocr_results.append(ocr_result)

        return ocr_results
