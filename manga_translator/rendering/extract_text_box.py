from ..schemas.interface import TranslationResult
from ..utils.common import combine_bbox

def extract_text_box(image, translation_result: TranslationResult) -> list[int]:
    return combine_bbox(translation_result.ocr_result.boxes)