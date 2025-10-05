from dataclasses import dataclass
from pydantic import BaseModel

from ..utils.common import refine_unit_value_type

class DetectionResult(BaseModel):
    bbox: list[int]
    form_type: str # bubble, free_text

class OCRResult(BaseModel):
    text: str
    boxes: list[list[int]] # each group of text that got from ocr [[x1, y1, x2, y2], ...]
    detection_result: DetectionResult

    def __post_init__(self):
        self.boxes = self._parse_bbox(self.boxes, "bbox")

    @staticmethod
    def _parse_bbox(value, field_name):
        if isinstance(value, str):
            try:
                return refine_unit_value_type(value)
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid {field_name} format: {value}")
        return value

class TranslationResult(BaseModel):
    ocr_result: OCRResult
    translated_text: str
