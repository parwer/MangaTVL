"""Tests for AsyncTranslatorBase.translate() — mapping LLM output back to OCR
items by text_no (not positional zip), with fallback to original text."""
import asyncio
import json

from manga_translator.translators.translator import AsyncTranslatorBase
from manga_translator.schemas.interface import OCRResult, DetectionResult


def _ocr(text):
    return OCRResult(
        text=text,
        boxes=[[0, 0, 1, 1]],
        detection_result=DetectionResult(bbox=[0, 0, 1, 1], form_type="bubble"),
    )


class FakeTranslator(AsyncTranslatorBase):
    """Bypasses real __init__ (no client/API) — only exercises translate() logic."""

    def __init__(self, fake_response):
        self._fake_response = fake_response

    async def _translate(self, inputs, image=None):
        return self._fake_response


def _run(fake_response, ocr_items):
    return asyncio.run(FakeTranslator(fake_response).translate(ocr_items))


THREE = lambda: [_ocr("AAA"), _ocr("BBB"), _ocr("CCC")]


def test_complete_response_maps_in_order():
    resp = json.dumps([
        {"text_no": 0, "translated_text": "ก"},
        {"text_no": 1, "translated_text": "ข"},
        {"text_no": 2, "translated_text": "ค"},
    ])
    out = _run(resp, THREE())
    assert [r.translated_text for r in out] == ["ก", "ข", "ค"]


def test_reordered_response_still_maps_by_text_no():
    resp = json.dumps([
        {"text_no": 2, "translated_text": "ค"},
        {"text_no": 0, "translated_text": "ก"},
        {"text_no": 1, "translated_text": "ข"},
    ])
    out = _run(resp, THREE())
    assert [r.translated_text for r in out] == ["ก", "ข", "ค"]


def test_missing_item_falls_back_to_original_text():
    resp = json.dumps([
        {"text_no": 0, "translated_text": "ก"},
        {"text_no": 2, "translated_text": "ค"},
    ])
    out = _run(resp, THREE())
    # index 1 was not returned -> keep original "BBB"
    assert [r.translated_text for r in out] == ["ก", "BBB", "ค"]


def test_extra_text_no_is_ignored():
    resp = json.dumps([
        {"text_no": 0, "translated_text": "ก"},
        {"text_no": 1, "translated_text": "ข"},
        {"text_no": 2, "translated_text": "ค"},
        {"text_no": 9, "translated_text": "X"},
    ])
    out = _run(resp, THREE())
    assert [r.translated_text for r in out] == ["ก", "ข", "ค"]


def test_garbage_response_falls_back_to_all_originals():
    out = _run("not valid json at all", THREE())
    assert [r.translated_text for r in out] == ["AAA", "BBB", "CCC"]


def test_none_response_falls_back_to_all_originals():
    out = _run(None, THREE())
    assert [r.translated_text for r in out] == ["AAA", "BBB", "CCC"]


def test_empty_ocr_returns_empty():
    assert _run("[]", []) == []


def test_result_count_always_matches_input():
    out = _run(json.dumps([{"text_no": 0, "translated_text": "ก"}]), THREE())
    assert len(out) == 3
