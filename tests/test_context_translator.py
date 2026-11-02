"""Tests for ContextTranslator: page-level translation with set-of-mark tags, speaker and
chapter context, on a fake provider (no network)."""
import asyncio
import json

from PIL import Image

from manga_translator.schemas.interface import DetectionResult, OCRResult
from manga_translator.translators.context_translator import ContextTranslator, same_text
from manga_translator.translators.translator import AsyncTranslatorBase
from manga_translator.translators.utils.common import parse_response
from manga_translator.translators.utils.set_of_mark import mark_image


class FakeTranslator(AsyncTranslatorBase):
    """Returns the queued replies in order and records every call."""

    def __init__(self, replies):
        super().__init__(client=None, model="fake")
        self.replies, self.calls = list(replies), []

    async def _translate(self, inputs, image=None, *, model=None, system_prompt=None, capture=None):
        self.calls.append({"inputs": inputs, "image": image, "system": system_prompt})
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def ocr(text, box):
    return OCRResult(text=text, boxes=[box], detection_result=DetectionResult(bbox=box, form_type="bubble"))


PAGE = Image.new("RGB", (400, 300), "white")
REGIONS = [ocr("HELLO THERE", [10, 10, 120, 60]), ocr("WHO ARE YOU?", [200, 100, 380, 160])]
CTX = {"characters": {"Iori": "male, casual"}, "glossary": {"Iori": "อิโอริ"}, "story": "s"}


def run(tr, **kw):
    return asyncio.run(ContextTranslator(tr).translate(REGIONS, PAGE, **kw))


def test_maps_by_tag_and_returns_speaker_source_context():
    fake = FakeTranslator([{"regions": [
        {"n": 1, "source": "WHO ARE YOU?", "speaker": "Chisa", "translation": "นายเป็นใคร?"},
        {"n": 0, "source": "HELLO THERE!", "speaker": "Iori", "translation": "หวัดดี!"}], "context": CTX}])
    out, ctx = run(fake)
    assert [r.translated_text for r in out] == ["หวัดดี!", "นายเป็นใคร?"]
    assert [r.speaker for r in out] == ["Iori", "Chisa"]
    assert out[0].source_text == "HELLO THERE!"
    assert ctx == CTX
    assert len(fake.calls) == 1 and fake.calls[0]["image"].startswith("data:image/jpeg;base64,")


def test_previous_context_is_sent_and_kept_when_reply_has_none():
    fake = FakeTranslator([{"regions": [{"n": 0, "source": "HELLO THERE", "translation": "a"},
                                        {"n": 1, "source": "WHO ARE YOU?", "translation": "b"}]}])
    _, ctx = run(fake, context=CTX)
    assert "อิโอริ" in fake.calls[0]["system"]
    assert ctx == CTX


def test_missing_region_falls_back_to_ocr_and_is_reasked():
    fake = FakeTranslator([{"regions": [{"n": 0, "source": "HELLO THERE", "translation": "หวัดดี"}]},
                           {"regions": [{"n": 1, "translation": "เธอเป็นใคร"}]}])
    out, _ = run(fake)
    assert out[1].translated_text == "เธอเป็นใคร"
    assert len(fake.calls) == 2 and fake.calls[1]["image"] is None
    assert "WHO ARE YOU?" in fake.calls[1]["inputs"]


def test_swapped_source_is_reasked_with_ocr_text():
    # the model attached region 1's text to tag 0
    fake = FakeTranslator([{"regions": [{"n": 0, "source": "WHO ARE YOU?", "translation": "เป็นใคร"},
                                        {"n": 1, "source": "WHO ARE YOU?", "translation": "เป็นใคร"}]},
                           {"regions": [{"n": 0, "translation": "สวัสดี"}]}])
    out, _ = run(fake)
    assert out[0].translated_text == "สวัสดี" and out[0].source_text == "HELLO THERE"
    assert out[1].translated_text == "เป็นใคร"


def test_failed_call_keeps_ocr_text_without_reask():
    class Boom(FakeTranslator):
        async def _translate(self, *a, **k):
            self.calls.append(1)
            raise RuntimeError("quota")
    fake = Boom([])
    out, ctx = run(fake, context=CTX)
    assert [r.translated_text for r in out] == ["HELLO THERE", "WHO ARE YOU?"]
    assert ctx == CTX and len(fake.calls) == 1


def test_empty_page():
    assert asyncio.run(ContextTranslator(FakeTranslator([])).translate([], PAGE)) == ([], {})


def test_same_text():
    assert same_text("THAT' S JuSt BECAUSE", "THAT'S JUST BECAUSE")
    assert not same_text("HELLO THERE", "WHO ARE YOU?")
    assert same_text("", "anything")


def test_mark_image_downscales_and_draws_tags():
    big = Image.new("RGB", (2560, 1000), "white")
    m = mark_image(big, [[100, 100, 600, 400]], max_side=1280)
    assert m.size == (1280, 500)
    assert m.getpixel((60, 60)) != (255, 255, 255)  # tag drawn at the scaled box corner
    assert big.getpixel((100, 100)) == (255, 255, 255)  # original untouched


def test_parse_response_keeps_first_object_on_extra_data():
    assert parse_response('{"regions": [{"n": 0}]}\n}') == {"regions": [{"n": 0}]}
