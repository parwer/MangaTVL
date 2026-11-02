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


def test_keep_nontext_is_off_by_default_and_skips_marked_regions_when_on():
    reply = {"regions": [{"n": 0, "source": "HELLO THERE", "translation": "สวัสดี"},
                         {"n": 1, "keep": True, "translation": ""}]}
    off = asyncio.run(ContextTranslator(FakeTranslator([reply, {"regions": []}])).translate(REGIONS, PAGE))[0]
    assert off[1].translated_text == "WHO ARE YOU?"            # off: falls back to the OCR text
    fake = FakeTranslator([reply])
    on = asyncio.run(ContextTranslator(fake, keep_nontext=True).translate(REGIONS, PAGE))[0]
    assert [r.translated_text for r in on] == ["สวัสดี", ""]
    assert '"keep": true' in fake.calls[0]["system"] and len(fake.calls) == 1


def test_merge_context_keeps_first_glossary_spelling_and_updates_the_rest():
    from manga_translator.translators.context_translator import merge_context
    old = {"glossary": {"Aina": "ไออินะ"}, "characters": {"Iori": "polite"}, "story": "a"}
    new = {"glossary": {"Aina": "ไอน่า", "Chisa": "ชิสะ"}, "characters": {"Iori": "rude with friends"}, "story": "b"}
    m = merge_context(old, new)
    assert m["glossary"] == {"Aina": "ไออินะ", "Chisa": "ชิสะ"}
    assert m["characters"] == {"Iori": "rude with friends"} and m["story"] == "b"


def test_glossary_misses():
    from manga_translator.translators.context_translator import glossary_misses
    g = {"Aina": "ไออินะ", "Yakyuken": "ยาคิวเคน (เกม)", "Io": "ไอโอ"}
    assert glossary_misses("CALM DOWN, AINA!", "ใจเย็นนะไอน่า", g) == ["Aina"]
    assert glossary_misses("CALM DOWN, AINA!", "ใจเย็นนะไออินะ", g) == []
    assert glossary_misses("YAKYUKEN TIME", "ได้เวลายาคิวเคนแล้ว", g) == []
    assert glossary_misses("AINAMI", "x", g) == []           # whole words only; short terms ignored


def test_lock_glossary_reasks_a_line_with_the_wrong_spelling():
    ctx = {"glossary": {"Iori": "อิโอริ"}}
    fake = FakeTranslator([{"regions": [{"n": 0, "source": "HELLO THERE", "translation": "หวัดดี"},
                                        {"n": 1, "source": "WHO ARE YOU?", "translation": "นายคือใคร"}],
                            "context": {"glossary": {"Iori": "อีโอริ"}}},
                           {"regions": [{"n": 0, "translation": "หวัดดี อิโอริ"}]}])
    regions = [ocr("HELLO THERE, IORI", [10, 10, 120, 60]), REGIONS[1]]
    out, new_ctx = asyncio.run(ContextTranslator(fake, lock_glossary=True).translate(regions, PAGE, context=ctx))
    assert new_ctx["glossary"]["Iori"] == "อิโอริ"            # the first spelling stays
    assert out[0].translated_text == "หวัดดี อิโอริ" and len(fake.calls) == 2
    assert "glossary spellings exactly" in fake.calls[1]["inputs"]


def test_source_read_from_the_image_over_garbage_ocr_is_trusted():
    # OCR read "DEE? eRenz"; the model read "DEEP BREATH" from the image: not a swap, no re-ask
    regions = [ocr("DEE? eRenz", [10, 10, 120, 60]), REGIONS[1]]
    fake = FakeTranslator([{"regions": [{"n": 0, "source": "DEEP BREATH", "translation": "สูดหายใจ"},
                                        {"n": 1, "source": "WHO ARE YOU?", "translation": "เป็นใคร"}]}])
    out, _ = asyncio.run(ContextTranslator(fake).translate(regions, PAGE))
    assert out[0].translated_text == "สูดหายใจ" and out[0].source_text == "DEEP BREATH"
    assert len(fake.calls) == 1


def test_short_words_sharing_letters_are_not_a_swap():
    from manga_translator.translators.context_translator import is_swap
    ocr_texts = ["DEE? eRenz", "REALLY?", "ON ONe CONDITION:"]
    assert not is_swap(0, "BREATH", ocr_texts)               # shares letters with REALLY? by chance
    assert is_swap(0, "ON ONE CONDITION.", ocr_texts)        # really another region's text
