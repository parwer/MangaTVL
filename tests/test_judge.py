"""Tests for eval.judge — two-image (original+rendered) attachment + parsing,
using a fake OpenAI-compatible client (no network)."""
import asyncio

from PIL import Image

from manga_translator.schemas.interface import DetectionResult, OCRResult, TranslationResult
from eval.judge import run_judge


# ---- fake async OpenAI-compatible client that records the request ----
class _Resp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]
        self.usage = None


class _Completions:
    def __init__(self, outer):
        self.outer = outer

    async def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        return _Resp(self.outer.reply)


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})()


def _tr(i, src, tr):
    det = DetectionResult(bbox=[0, 0, 1, 1], form_type="bubble")
    ocr = OCRResult(text=src, boxes=[[0, 0, 1, 1]], detection_result=det)
    return TranslationResult(ocr_result=ocr, translated_text=tr)


def _img():
    return Image.new("RGB", (8, 8), (255, 255, 255))


REPLY = '[{"text_no": 0, "adequacy": 4, "fluency": 5}, {"text_no": 1, "adequacy": 2, "fluency": 3}]'


def _user_content(client):
    return client.calls[0]["messages"][1]["content"]


def test_two_images_attached_and_parsed():
    client = FakeClient(REPLY)
    trans = [_tr(0, "hello", "สวัสดี"), _tr(1, "bye", "ลาก่อน")]
    out = asyncio.run(run_judge("openrouter", client, "m", trans,
                                original_image=_img(), rendered_image=_img()))

    per = {p["bubble_index"]: p for p in out["per"]}
    assert per[0]["adequacy"] == 4 and per[0]["fluency"] == 5
    assert per[1]["adequacy"] == 2 and per[1]["low_adequacy"] is True

    user = _user_content(client)
    assert isinstance(user, list)
    img_parts = [p for p in user if p.get("type") == "image_url"]
    labels = " ".join(p["text"] for p in user if p.get("type") == "text")
    assert len(img_parts) == 2
    assert "ORIGINAL" in labels and "TRANSLATED" in labels
    assert out["raw"]["images"] == [
        "IMAGE 1 — ORIGINAL (source) page:",
        "IMAGE 2 — TRANSLATED (rendered) page:",
    ]


def test_no_images_uses_string_content():
    client = FakeClient(REPLY)
    out = asyncio.run(run_judge("openrouter", client, "m", [_tr(0, "hi", "ไง")]))
    assert isinstance(_user_content(client), str)   # backward path: plain text
    assert out["raw"]["images"] == []


def test_single_image_one_part():
    client = FakeClient(REPLY)
    out = asyncio.run(run_judge("openrouter", client, "m", [_tr(0, "hi", "ไง")],
                                rendered_image=_img()))
    img_parts = [p for p in _user_content(client)
                 if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts) == 1
    assert out["raw"]["images"] == ["IMAGE 2 — TRANSLATED (rendered) page:"]
