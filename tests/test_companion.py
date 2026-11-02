"""Tests for the companion app API with a stub pipeline (no models)."""
import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

import apps.companion.server as srv
from manga_translator.schemas.interface import DetectionResult, OCRResult
from pipeline import PageState

EXT = {"origin": "chrome-extension://" + "a" * 32}


def data_url(color="white"):
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class StubPipe:
    def __init__(self, texts):
        self.texts, self.finished = texts, []

    def analyze(self, image, from_lang=None):
        det = DetectionResult(bbox=[0, 0, 10, 10], form_type="bubble")
        return PageState(image=image, ocr=[OCRResult(text=t, boxes=[], detection_result=det) for t in self.texts])

    def finish(self, page, results, to_lang=None, font=None, text_scale=None):
        self.finished.append([r.translated_text for r in results])
        return Image.new("RGB", page.image.size, "black")


def client(monkeypatch, texts):
    stub = StubPipe(texts)
    monkeypatch.setattr(srv, "PIPE", stub)
    monkeypatch.setattr(srv, "relay_ok", lambda: False)
    return TestClient(srv.app), stub


def test_web_pages_cannot_call_the_app(monkeypatch):
    c, _ = client(monkeypatch, ["HELLO"])
    assert c.post("/v1/translate", json={"image": data_url()}, headers={"origin": "https://evil.example"}).status_code == 403


def test_page_without_text_is_returned_unchanged(monkeypatch):
    c, _ = client(monkeypatch, [])
    url = data_url()
    r = c.post("/v1/translate", json={"image": url, "context": {"a": 1}}, headers=EXT).json()
    assert r == {"image": url, "context": {"a": 1}}


def test_builtin_flow_hands_text_to_the_extension_then_renders(monkeypatch):
    c, stub = client(monkeypatch, ["HELLO", "BYE"])
    r = c.post("/v1/translate", json={"image": data_url(), "llm": {"provider": "builtin"}}, headers=EXT).json()
    assert r["needs_client_translation"] and [b["text"] for b in r["bubbles"]] == ["HELLO", "BYE"]
    out = c.post("/v1/render", json={"job": r["job"], "translations": ["สวัสดี", "ลาก่อน"]}, headers=EXT).json()
    assert out["image"].startswith("data:image/webp;base64,")
    assert stub.finished == [["สวัสดี", "ลาก่อน"]]
    assert c.post("/v1/render", json={"job": r["job"], "translations": []}, headers=EXT).status_code == 404


def test_oauth_code_handoff(monkeypatch):
    c, _ = client(monkeypatch, [])
    assert c.get("/oauth/callback/xyz?code=abc").status_code == 200
    assert c.get("/oauth/code?state=xyz").json() == {"code": "abc"}
    assert c.get("/oauth/code?state=xyz").status_code == 404
