import asyncio
import json

import cv2
import numpy as np
from PIL import Image

from manga_translator.detection.refine import complete_free_text, missed_blocks, orphan_glyphs, straddling_blocks
from manga_translator.inpainting.text_eraser import EraseResult
from manga_translator.ocr.region_ocr import read_regions
from manga_translator.schemas.interface import DetectionResult, OCRResult
from manga_translator.translators.translator import AsyncTranslatorBase
from pipeline import QualityPipeline


def ellipse(cx, cy, ax, ay, n=40):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[int(cx + ax * np.cos(a)), int(cy + ay * np.sin(a))] for a in t]


# ---------------------------------------------------------------- detection.refine
def test_straddling_block_becomes_free_text_for_its_outside_part():
    poly = ellipse(150, 150, 100, 70)
    glyph = np.zeros((300, 400), np.uint8)
    glyph[140:160, 120:180] = 255          # dialogue, inside
    glyph[140:160, 240:300] = 255          # aside, crossing the right edge (x = 250)
    blocks = [[110, 135, 310, 165, 0.9]]
    out = straddling_blocks([DetectionResult(bbox=[50, 80, 250, 220], segmentation=poly, form_type="bubble")],
                            blocks, glyph, core=glyph, gray=np.full((300, 400), 255, np.uint8))
    assert len(out) == 1 and out[0].form_type == "free_text"
    x1, y1, x2, y2 = out[0].bbox
    assert 230 <= x1 <= 245 and x2 >= 295 and x1 > 180     # the aside only, not the dialogue


def test_missed_block_and_orphan_glyphs_become_free_text():
    glyph = np.zeros((300, 400), np.uint8)
    glyph[20:60, 300:380] = 255
    dets = [DetectionResult(bbox=[10, 10, 100, 100], form_type="bubble")]
    extra = missed_blocks(dets, [[295, 15, 385, 65, 0.8]], glyph)
    assert [e.bbox for e in extra] == [[300, 20, 379, 59]]
    core = np.zeros((300, 400), np.uint8); core[200:240, 200:260] = 255
    orphans = orphan_glyphs(dets + extra, core)
    assert [o.bbox for o in orphans] == [[200, 200, 260, 240]]


def test_complete_free_text_grows_to_the_whole_line_and_merges():
    glyph = np.zeros((200, 400), np.uint8)
    glyph[50:70, 50:350] = 255
    a = DetectionResult(bbox=[45, 45, 220, 75], form_type="free_text")
    b = DetectionResult(bbox=[200, 45, 260, 75], form_type="free_text")
    dets = [a, b]
    complete_free_text(dets, [[40, 40, 360, 80, 0.9]], glyph)
    assert dets == [a] and a.bbox[0] <= 50 and a.bbox[2] >= 350


# ---------------------------------------------------------------- ocr.region_ocr
class EchoOCR:
    """Reads 'X' for every dark pixel row-run it sees inside the box (enough to tell hidden from not)."""

    def get_ocr(self, image, dets):
        arr = np.asarray(image.convert("L"))
        out = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            ink = int((arr[y1:y2, x1:x2] < 100).sum())
            text = {"bubble": "BALLOON", "free_text": "ASIDE"}[d.form_type] if ink else ""
            if d.form_type == "bubble" and ink > 300:
                text += " +ASIDE"           # the aside's letters leaked into the balloon read
            out.append(OCRResult(text=text, boxes=[], detection_result=d))
        return out


def test_overlapping_free_text_is_hidden_from_the_balloon():
    img = np.full((200, 300, 3), 255, np.uint8)
    img[50:60, 40:60] = 0                      # dialogue letters (200 px)
    img[100:120, 150:200] = 0                  # aside letters (1000 px), inside the balloon box
    core = ((img.max(2) < 100) * 255).astype(np.uint8)
    dets = [DetectionResult(bbox=[20, 30, 220, 140], form_type="bubble"),
            DetectionResult(bbox=[150, 100, 200, 120], form_type="free_text")]
    out = read_regions(EchoOCR(), Image.fromarray(img), dets, core)
    assert [o.text for o in out] == ["BALLOON", "ASIDE"]


def test_unreadable_and_letterless_regions_are_dropped():
    class Fixed:
        def get_ocr(self, image, dets):
            return [OCRResult(text=t, boxes=[], detection_result=d) for t, d in zip(["", "1", "OK!"], dets)]
    dets = [DetectionResult(bbox=[0, 0, 10, 10], form_type="bubble"),
            DetectionResult(bbox=[20, 0, 30, 10], form_type="free_text"),
            DetectionResult(bbox=[40, 0, 50, 10], form_type="free_text")]
    out = read_regions(Fixed(), Image.new("RGB", (60, 20), "white"), dets)
    assert [o.text for o in out] == ["OK!"]


# ---------------------------------------------------------------- QualityPipeline flow
class FakeLLM(AsyncTranslatorBase):
    def __init__(self):
        super().__init__(client=None, model="fake")
        self.systems = []

    async def _translate(self, inputs, image=None, *, model=None, system_prompt=None, capture=None):
        self.systems.append(system_prompt)
        page = len(self.systems)
        return json.dumps({"regions": [{"n": 0, "source": "HELLO", "speaker": "A", "translation": f"สวัสดี{page}"}],
                           "context": {"characters": {"A": f"seen on page {page}"}}}, ensure_ascii=False)


class Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def stub_pipeline(llm):
    poly = ellipse(150, 100, 100, 60)
    qp = QualityPipeline.__new__(QualityPipeline)
    qp.det_model = Stub(detect=lambda im: [DetectionResult(bbox=[50, 40, 250, 160], segmentation=poly, form_type="bubble")])
    qp.masker = Stub(predict=lambda bgr: (bgr.max(2) < 100).astype(np.float32), blocks=[])
    qp._get_ocr = lambda lang: Stub(get_ocr=lambda im, dets: [OCRResult(text="HELLO", boxes=[], detection_result=d) for d in dets])
    qp._get_translator = lambda provider, key: llm
    qp._resolve_key = lambda provider, key: None
    qp.eraser = Stub(erase=lambda bgr, masks, dets, keep: EraseResult(np.full_like(bgr, 255), masks))
    rendered = []
    qp.renderer = Stub(render=lambda clean, results, **kw: rendered.append([r.translated_text for r in results]) or Image.new("RGB", (300, 200)))
    qp.provider, qp.to_lang, qp.device = "fake", "thai", "cpu"
    qp.keep_nontext = False
    qp.foreign_filter = False
    return qp, rendered


def page(path):
    img = np.full((200, 300, 3), 255, np.uint8)
    cv2.putText(img, "HELLO", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    Image.fromarray(img).save(path)
    return str(path)


def test_chapter_context_flows_page_to_page(tmp_path):
    llm = FakeLLM()
    qp, rendered = stub_pipeline(llm)
    paths = [page(tmp_path / f"{i}.png") for i in range(3)]
    out = asyncio.run(qp.run_batch(paths, show_progress=False))
    assert len(out) == 3 and all(o is not None for o in out)
    assert rendered == [["สวัสดี1"], ["สวัสดี2"], ["สวัสดี3"]]
    assert "seen on page 1" in llm.systems[1] and "seen on page 2" in llm.systems[2]
    assert "seen on page" not in llm.systems[0]


def test_each_batch_starts_a_new_chapter(tmp_path):
    llm = FakeLLM()
    qp, _ = stub_pipeline(llm)
    p = page(tmp_path / "a.png")
    asyncio.run(qp.run_batch([p], show_progress=False))
    asyncio.run(qp.run_batch([p], show_progress=False))
    assert "seen on page" not in llm.systems[1]


def test_reverted_region_is_not_lettered(tmp_path):
    llm = FakeLLM()
    qp, rendered = stub_pipeline(llm)
    qp.eraser = Stub(erase=lambda bgr, masks, dets, keep: EraseResult(bgr.copy(), masks, reverted=[(0, 500)]))
    asyncio.run(qp.run_batch([page(tmp_path / "a.png")], show_progress=False))
    assert rendered == [[""]] and qp.last_page["reverted"] == [(0, 500)]


def test_free_text_read_as_japanese_is_dropped():
    from manga_translator.ocr.region_ocr import drop_foreign_text
    img = Image.new("RGB", (100, 50), "white")
    items = [OCRResult(text=t, boxes=[], detection_result=DetectionResult(bbox=[0, 0, 10, 10], form_type=f))
             for t, f in (("Huh? 3", "free_text"), ("WATER", "free_text"), ("HELLO", "bubble"))]
    reads = iter(["どうやってるわ", "WATER"])
    out = drop_foreign_text(items, img, lambda crop: next(reads))
    assert [o.text for o in out] == ["WATER", "HELLO"]     # balloons are never checked
