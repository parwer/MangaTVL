import cv2
import numpy as np
from PIL import Image

from manga_translator.rendering.renderer import TextRenderer, estimate_glyph_masks
from manga_translator.rendering.typeset import break_units
from manga_translator.schemas.interface import DetectionResult, OCRResult, TranslationResult


def ellipse_poly(cx, cy, ax, ay, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[int(cx + ax * np.cos(a)), int(cy + ay * np.sin(a))] for a in t]


def page_with_balloon(poly, size=(600, 500), tone=200):
    """Grey page with a white balloon (black outline) at ``poly``."""
    img = np.full((size[1], size[0], 3), tone, np.uint8)
    pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    return img


def item(text, poly, form="bubble", ocr="HELLO THERE FRIEND"):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    det = DetectionResult(bbox=[min(xs), min(ys), max(xs), max(ys)], segmentation=poly, form_type=form)
    return TranslationResult(ocr_result=OCRResult(text=ocr, boxes=[det.bbox], detection_result=det),
                             translated_text=text)


def drawn_mask(before, after):
    a = np.asarray(after.convert("L"), np.int16)
    b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return np.abs(a - b) > 30


def test_thai_text_is_drawn_inside_the_balloon():
    poly = ellipse_poly(300, 250, 170, 110)
    clean = page_with_balloon(poly)
    r = TextRenderer()
    out = r.render(clean, [item("สวัสดีครับ วันนี้อากาศดีมากเลยนะ", poly)], lang="thai")
    m = drawn_mask(clean, out)
    assert m.sum() > 500                                  # something was lettered
    inside = np.zeros(m.shape, np.uint8)
    cv2.fillPoly(inside, [np.asarray(poly, np.int32).reshape(-1, 1, 2)], 1)
    assert (m & (inside == 0)).sum() == 0                 # nothing outside the balloon
    assert r.last_layout[0]["drawn"] and r.last_layout[0]["font_size"] >= 11


def test_empty_translation_is_left_untouched():
    poly = ellipse_poly(300, 250, 170, 110)
    clean = page_with_balloon(poly)
    r = TextRenderer()
    out = r.render(clean, [item("", poly)], lang="thai")
    assert not drawn_mask(clean, out).any()
    assert r.last_layout == [{"font_size": None, "drawn": False}]


def test_regions_do_not_letter_over_each_other():
    a, b = ellipse_poly(170, 250, 140, 100), ellipse_poly(440, 250, 140, 100)
    clean = page_with_balloon(a)
    pts = np.asarray(b, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(clean, [pts], (255, 255, 255)); cv2.polylines(clean, [pts], True, (0, 0, 0), 3)
    r = TextRenderer()
    out = r.render(clean, [item("ฝั่งซ้ายพูดก่อนนะ", a), item("แล้วฝั่งขวาก็ตอบกลับมา", b)], lang="thai")
    m = drawn_mask(clean, out)
    assert m[:, :300].sum() > 200 and m[:, 300:].sum() > 200
    assert all(x["drawn"] for x in r.last_layout)


def test_english_and_text_scale():
    poly = ellipse_poly(300, 250, 200, 120)
    clean = page_with_balloon(poly)
    small, big = TextRenderer(), TextRenderer()
    small.render(clean, [item("WHERE ARE YOU GOING?", poly)], lang="english", text_scale=0.5)
    big.render(clean, [item("WHERE ARE YOU GOING?", poly)], lang="english", text_scale=1.5)
    assert small.last_layout[0]["font_size"] < big.last_layout[0]["font_size"]


def test_pil_input_and_output():
    poly = ellipse_poly(300, 250, 170, 110)
    clean = Image.fromarray(cv2.cvtColor(page_with_balloon(poly), cv2.COLOR_BGR2RGB))
    out = TextRenderer().render(clean, [item("ทดสอบ", poly)], lang="thai")
    assert isinstance(out, Image.Image) and out.mode == "RGB" and out.size == clean.size


def test_glyph_masks_estimated_from_original():
    poly = ellipse_poly(300, 250, 170, 110)
    clean = page_with_balloon(poly)
    orig = clean.copy()
    cv2.putText(orig, "HELLO", (220, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    gm = estimate_glyph_masks(orig, clean, [item("x", poly)])[0]
    ys, xs = np.nonzero(gm)
    assert len(xs) > 100 and 215 <= xs.min() and xs.max() <= 380
    # with the original known the text is anchored on the old lettering
    r = TextRenderer()
    r.render(clean, [item("สวัสดี", poly)], lang="thai", original=orig)
    assert r.last_layout[0]["drawn"]


def test_bold_face_for_emphasis():
    # the big caption's original lettering is much larger than the dialogue's -> bold face
    a, b = ellipse_poly(150, 250, 120, 90), ellipse_poly(450, 250, 130, 110)
    clean = page_with_balloon(a)
    pts = np.asarray(b, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(clean, [pts], (255, 255, 255)); cv2.polylines(clean, [pts], True, (0, 0, 0), 3)
    orig = clean.copy()
    cv2.putText(orig, "hi there", (95, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(orig, "NO!", (380, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 0, 0), 9)
    r = TextRenderer()
    r.render(clean, [item("สวัสดีจ้ะ", a, ocr="hi there"), item("ไม่!", b, ocr="NO!")], lang="thai", original=orig)
    assert r.last_layout[1]["bold"] and not r.last_layout[0]["bold"]


def test_break_units():
    assert break_units("WHERE ARE you?", "english") == ["WHERE ", "ARE ", "you?"]
    th = break_units("สวัสดีครับ!", "thai")
    assert "".join(th) == "สวัสดีครับ!" and th[-1].endswith("!") and len(th) >= 2
    assert break_units("こんにちは。", "japanese")[-1] == "は。"
    assert break_units("   ", "thai") == []


def test_thai_punctuation_and_kept_names_stay_whole():
    assert break_units("ตัวมาร...", "thai") == ["ตัว", "มาร..."]           # was "มา" + "ร..." (abbreviation)
    units = break_units("ชิโอริตัวมารกลับบ้านไปซะที", "thai", keep_words=["ชิโอริ"])
    assert units[0] == "ชิโอริ" and "".join(units) == "ชิโอริตัวมารกลับบ้านไปซะที"
    assert break_units("WHERE ARE you?", "english", keep_words=["ARE"]) == ["WHERE ", "ARE ", "you?"]
