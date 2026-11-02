from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from ..schemas.interface import OCRResult, TranslationResult
from ..utils.common import convert_img_to_base64
from .translator import AsyncTranslatorBase, OpenAICompatibleTranslator
from .utils.common import parse_response
from .utils.prompt import CONTEXT_REASK_PROMPT, get_context_sys_prompt
from .utils.set_of_mark import mark_image


def merge_context(old: dict, new: dict) -> dict:
    """Chapter context across pages: glossary spellings are kept (first wins), the rest updates."""
    out = dict(old or {})
    out.update({k: v for k, v in (new or {}).items() if v and k not in ("characters", "glossary")})
    out["characters"] = {**(old or {}).get("characters", {}), **(new or {}).get("characters", {})}
    out["glossary"] = {**(new or {}).get("glossary", {}), **(old or {}).get("glossary", {})}
    return out


def glossary_misses(source: str, translation: str, glossary: dict) -> list[str]:
    """Glossary terms that occur in ``source`` but whose spelling is missing from ``translation``."""
    miss = []
    for term, spelling in (glossary or {}).items():
        target = str(spelling).split(" (")[0].strip()  # "ยาคิวเคน (เกม...)" -> "ยาคิวเคน"
        if len(term) < 3 or not target:
            continue
        if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", source or "", re.I) and target not in (translation or ""):
            miss.append(term)
    return miss


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9぀-ヿ一-鿿]+", "", (t or "").lower())


def similarity(a: str, b: str) -> float:
    """0..1 similarity of two texts after normalisation (1 when either is empty)."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def is_swap(n: int, source: str, ocr_texts: list[str]) -> bool:
    """The model's source for tag n is clearly another region's text, not a reading of its own."""
    own = similarity(ocr_texts[n], source)
    if own >= 0.45:
        return False
    best = max((similarity(t, source) for m, t in enumerate(ocr_texts) if m != n and _norm(t)), default=0.0)
    return best >= max(0.6, own + 0.2)  # short words share letters by chance ("BREATH" / "REALLY?")


def same_text(ocr: str, source: str, thr: float = 0.45) -> bool:
    """True when the model's corrected source plausibly is this region's OCR text."""
    a, b = _norm(ocr), _norm(source)
    if not a or not b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= thr


def region_line(n: int, o: OCRResult) -> str:
    """One OCR input line; a second reading is added when it differs."""
    alt = (o.alt_text or "").strip()
    if _norm(alt) and _norm(alt) != _norm(o.text):
        return f"[{n}] {o.text}  (2nd OCR: {alt})"
    return f"[{n}] {o.text}"


class ContextTranslator:
    def __init__(self, translator: AsyncTranslatorBase, *, max_side: int = 1280, jpeg_quality: int = 88,
                 keep_nontext: bool = False, lock_glossary: bool = False):
        self.translator = translator
        self.lock_glossary = lock_glossary  # names keep one spelling across the chapter
        self.keep_nontext = keep_nontext  # experimental: let the model skip logos / kana SFX / art
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality

    async def _call(self, inputs, image, model, system_prompt, capture=None):
        kw = {"image_detail": "high"} if image is not None and isinstance(self.translator, OpenAICompatibleTranslator) else {}
        if capture is not None:
            kw["capture"] = capture
        return await self.translator._translate(inputs, image=image, model=model, system_prompt=system_prompt, **kw)

    async def translate(self, ocr_result: list[OCRResult], image, *, context: dict | None = None,
                        model=None, from_lang=None, to_lang=None, custom_instruction=None,
                        capture: dict | None = None) -> tuple[list[TranslationResult], dict]:
        """Translate one page."""
        context = context or {}
        if not ocr_result:
            return [], context
        tr = self.translator
        model = model or tr.model
        src, dst = from_lang or tr.from_lang, to_lang or tr.to_lang
        marked = mark_image(image, [o.detection_result.bbox for o in ocr_result], self.max_side)
        system = get_context_sys_prompt(json.dumps(context, ensure_ascii=False), src, dst, custom_instruction,
                                        keep_nontext=self.keep_nontext, lock_glossary=self.lock_glossary)
        inputs = "OCR per region:\n" + "\n".join(region_line(n, o) for n, o in enumerate(ocr_result))
        if capture is not None:
            capture.update(model=model, system_prompt=system, user_input=inputs, image_attached=True)
        try:
            raw = await self._call(inputs, convert_img_to_base64(marked, quality=self.jpeg_quality), model, system, capture)
        except Exception as e:
            print(f"[context-translate] API call failed: {e}")
            raw = None
            if capture is not None:
                capture["error"] = str(e)
        data = parse_response(raw) if raw else {}
        data = data if isinstance(data, dict) else {"regions": data}
        by_n = {}
        for x in data.get("regions") or []:
            try:
                by_n[int(x["n"])] = x
            except (KeyError, TypeError, ValueError):
                continue

        out, suspect = [], []
        for n, o in enumerate(ocr_result):
            x = by_n.get(n, {})
            if self.keep_nontext and x.get("keep") is True:  # not text: left unerased and unlettered
                out.append(TranslationResult(ocr_result=o, translated_text="", speaker=x.get("speaker") or "",
                                             source_text=o.text))
                continue
            source = x.get("source") or o.text
            out.append(TranslationResult(ocr_result=o, translated_text=x.get("translation") or o.text,
                                         speaker=x.get("speaker") or "", source_text=source))
            # re-ask only a real swap (the source is another region's text): a source that differs
            # from garbage OCR is the model reading the image correctly (ch.29: "DEE? eRenz" -> BREATH)
            if raw and (not x or (is_swap(n, source, [r.text for r in ocr_result])
                                  and not (_norm(o.alt_text) and same_text(o.alt_text, source, 0.6)))):
                suspect.append(n)
        new_context = data.get("context") or context
        respell = []
        if self.lock_glossary:
            new_context = merge_context(context, data.get("context") or {})
            gloss = new_context.get("glossary", {})
            respell = [n for n, r in enumerate(out) if n not in suspect and r.translated_text
                       and r.translated_text != r.ocr_result.text
                       and glossary_misses(f"{r.source_text or ''} {r.ocr_result.text}", r.translated_text, gloss)]
        # swaps are re-asked with the OCR text, spelling fixes with the model's own reading
        texts = {n: out[n].ocr_result.text for n in suspect}
        texts.update({n: out[n].source_text or out[n].ocr_result.text for n in respell})
        if texts:
            fixed = await self._reask(out, texts, model, src, dst, new_context)
            for n, t in fixed.items():
                out[n].translated_text, out[n].source_text = t, texts[n]
        if capture is not None:
            capture["parsed"] = {n: r.translated_text for n, r in enumerate(out)}
            capture["reasked"] = list(texts)
        return out, new_context

    async def _reask(self, out, texts: dict[int, str], model, src, dst, context) -> dict[int, str]:
        suspect = list(texts)
        page = "\n".join(f"[{n}] ({r.speaker or '?'}) {r.source_text or r.ocr_result.text} => {r.translated_text}"
                         for n, r in enumerate(out) if n not in texts)
        ask = "\n".join(f"[{n}] {texts[n]}" for n in suspect)
        msg = CONTEXT_REASK_PROMPT.format(from_lang=src, to_lang=dst, page=page, ask=ask,
                                          context=json.dumps(context, ensure_ascii=False))
        try:
            raw = await self._call(msg, None, model, f"You are a professional manga localizer ({src} -> {dst}).")
            data = parse_response(raw) if raw else {}
        except Exception:
            return {}
        items = data.get("regions", []) if isinstance(data, dict) else data
        res = {}
        for x in items or []:
            try:
                n = int(x["n"])
            except (KeyError, TypeError, ValueError):
                continue
            if n in suspect and x.get("translation"):
                res[n] = x["translation"]
        return res
