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


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9぀-ヿ一-鿿]+", "", (t or "").lower())


def same_text(ocr: str, source: str, thr: float = 0.45) -> bool:
    """True when the model's corrected source plausibly is this region's OCR text."""
    a, b = _norm(ocr), _norm(source)
    if not a or not b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= thr


class ContextTranslator:
    def __init__(self, translator: AsyncTranslatorBase, *, max_side: int = 1280, jpeg_quality: int = 88,
                 keep_nontext: bool = False):
        self.translator = translator
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
                                        keep_nontext=self.keep_nontext)
        inputs = "OCR per region:\n" + "\n".join(f"[{n}] {o.text}" for n, o in enumerate(ocr_result))
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
            if raw and (not x or not same_text(o.text, source)):
                suspect.append(n)
        new_context = data.get("context") or context
        if suspect:
            fixed = await self._reask(out, suspect, model, src, dst, new_context)
            for n, t in fixed.items():
                out[n].translated_text, out[n].source_text = t, out[n].ocr_result.text
        if capture is not None:
            capture["parsed"] = {n: r.translated_text for n, r in enumerate(out)}
            capture["reasked"] = suspect
        return out, new_context

    async def _reask(self, out, suspect, model, src, dst, context) -> dict[int, str]:
        page = "\n".join(f"[{n}] ({r.speaker or '?'}) {r.ocr_result.text} => {r.translated_text}"
                         for n, r in enumerate(out) if n not in suspect)
        ask = "\n".join(f"[{n}] {out[n].ocr_result.text}" for n in suspect)
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
