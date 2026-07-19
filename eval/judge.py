"""LLM-as-judge for the translation stage (optional, opt-in).

This is a SECOND, separate API call that runs *after* translation: it takes the
(source, translated) pairs the translator already produced and scores each one
for adequacy + fluency. It does not re-translate.

The judge prompt is a standalone rubric — it is NOT the translation system
prompt, so there's no prompt reuse. The remaining bias to be aware of is
**model bias**: if the judge model == the translator model, the model tends to
over-score its own output (self-preference). Pass a *different* ``judge_model``
to remove that; until then the summary should be read as a relative signal, not
an absolute quality score.
"""

import json

from manga_translator.translators.utils.common import parse_response


JUDGE_SYSTEM = """You are a strict, impartial translation-quality evaluator for
You are given items, each with its {from_lang} source and its {to_lang} transla
You did not produce this translation; evaluate it purely on its own merits agai
You may also be given up to two images of the same comic page: IMAGE 1 = the
ORIGINAL page (the {from_lang} source text in its bubbles) and IMAGE 2 = the
TRANSLATED page (the {to_lang} output rendered back into the bubbles). Use the
images ONLY as visual context: (a) to disambiguate the source meaning from the
scene, and (b) to confirm the translation reads correctly in place. Still score
ONLY the per-item (source -> translated) text pairs listed below. Do NOT score
typesetting, placement, font, overflow, or image quality — judge meaning and
language only, on the two axes below.
Score every item on two independent axes, integers 1-5. Use the full anchor
descriptions below; do not interpolate freely between them.
ADEQUACY (meaning preservation only — ignore how natural it sounds):
5 = All meaning preserved; nothing lost, added, or distorted.
4 = Meaning fully preserved; at most a trivial nuance (e.g. honorific, tone par
3 = Core meaning preserved but a non-trivial detail is lost, added, or mistrans
(e.g. wrong pronoun referent, missing negation, swapped speaker intent).
2 = Partially conveys the source; a key clause or fact is wrong, missing, or re
1 = Meaning is wrong, nonsensical, or the text is left untranslated / empty.
FLUENCY (naturalness of the {to_lang} text read on its own — ignore source accu
5 = Reads as native {to_lang}; natural word order, register, and punctuation fo
4 = Reads naturally; at most a minor stylistic awkwardness a native speaker wou
3 = Understandable but noticeably stiff, literal, or grammatically off in a way
2 = Hard to parse without re-reading; word order or grammar is broken in places
1 = Broken, garbled, or not real {to_lang} (e.g. word salad, wrong script, mixe
Score the two axes independently: a sentence can be fluency=5 but adequacy=1 if
reads beautifully but says something different from the source, and vice versa.
Do not reward longer, more elaborate, or more detailed translations — verbosity
Do not reward confident or fluent-sounding phrasing on its own; check it agains
Penalise text left in the source language, hallucinated additions, and dropped
under adequacy specifically, not fluency.
Return ONLY a JSON array, one object per input item, no prose:
[{{"text_no": 0, "adequacy": 4, "fluency": 5}}, ...]
"""



def _build_input(pairs) -> str:
    """pairs: list of (text_no, source, translated) -> user message."""
    lines = ["Evaluate these translations:"]
    for no, src, tr in pairs:
        lines.append(json.dumps(
            {"text_no": no, "source": src, "translated": tr},
            ensure_ascii=False,
        ))
    return "\n".join(lines)


async def _call(provider, client, model, system, user, images=None, max_tokens=3000):
    """Dispatch one chat call by provider family. ``images`` is an ordered list of
    ``(label, b64_data_uri)`` tuples ("data:image/jpeg;base64,...") attached as
    real multimodal parts (each preceded by its text label) so the judge can
    score against the page context. Returns (raw_text, usage_dict)."""
    from manga_translator.translators.translator import _usage_to_dict
    images = images or []
    if provider == "gemini":
        import base64
        from google.genai import types
        from google.genai.types import GenerateContentConfig
        contents = []
        for label, b64 in images:
            raw = base64.b64decode(b64.split(",", 1)[1])
            contents.append(types.Part.from_text(text=label))
            contents.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
        contents.append(user)
        resp = await client.aio.models.generate_content(
            model=model, contents=contents,
            config=GenerateContentConfig(system_instruction=system),
        )
        if not resp:
            return None, None
        return resp.text, _usage_to_dict(getattr(resp, "usage_metadata", None))
    # openrouter / groq -> OpenAI-compatible chat.completions
    if images:
        user_content = []
        for label, b64 in images:
            user_content.append({"type": "text", "text": label})
            user_content.append({"type": "image_url", "image_url": {"url": b64, "detail": "low"}})
        user_content.append({"type": "text", "text": user})
    else:
        user_content = user
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_content}],
        max_tokens=max_tokens,
    )
    if not resp:
        return None, None
    return resp.choices[0].message.content, _usage_to_dict(getattr(resp, "usage", None))


async def run_judge(provider, client, model, translation, *,
                    from_lang="English", to_lang="Thai", same_model_as_translator=True,
                    original_image=None, rendered_image=None) -> dict:
    """Score every non-empty-source bubble. Returns {"per": [...], "summary": {...}, "raw": {...}}.
    Bubbles the judge doesn't return a score for are skipped (not penalised).

    ``original_image`` / ``rendered_image`` (optional): PIL/cv images of the page
    before and after translation. When given they are attached (labelled) to the
    judge call so adequacy/fluency are scored against the real visual context.
    Both require a vision-capable judge model."""
    pairs = []
    for i, t in enumerate(translation):
        src = (t.ocr_result.text or "").strip()
        if not src:
            continue
        pairs.append((i, src, (t.translated_text or "").strip()))
    if not pairs:
        return {"per": [], "summary": {"bubbles": 0}, "raw": {"provider": provider, "model": model}}

    images = []
    if original_image is not None or rendered_image is not None:
        from manga_translator.utils.common import convert_img_to_base64
        if original_image is not None:
            images.append(("IMAGE 1 — ORIGINAL (source) page:", convert_img_to_base64(original_image)))
        if rendered_image is not None:
            images.append(("IMAGE 2 — TRANSLATED (rendered) page:", convert_img_to_base64(rendered_image)))

    system = JUDGE_SYSTEM.format(from_lang=from_lang, to_lang=to_lang)
    user_input = _build_input(pairs)
    raw_meta = {"provider": provider, "model": model,
                "system_prompt": system, "user_input": user_input,
                "images": [label for label, _ in images],
                "raw_response": None, "usage": None, "error": None}
    try:
        raw, usage = await _call(provider, client, model, system, user_input, images=images)
        raw_meta["raw_response"] = raw
        raw_meta["usage"] = usage
    except Exception as e:
        print(f"[judge] API call failed: {e}")
        raw = None
        raw_meta["error"] = str(e)

    by_no = {}
    if raw:
        for r in parse_response(raw):
            if isinstance(r, dict) and "text_no" in r:
                try:
                    by_no[int(r["text_no"])] = (int(r["adequacy"]), int(r["fluency"]))
                except (ValueError, TypeError, KeyError):
                    continue

    per = []
    for no, src, tr in pairs:
        score = by_no.get(no)
        if score is None:
            continue  # judge didn't return this one — don't fabricate a score
        adequacy, fluency = score
        per.append({
            "bubble_index": no,         # == text_no == detection index (provenance)
            "adequacy": adequacy,
            "fluency": fluency,
            "low_adequacy": adequacy <= 2,
        })

    n = len(per)
    summary = {
        "scored_bubbles": n,
        "unscored_bubbles": len(pairs) - n,
        "mean_adequacy": round(sum(p["adequacy"] for p in per) / n, 3) if n else None,
        "mean_fluency": round(sum(p["fluency"] for p in per) / n, 3) if n else None,
        "low_adequacy_rate": round(sum(p["low_adequacy"] for p in per) / n, 3) if n else None,
        "self_judged": same_model_as_translator,  # True => scores are optimistic (self-bias)
    }
    raw_meta["parsed"] = {no: list(v) for no, v in by_no.items()}
    return {"per": per, "summary": summary, "raw": raw_meta}
