"""Run the pipeline on one image while capturing every stage's intermediate
output AND its raw inputs/outputs, so the evaluators (and a human auditor) can
verify exactly what each stage produced.

This deliberately mirrors ``Pipeline._run_on_image`` (pipeline.py) instead of
modifying it — the eval stays non-invasive. Two differences:
  * the final output **upscale** step is skipped, so the rendered image keeps
    the same dimensions as the original / inpainted image (required for the
    per-pixel spill / fill rendering metrics to line up with the polygons);
  * a ``capture`` sink is passed into ``translator.translate`` so the raw LLM
    request / response / token usage is recorded (see translator.py).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from manga_translator.schemas.interface import (
    DetectionResult,
    OCRResult,
    TranslationResult,
)


def make_image_id(image_path: str, root: Optional[str]) -> str:
    """Stable, filesystem-safe id for an image, derived from its path relative
    to the dataset root (so different chapters/folders don't collide)."""
    p = Path(image_path)
    try:
        rel = p.resolve().relative_to(Path(root).resolve()) if root else Path(p.name)
    except (ValueError, OSError):
        rel = Path(p.name)
    rel = rel.with_suffix("")
    return "__".join(rel.parts) or p.stem


def _params_snapshot(pipeline, *, provider, model, from_lang, to_lang,
                     process_image, font, text_scale, fake_translate) -> dict:
    """Record the configuration the run used, for the run log. Introspects the
    pipeline's stage objects (detector conf, ocr engine, inpainter/renderer)
    with getattr so it never fails on a missing attribute."""
    det = pipeline.det_model
    inp = pipeline.inpainter
    rnd = pipeline.renderer
    return {
        "provider": provider or pipeline.provider,
        "model": model or pipeline.model,
        "from_lang": from_lang or pipeline.from_lang,
        "to_lang": to_lang or pipeline.to_lang,
        "process_image": process_image,
        "fake_translate": fake_translate,
        "device": getattr(pipeline, "device", None),
        "resize_max": getattr(pipeline, "resize_max", None),
        "font": font,
        "text_scale": text_scale if text_scale is not None else getattr(rnd, "text_scale", None),
        "detector": {
            "class": type(det).__name__,
            "conf": getattr(det, "conf", None),
            "iou": getattr(det, "iou", None),
            "model_path": str(getattr(det, "model_path", "")) or None,
        },
        "ocr_engine": type(pipeline._get_ocr(from_lang)).__name__,
        "inpainter": {
            "class": type(inp).__name__,
            "method": getattr(inp, "method", None),
        },
        "renderer": {
            "class": type(rnd).__name__,
            "max_font_size": getattr(rnd, "max_font_size", None),
            "min_font_size": getattr(rnd, "min_font_size", None),
            "supersample": getattr(rnd, "supersample", None),
        },
    }


@dataclass
class Capture:
    """Everything one pipeline run produced + its raw stage I/O, kept for the
    evaluators and the run log."""

    image_path: str
    image_id: str = ""
    params: dict = field(default_factory=dict)
    original: Optional[Image.Image] = None
    detection: list[DetectionResult] = field(default_factory=list)
    ocr: list[OCRResult] = field(default_factory=list)
    translation: list[TranslationResult] = field(default_factory=list)
    translation_raw: dict = field(default_factory=dict)   # raw LLM req/resp/usage (capture hook)
    judge_raw: Optional[dict] = None                      # set by run_eval after judging
    inpainted: Optional[Image.Image] = None
    rendered: Optional[Image.Image] = None
    timings: dict = field(default_factory=dict)           # stage -> seconds
    skipped: bool = False                                 # True when detection found nothing
    error: Optional[str] = None

    def to_record(self, *, include_raw: bool = True) -> dict:
        """JSON-safe full record of this page: params + per-stage raw I/O +
        per-bubble source/translation. Images are NOT included (saved separately
        via --save-images). Metrics are added by run_eval, not here."""
        bubbles = []
        for i, det in enumerate(self.detection):
            tr = self.translation[i] if i < len(self.translation) else None
            ocr = self.ocr[i] if i < len(self.ocr) else None
            bubbles.append({
                "bubble_index": i,
                "form_type": det.form_type,
                "bbox": [int(v) for v in det.bbox],
                "has_polygon": bool(det.segmentation),
                "polygon_points": len(det.segmentation) if det.segmentation else 0,
                "ocr_text": ocr.text if ocr else None,
                "ocr_boxes": ocr.boxes if ocr else None,
                "source_text": (tr.ocr_result.text if tr else (ocr.text if ocr else None)),
                "translated_text": tr.translated_text if tr else None,
            })
        rec = {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "params": self.params,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
            "skipped": self.skipped,
            "error": self.error,
            "n_bubbles": len(self.detection),
            "bubbles": bubbles,
        }
        if include_raw:
            rec["translation_raw"] = self.translation_raw
            rec["judge_raw"] = self.judge_raw
        return rec


async def run_capture(
    pipeline,
    image_path: str,
    *,
    dataset_root: Optional[str] = None,
    process_image: bool = False,
    provider=None,
    api_key=None,
    model=None,
    from_lang=None,
    to_lang=None,
    font=None,
    text_scale=None,
    fake_translate: bool = False,
) -> Capture:
    """Run detection -> OCR -> translation -> inpaint -> render on a single page,
    capturing each intermediate + raw I/O. Errors are caught and recorded so a
    single bad page doesn't abort a whole eval batch.

    ``fake_translate=True`` skips the LLM and passes the OCR text straight
    through (translated_text = original) — lets stages 4/5 be measured with no
    API key / zero cost."""
    from manga_translator.utils.common import load_image

    cap = Capture(image_path=image_path, image_id=make_image_id(image_path, dataset_root))
    cap.params = _params_snapshot(
        pipeline, provider=provider, model=model, from_lang=from_lang, to_lang=to_lang,
        process_image=process_image, font=font, text_scale=text_scale, fake_translate=fake_translate,
    )
    try:
        cap.original = load_image(image_path)
    except Exception as e:  # unreadable / missing file
        cap.error = f"load_image: {e}"
        return cap

    image = cap.original
    try:
        t = time.perf_counter()
        cap.detection = pipeline.det_model.detect(image)
        cap.timings["detection"] = time.perf_counter() - t
        if not cap.detection:
            cap.skipped = True
            return cap  # nothing to translate — matches Pipeline's early return

        t = time.perf_counter()
        cap.ocr = pipeline._get_ocr(from_lang).get_ocr(image, cap.detection)
        cap.timings["ocr"] = time.perf_counter() - t

        t = time.perf_counter()
        if fake_translate:
            cap.translation = [
                TranslationResult(ocr_result=o, translated_text=o.text)
                for o in cap.ocr
            ]
            cap.translation_raw = {"fake_translate": True}
        else:
            context_image = (
                pipeline.resize_image(image, max_size=pipeline.resize_max)
                if process_image
                else None
            )
            provider = provider or pipeline.provider
            translator = pipeline._get_translator(
                provider, pipeline._resolve_key(provider, api_key)
            )
            capture: dict = {}
            cap.translation = await translator.translate(
                ocr_result=cap.ocr, image=context_image,
                model=model, from_lang=from_lang, to_lang=to_lang,
                capture=capture,
            )
            cap.translation_raw = capture
        cap.timings["translation"] = time.perf_counter() - t

        t = time.perf_counter()
        cap.inpainted = pipeline.inpainter.inpaint(image, cap.translation)
        cap.timings["inpainting"] = time.perf_counter() - t

        t = time.perf_counter()
        cap.rendered = pipeline.renderer.render(
            cap.inpainted, cap.translation,
            lang=to_lang or pipeline.to_lang, font=font, text_scale=text_scale,
        )
        cap.timings["rendering"] = time.perf_counter() - t
    except Exception as e:
        cap.error = f"{type(e).__name__}: {e}"
    return cap
