import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from google import genai
from openai import AsyncOpenAI
from groq import AsyncGroq

from manga_translator.utils.common import cv2pil, load_image
from manga_translator.utils.lang import ocr_lang_code, ocr_engine_name
from manga_translator.detection import build_detector
from manga_translator.ocr.easyocr_engine import EasyOCREngine
from manga_translator.ocr.mangaocr_engine import MangaOCREngine
from manga_translator.inpainting.opencv_inpainter import OpenCVInpainter
from manga_translator.rendering.renderer import TextRenderer
from manga_translator.rendering.upscale import upscale_image
from manga_translator.schemas.interface import OCRResult, TranslationResult
from manga_translator.translators.groq import AsyncGroqTranslator
from manga_translator.translators.gemini import AsyncGeminiTranslator
from manga_translator.translators.openrouter import AsyncOpenRouterTranslator

load_dotenv()


class Pipeline:
    """Orchestrates the 5 stages: detection -> OCR -> translation -> inpainting -> rendering."""

    def __init__(
        self,
        det_model=None,
        free_text_model=None,  # path, or "default" (assets/models/bubble_freetext_yolo26s.pt) -> hybrid detection
        ocr_engine=None,
        provider: str = "gemini",
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        guidelines: Optional[str] = None,
        from_lang: str = "english",
        to_lang: str = "thai",
        inpainter=None,
        renderer: Optional[TextRenderer] = None,
        resize_max: int = 1024,  # longest side sent to the VLM as visual context (~1300-1600 tokens/page)
        device: str = "cpu",
        font_path: Optional[str] = None,
    ):
        self.det_model = det_model or build_detector(free_text_model=free_text_model, device=device)
        # OCR engine(s): if the caller passes one, always use it. Otherwise build an
        # EasyOCREngine per source language on demand (cached) so from_lang can vary
        # per request without reloading the model every time.
        self._ocr_override = ocr_engine
        self._ocr_engines = {}
        self.inpainter = inpainter or OpenCVInpainter(device=device)
        self.renderer = renderer or (TextRenderer(font_path=font_path) if font_path else TextRenderer())
        self.resize_max = resize_max
        self.device = device

        self.provider = provider
        self.model = model
        self.user_prompt = user_prompt
        self.guidelines = guidelines
        self.from_lang = from_lang
        self.to_lang = to_lang

        # Translators are built per (provider, api_key) on demand and cached, so
        # provider/key can be chosen per request. Build the default one now to
        # validate config at startup and keep `self.translator` for convenience.
        self._translators = {}
        self.translator = self._get_translator(provider, self._resolve_key(provider, None))

        # Warm the OCR engine for the default source language (also keeps the
        # `self.ocr_engine` attribute for backward compatibility).
        self.ocr_engine = self._get_ocr(self.from_lang)

    def _get_ocr(self, from_lang):
        """OCR engine for the given source language."""
        if self._ocr_override is not None:
            return self._ocr_override

        lang = from_lang or self.from_lang
        engine = ocr_engine_name(lang)
        if engine == "mangaocr":
            cache_key = ("mangaocr", "ja")  # manga-ocr is Japanese-only (no per-language code)
            if cache_key not in self._ocr_engines:
                self._ocr_engines[cache_key] = MangaOCREngine(device=self.device)
            return self._ocr_engines[cache_key]

        code = ocr_lang_code(lang, engine="easyocr")
        cache_key = ("easyocr", code)
        if cache_key not in self._ocr_engines:
            self._ocr_engines[cache_key] = EasyOCREngine(language=code, device=self.device)
        return self._ocr_engines[cache_key]

    _ENV_KEY = {
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    def _resolve_key(self, provider, api_key):
        if api_key:
            return api_key
        return os.getenv(self._ENV_KEY.get(provider, ""))

    def _get_translator(self, provider, api_key):
        cache_key = (provider, api_key)
        if cache_key not in self._translators:
            self._translators[cache_key] = self._init_translator(provider, api_key)
        return self._translators[cache_key]

    def _init_translator(self, provider, api_key):
        kwargs = dict(
            model=self.model,
            user_prompt=self.user_prompt,
            guidelines=self.guidelines,
            from_lang=self.from_lang,
            to_lang=self.to_lang,
        )
        if provider == "openrouter":
            client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            return AsyncOpenRouterTranslator(client=client, **kwargs)
        if provider == "gemini":
            client = genai.Client(api_key=api_key)
            return AsyncGeminiTranslator(client=client, **kwargs)
        if provider == "groq":
            client = AsyncGroq(api_key=api_key)
            return AsyncGroqTranslator(client=client, **kwargs)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _run_on_image(self, image: Image.Image, process_image: bool = False, *,
                            provider=None, api_key=None, model=None,
                            from_lang=None, to_lang=None, font=None, text_scale=None,
                            upscale=None, upscaler=None, custom_instruction=None) -> Image.Image:
        """Run the full pipeline on a single already-loaded image."""
        detection_result = self.det_model.detect(image)
        if not detection_result:
            return image  # nothing to translate

        ocr_result = self._get_ocr(from_lang).get_ocr(image, detection_result)
        # A resized copy is only needed when the translator uses the image as visual context.
        context_image = self.resize_image(image, max_size=self.resize_max) if process_image else None

        provider = provider or self.provider
        translator = self._get_translator(provider, self._resolve_key(provider, api_key))
        translation_result = await translator.translate(
            ocr_result=ocr_result, image=context_image,
            model=model, from_lang=from_lang, to_lang=to_lang,
            custom_instruction=custom_instruction,
        )
        inpainted_image = self.inpainter.inpaint(image, translation_result)
        rendered = self.renderer.render(inpainted_image, translation_result,
                                        lang=to_lang or self.to_lang, font=font, text_scale=text_scale,
                                        original=image)
        # Optional output upscaling (final step): enlarge / sharpen the returned page.
        return upscale_image(rendered, factor=upscale or 1.0,
                             method=upscaler or "lanczos", device=self.device)

    async def run(self, image_path, process_image: bool = False, *,
                  provider=None, api_key=None, model=None,
                  from_lang=None, to_lang=None, font=None, text_scale=None,
                  upscale=None, upscaler=None, custom_instruction=None) -> Optional[Image.Image]:
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        return await self._run_on_image(
            image, process_image, provider=provider, api_key=api_key,
            model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
            upscale=upscale, upscaler=upscaler, custom_instruction=custom_instruction,
        )

    async def run_batch(self, image_paths: List[str], process_image: bool = False,
                        show_progress: bool = True, *, provider=None, api_key=None,
                        model=None, from_lang=None, to_lang=None, font=None, text_scale=None,
                        upscale=None, upscaler=None, custom_instruction=None) -> List[Image.Image]:
        """Process multiple images concurrently."""
        tasks = [
            self.run(path, process_image, provider=provider, api_key=api_key,
                     model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
                     upscale=upscale, upscaler=upscaler, custom_instruction=custom_instruction)
            for path in image_paths
        ]
        if show_progress:
            return await tqdm_asyncio.gather(*tasks, desc="Processing")
        return await asyncio.gather(*tasks)

    async def run_batch_stream(self, image_paths: List[str], process_image: bool = False, *,
                               provider=None, api_key=None, model=None,
                               from_lang=None, to_lang=None, font=None, text_scale=None,
                               upscale=None, upscaler=None, custom_instruction=None):
        """Like run_batch, but yields ``(index, image)`` as each image finishes."""
        async def _indexed(i, path):
            try:
                img = await self.run(
                    path, process_image, provider=provider, api_key=api_key,
                    model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
                    upscale=upscale, upscaler=upscaler, custom_instruction=custom_instruction,
                )
                return i, img
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                return i, None

        tasks = [asyncio.create_task(_indexed(i, p)) for i, p in enumerate(image_paths)]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    def resize_image(self, image, max_size=1024):
        width, height = image.size
        if max(width, height) <= max_size:
            return image
        if width > height:
            new_size = (max_size, int(height * max_size / width))
        else:
            new_size = (int(width * max_size / height), max_size)
        return cv2pil(image.resize(new_size))


class QualityPipeline(Pipeline):
    """Glyph-level pipeline with chapter context; same API as Pipeline."""

    def __init__(self, *, free_text_model="default", masker=None, inpainter=None, lama_onnx=None,
                 device: str = "cpu", **kw):
        from manga_translator.detection.hybrid_detection import DEFAULT_FREE_TEXT_MODEL
        from manga_translator.inpainting import LamaInpainter, TextEraser
        from manga_translator.masking import TextMasker

        if free_text_model == "default" and not DEFAULT_FREE_TEXT_MODEL.exists():
            free_text_model = None
        super().__init__(free_text_model=free_text_model, device=device,
                         inpainter=inpainter or LamaInpainter(device=device, onnx_path=lama_onnx), **kw)
        self.masker = masker or TextMasker(device=device)
        self.eraser = TextEraser(self.masker, self.inpainter)
        self.last_page: dict = {}  # diagnostics of the last page (timings, regions, reverted)

    async def _run_on_image(self, image: Image.Image, process_image: bool = False, *,
                            provider=None, api_key=None, model=None, from_lang=None, to_lang=None,
                            font=None, text_scale=None, upscale=None, upscaler=None,
                            custom_instruction=None, chapter: dict | None = None) -> Image.Image:
        import time
        import cv2
        from manga_translator.detection.refine import refine_regions
        from manga_translator.masking import refine_mask, region_glyph_masks
        from manga_translator.ocr.region_ocr import read_regions
        from manga_translator.translators.context_translator import ContextTranslator
        from manga_translator.utils.common import pil2cv

        chapter = chapter if chapter is not None else {}
        tm, t = {}, time.perf_counter()

        def lap(name):
            nonlocal t
            tm[name] = round(time.perf_counter() - t, 3)
            t = time.perf_counter()

        image = image.convert("RGB")
        bgr = pil2cv(image)
        dets = self.det_model.detect(image)
        lap("detect")
        prob = self.masker.predict(bgr)
        glyph = refine_mask(bgr, prob)
        core = ((prob > 0.6) * 255).astype("uint8")  # confident letter pixels
        dets = refine_regions(dets, self.masker.blocks, glyph, core, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        lap("textmask")
        ocr = read_regions(self._get_ocr(from_lang), image, dets, core)
        lap("ocr")
        self.last_page = {"timings": tm, "regions": len(ocr), "reverted": []}
        if not ocr:
            return image
        masks = region_glyph_masks(bgr, glyph, ocr)

        provider = provider or self.provider
        translator = ContextTranslator(self._get_translator(provider, self._resolve_key(provider, api_key)))
        results, chapter["context"] = await translator.translate(
            ocr, image, context=chapter.get("context"), model=model, from_lang=from_lang, to_lang=to_lang,
            custom_instruction=custom_instruction)
        lap("translate")
        keep = [i for i, r in enumerate(results) if (r.translated_text or "").strip()]
        erased = self.eraser.erase(bgr, masks, [o.detection_result for o in ocr], keep)
        for i, _ in erased.reverted:  # text survived erasing: original pixels are back, don't letter
            results[i].translated_text = ""
        lap("inpaint")
        rendered = self.renderer.render(erased.clean, results, lang=to_lang or self.to_lang, font=font,
                                        text_scale=text_scale, original=image, glyph_masks=erased.masks)
        lap("render")
        self.last_page.update(reverted=erased.reverted, results=results)
        return upscale_image(rendered, factor=upscale or 1.0, method=upscaler or "lanczos", device=self.device)

    async def run(self, image_path, process_image: bool = False, *, chapter: dict | None = None, **kw):
        """Run one page; pass the same ``chapter`` dict to carry the context."""
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        return await self._run_on_image(image, process_image, chapter=chapter, **kw)

    async def run_batch(self, image_paths: List[str], process_image: bool = False,
                        show_progress: bool = True, **kw) -> List[Image.Image]:
        return [img async for _, img in self.run_batch_stream(image_paths, process_image, **kw)]

    async def run_batch_stream(self, image_paths: List[str], process_image: bool = False, **kw):
        """Run pages in reading order with one shared chapter context."""
        chapter: dict = {}
        for i, path in enumerate(image_paths):
            try:
                yield i, await self.run(path, process_image, chapter=chapter, **kw)
            except Exception as e:
                print(f"Error processing image {path}: {e}")
                yield i, None
