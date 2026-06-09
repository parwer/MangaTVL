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
from manga_translator.detection.yolo_detection import YoloDetection
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
        self.det_model = det_model or YoloDetection(device=device)
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
        """OCR engine for the given source language. Uses the caller-supplied
        engine if one was passed; otherwise picks a backend per language
        (Japanese -> manga-ocr for vertical text, others -> EasyOCR) and caches
        one instance per (engine, code) so switching from_lang doesn't reload
        the model each time."""
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
                            upscale=None, upscaler=None) -> Image.Image:
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
        )
        inpainted_image = self.inpainter.inpaint(image, translation_result)
        rendered = self.renderer.render(inpainted_image, translation_result,
                                        lang=to_lang or self.to_lang, font=font, text_scale=text_scale)
        # Optional output upscaling (final step): enlarge / sharpen the returned page.
        return upscale_image(rendered, factor=upscale or 1.0,
                             method=upscaler or "lanczos", device=self.device)

    async def run(self, image_path, process_image: bool = False, *,
                  provider=None, api_key=None, model=None,
                  from_lang=None, to_lang=None, font=None, text_scale=None,
                  upscale=None, upscaler=None) -> Optional[Image.Image]:
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        return await self._run_on_image(
            image, process_image, provider=provider, api_key=api_key,
            model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
            upscale=upscale, upscaler=upscaler,
        )

    async def run_batch(self, image_paths: List[str], process_image: bool = False,
                        show_progress: bool = True, *, provider=None, api_key=None,
                        model=None, from_lang=None, to_lang=None, font=None, text_scale=None,
                        upscale=None, upscaler=None) -> List[Image.Image]:
        """Process multiple images concurrently. API-call concurrency is throttled by the
        translator's semaphore (CONCURRENT_REQUESTS)."""
        tasks = [
            self.run(path, process_image, provider=provider, api_key=api_key,
                     model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
                     upscale=upscale, upscaler=upscaler)
            for path in image_paths
        ]
        if show_progress:
            return await tqdm_asyncio.gather(*tasks, desc="Processing")
        return await asyncio.gather(*tasks)

    async def run_batch_stream(self, image_paths: List[str], process_image: bool = False, *,
                               provider=None, api_key=None, model=None,
                               from_lang=None, to_lang=None, font=None, text_scale=None,
                               upscale=None, upscaler=None):
        """Like run_batch, but yields ``(index, image)`` as each image finishes
        (completion order, not request order) so callers can stream results
        instead of waiting for the whole batch. A failed image yields
        ``(index, None)``; ``index`` lets the caller map it back to the request."""
        async def _indexed(i, path):
            try:
                img = await self.run(
                    path, process_image, provider=provider, api_key=api_key,
                    model=model, from_lang=from_lang, to_lang=to_lang, font=font, text_scale=text_scale,
                    upscale=upscale, upscaler=upscaler,
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
