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
from manga_translator.detection.onnx_detection import ONNXDetection
from manga_translator.ocr.paddleocr_engine import PaddleOCREngine
from manga_translator.ocr.easyocr_engine import EasyOCREngine
from manga_translator.inpainting.opencv_inpainter import OpenCVInpainter
from manga_translator.rendering.renderer import TextRenderer
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
        resize_max: int = 256,
        device: str = "cpu",
        font_path: Optional[str] = None,
    ):
        self.det_model = det_model or ONNXDetection()
        self.ocr_engine = ocr_engine or EasyOCREngine(language="en", device=device)
        self.inpainter = inpainter or OpenCVInpainter(device=device)
        self.renderer = renderer or (TextRenderer(font_path=font_path) if font_path else TextRenderer())
        self.resize_max = resize_max
        self.device = device

        self.provider = provider
        self.model = model
        self.translator = self._init_translator(provider, user_prompt, guidelines, from_lang, to_lang)

    def _init_translator(self, provider, user_prompt, guidelines, from_lang, to_lang):
        kwargs = dict(
            model=self.model,
            user_prompt=user_prompt,
            guidelines=guidelines,
            from_lang=from_lang,
            to_lang=to_lang,
        )
        if provider == "openrouter":
            client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
            return AsyncOpenRouterTranslator(client=client, **kwargs)
        if provider == "gemini":
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            return AsyncGeminiTranslator(client=client, **kwargs)
        if provider == "groq":
            client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
            return AsyncGroqTranslator(client=client, **kwargs)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _run_on_image(self, image: Image.Image, process_image: bool = False) -> Image.Image:
        """Run the full pipeline on a single already-loaded image."""
        detection_result = self.det_model.detect(image)
        if not detection_result:
            return image  # nothing to translate

        ocr_result = self.ocr_engine.get_ocr(image, detection_result)
        # A resized copy is only needed when the translator uses the image as visual context.
        context_image = self.resize_image(image, max_size=self.resize_max) if process_image else None
        translation_result = await self.translator.translate(ocr_result=ocr_result, image=context_image)
        inpainted_image = self.inpainter.inpaint(image, translation_result)
        return self.renderer.render(inpainted_image, translation_result)

    async def run(self, image_path, process_image: bool = False) -> Optional[Image.Image]:
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        return await self._run_on_image(image, process_image)

    async def run_batch(self, image_paths: List[str], process_image: bool = False,
                        show_progress: bool = True) -> List[Image.Image]:
        """Process multiple images concurrently. API-call concurrency is throttled by the
        translator's semaphore (CONCURRENT_REQUESTS)."""
        tasks = [self.run(path, process_image) for path in image_paths]
        if show_progress:
            return await tqdm_asyncio.gather(*tasks, desc="Processing")
        return await asyncio.gather(*tasks)

    def resize_image(self, image, max_size=1024):
        width, height = image.size
        if max(width, height) <= max_size:
            return image
        if width > height:
            new_size = (max_size, int(height * max_size / width))
        else:
            new_size = (int(width * max_size / height), max_size)
        return cv2pil(image.resize(new_size))
