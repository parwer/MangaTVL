import asyncio
from typing import Optional, Any, Dict, List
import os

from PIL import Image

from ..utils.common import load_image, resize_image, show_images
from ..detection.onnx_detection import ONNXDetection
from ..ocr.paddleocr_engine import PaddleOCREngine
from ..inpainting.simple_lama_inpainter import SimpleLamaInpainter
from ..inpainting.opencv_inpainter import OpenCVInpainter
from ..rendering.renderer import TextRenderer
from ..schemas.interface import DetectionResult, OCRResult, TranslationResult
from ..translators.groq import AsyncGroqTranslator
from ..translators.gemini import AsyncGeminiTranslator
from ..translators.openrouter import AsyncOpenRouterTranslator

from google import genai
from openai import AsyncOpenAI
from groq import AsyncGroq

class Pipeline:
    """
    End-to-end pipeline:
      - detection (ONNXDetection)
      - OCR (PaddleOCREngine)
      - provider for translation (async, e.g. gemini, openrouter, groq)
      - model for translation (e.g. gemini-2.5-flash, moonshotai/kimi-k2-instruct, )
      - inpainting (InpainterBase)
      - rendering (TextRenderer)

    Usage:
      pipeline = Pipeline(...)
      result = pipeline.run()
      # result is a dict with detection/ocr/translation/inpainted/rendered...
    """

    def __init__(
        self,
        det_model: Optional[ONNXDetection] = None,
        ocr_engine: Optional[PaddleOCREngine] = None,
        provider: Optional[str] = "gemini",
        model: Optional[str] = None,
        user_prompt: Optional[str] = None,
        guidelines: Optional[str] = None,
        from_lang: Optional[str] = "english",
        to_lang: Optional[str] = "thai",
        inpainter: Optional[SimpleLamaInpainter] = None,
        renderer: Optional[TextRenderer] = None,
        resize_max: Optional[int] = 512,
        device: Optional[str]="cpu",
        font_path: Optional[str]="../assets/fonts/THSarabunNew.ttf"
    ):
        self.det_model = det_model or ONNXDetection()  # default model path from ONNXDetection
        self.ocr_engine = ocr_engine or PaddleOCREngine(language="en")
        self.provider = provider
        self.model = model
        self.user_prompt = user_prompt
        self.guidelines = guidelines
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.inpainter = inpainter or OpenCVInpainter(device=device)
        self.renderer = renderer or TextRenderer(font_path=font_path)
        self.resize_max = resize_max
        self.device = device

        self.translator = self._init_translator(provider)
    
    def _init_translator(self, provider: str):
        if provider == "openrouter":
            client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
            translator = AsyncOpenRouterTranslator(client=client, 
                                                   model=self.model, 
                                                   user_prompt=self.user_prompt, 
                                                   guidelines=self.guidelines,
                                                   from_lang=self.from_lang,
                                                   to_lang=self.to_lang)
        elif provider == "gemini":
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            translator = AsyncGeminiTranslator(client=client, 
                                                model=self.model, 
                                                user_prompt=self.user_prompt, 
                                                guidelines=self.guidelines,
                                                from_lang=self.from_lang,
                                                to_lang=self.to_lang)
        elif provider == "groq":
            client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
            translator = AsyncGroqTranslator(client=client, 
                                                model=self.model, 
                                                user_prompt=self.user_prompt, 
                                                guidelines=self.guidelines,
                                                from_lang=self.from_lang,
                                                to_lang=self.to_lang)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        return translator
        
    def _load(self, image_path) -> Image.Image:
        img = load_image(image_path)
        return img
    
    async def _translate(self, ocr_results: list[OCRResult], image_for_translate=None) -> list[TranslationResult]:
        return await self.translator.translate(ocr_result=ocr_results,
                                               image=image_for_translate)
    
    async def run(self, image_path, process_image=False):
        image = self._load(image_path)
        image_for_translate = None
        if process_image:
            image_for_translate = image

        detection_result = self.det_model.detect(image)
        ocr_result = self.ocr_engine.get_ocr(image, detection_result)
        translation_result = await self._translate(ocr_result, image_for_translate)
        inpainted_image = self.inpainter.inpaint(image, translation_result)
        rendered_image = self.renderer.render(inpainted_image, translation_result)
        return rendered_image
    
    
    async def run_batch(self, image_paths: List[str], process_image=False) -> List[Image.Image]:
        async def process_single(image_path):
            image = self._load(image_path)
            image_for_translate = image if process_image else None
            detection_result = self.det_model.detect(image)
            ocr_result = self.ocr_engine.get_ocr(image, detection_result)
            translation_result = await self._translate(ocr_result, image_for_translate)
            inpainted_image = self.inpainter.inpaint(image, translation_result)
            return self.renderer.render(inpainted_image, translation_result)

        tasks = [process_single(path) for path in image_paths]
        return await asyncio.gather(*tasks)
