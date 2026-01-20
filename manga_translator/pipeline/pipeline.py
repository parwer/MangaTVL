import asyncio
from typing import Optional, Any, Dict, List
import os
from tqdm.asyncio import tqdm_asyncio

from PIL import Image

from ..utils.common import cv2pil, load_image, resize_image, show_images
from ..detection.onnx_detection import ONNXDetection
from ..ocr.paddleocr_engine import PaddleOCREngine
from ..ocr.easyocr_engine import EasyOCREngine
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

from dotenv import load_dotenv

load_dotenv()

class Pipeline:
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
        resize_max: Optional[int] = 256,
        device: Optional[str]="cpu",
        font_path: Optional[str]="/home/parwer/MangaTVL/manga_translator/assets/fonts/THSarabunNew.ttf"
    ):
        self.det_model = det_model or ONNXDetection()  # default model path from ONNXDetection
        self.ocr_engine = ocr_engine or EasyOCREngine(language="en", device=device)  # default to English
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
        try:
            image = self._load(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        image_for_translate = None
        if process_image:
            image_for_translate = image

        detection_result = self.det_model.detect(image)
        if not detection_result:
            # print(f"No text detected in image: {image_path}")
            return image
        ocr_result = self.ocr_engine.get_ocr(image, detection_result)
        image_for_translate = self.resize_image(image_for_translate, max_size=self.resize_max) if image_for_translate else None # resize for faster translation if needed
        print("image size for translation:", image_for_translate.size if image_for_translate else "N/A")
        translation_result = await self._translate(ocr_result, image_for_translate)
        inpainted_image = self.inpainter.inpaint(image, translation_result)
        rendered_image = self.renderer.render(inpainted_image, translation_result)
        return rendered_image
    
    
    async def run_batch(self, image_paths: List[str], process_image=False, show_progress: bool = True) -> List[Image.Image]:
        """
        Process multiple images concurrently and show a tqdm progress bar.

        - image_paths: list of image file paths
        - process_image: pass the full image to translator if True
        - show_progress: whether to display a tqdm progress bar (auto-chooses notebook/terminal)
        """

        # choose proper tqdm for notebook vs terminal
        try:
            shell = get_ipython().__class__.__name__  # type: ignore
            in_notebook = shell == "ZMQInteractiveShell"
        except Exception:
            in_notebook = False

        if show_progress:
            try:
                if in_notebook:
                    from tqdm.notebook import tqdm as _tqdm
                else:
                    from tqdm import tqdm as _tqdm
            except Exception:
                # tqdm not installed -> fallback no progress
                _tqdm = None
        else:
            _tqdm = None

        async def process_single(image_path):
            try:
                image = self._load(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None

            image_for_translate = image if process_image else None
            detection_result = self.det_model.detect(image)
            if not detection_result:
                # no text detected, return original image
                return image
            ocr_result = self.ocr_engine.get_ocr(image, detection_result)
            resized_image = self.resize_image(image=image, max_size=self.resize_max) if image_for_translate else None  # resize for faster translation if needed
            translation_result = await self._translate(ocr_result, resized_image)
            inpainted_image = self.inpainter.inpaint(image, translation_result)
            return self.renderer.render(inpainted_image, translation_result)

        # create tasks
        tasks = [process_single(path) for path in image_paths]

        results = []
        if _tqdm is None:
            # no tqdm -> just gather
            return await asyncio.gather(*tasks)

        # iterate as tasks complete and update tqdm
        results = await tqdm_asyncio.gather(*tasks, desc="Processing")
        return results

    def resize_image(self, image, max_size=1024):
        width, height = image.size
        if max(width, height) <= max_size:
            return image
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return cv2pil(image.resize((new_width, new_height)))
