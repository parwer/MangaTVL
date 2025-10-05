from abc import ABC, abstractmethod
import asyncio

from typing import Optional

from ..utils.common import convert_img_to_base64
from .utils.common import parse_response
from .utils.prompt import get_sys_prompt, get_ocr_prompt

from ..schemas.interface import TranslationResult, OCRResult


class AsyncTranslatorBase(ABC):
    def __init__(self,
                 client: any,
                 model: Optional[str] = None,
                 timeout: int = 60,
                 user_prompt: str="",
                 guidelines: str="",
                 concurrent_limit: int=1,
                 from_lang: str="English",
                 to_lang: str="Thai"
                 ):
        self.client = client
        self.model = model
        self.timeout = timeout
        self.system_prompt = get_sys_prompt(user_prompt=user_prompt,
                                            guidelines=guidelines,
                                            from_lang=from_lang,
                                            to_lang=to_lang)
        self.semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def translate(self, ocr_result: list[OCRResult], image=None):
        if image is not None:
            image = convert_img_to_base64(image)

        processed_inputs = self._preprocess(ocr_result)
        response = await self._translate(processed_inputs, image=image)
        response_json = parse_response(response)

        response_sorted = sorted(response_json, key=lambda x: x['text_no'])

        translation_results = []

        for ocr_item, res_item in zip(ocr_result, response_sorted):
            translation_result = TranslationResult(
                ocr_result=ocr_item,
                translated_text=res_item['translated_text'],
            )
            translation_results.append(translation_result)

        return translation_results

    def _preprocess(self, inputs: list[OCRResult]) -> str:
        user_prompt = ""
        for i, ocr_item in enumerate(inputs):
            data = str({"text_no": i, "text": ocr_item.text})
            user_prompt += data + "\n"
        return get_ocr_prompt(user_prompt)


    @abstractmethod
    async def _translate(self, inputs: str, image=None):
        raise NotImplementedError