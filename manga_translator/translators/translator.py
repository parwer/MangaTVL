from abc import ABC, abstractmethod
import asyncio
import os

from typing import Optional

from ..utils.common import convert_img_to_base64
from .utils.common import parse_response
from .utils.prompt import get_sys_prompt, get_ocr_prompt

from ..schemas.interface import TranslationResult, OCRResult

from dotenv import load_dotenv
load_dotenv()


class AsyncTranslatorBase(ABC):
    def __init__(self,
                 client: any,
                 model: Optional[str] = None,
                 timeout: int = 60,
                 user_prompt: str = "",
                 guidelines: str = "",
                 concurrent_limit: int = os.getenv("CONCURRENT_REQUESTS") or 1,
                 from_lang: str = "English",
                 to_lang: str = "Thai",
                 max_retries: int = 6,
                 max_tokens: int = 8000,
                 ):
        self.client = client
        self.model = model
        self.timeout = timeout
        # Prompt components kept as defaults; the system prompt is built per-call
        # so from_lang/to_lang can be overridden per request.
        self.user_prompt = user_prompt
        self.guidelines = guidelines
        self.from_lang = from_lang
        self.to_lang = to_lang
        self._prompt_cache = {}
        self.semaphore = asyncio.Semaphore(int(concurrent_limit))
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.RATE_LIMIT_FLAGS = asyncio.Event()

    def _system_prompt(self, from_lang, to_lang):
        key = (from_lang, to_lang)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = get_sys_prompt(
                user_prompt=self.user_prompt, guidelines=self.guidelines,
                from_lang=from_lang, to_lang=to_lang,
            )
        return self._prompt_cache[key]

    async def translate(self, ocr_result: list[OCRResult], image=None, *,
                        model=None, from_lang=None, to_lang=None) -> list[TranslationResult]:
        if not ocr_result:
            return []

        if image is not None:
            image = convert_img_to_base64(image)

        model = model or self.model
        system_prompt = self._system_prompt(from_lang or self.from_lang, to_lang or self.to_lang)

        inputs = self._preprocess(ocr_result)
        try:
            response = await self._translate(inputs, image=image, model=model, system_prompt=system_prompt)
        except Exception as e:
            print(f"[translate] API call failed: {e}")
            response = None

        # Map by text_no (not positional zip) so missing/extra/reordered items
        # from the LLM don't shift every translation into the wrong bubble.
        by_no = {}
        if response:
            for r in parse_response(response):
                if isinstance(r, dict) and "text_no" in r and "translated_text" in r:
                    try:
                        by_no[int(r["text_no"])] = r["translated_text"]
                    except (ValueError, TypeError):
                        continue

        results = []
        for i, ocr_item in enumerate(ocr_result):
            translated = by_no.get(i)
            if translated is None:
                translated = ocr_item.text  # fallback: keep original text
            results.append(TranslationResult(ocr_result=ocr_item, translated_text=translated))
        return results

    async def _call_with_retry(self, do_call):
        """Shared rate-limit + exponential-backoff retry. ``do_call`` is a
        zero-arg callable returning the provider's API coroutine."""
        for attempt in range(self.max_retries):
            if self.RATE_LIMIT_FLAGS.is_set():
                await self.RATE_LIMIT_FLAGS.wait()
            try:
                async with self.semaphore:
                    return await do_call()
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    self.RATE_LIMIT_FLAGS.set()
                    wait_time = 2 ** attempt
                    print(f"Rate limit exceeded. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    self.RATE_LIMIT_FLAGS.clear()
                else:
                    raise
        return None

    def _preprocess(self, inputs: list[OCRResult]) -> str:
        user_prompt = ""
        for i, ocr_item in enumerate(inputs):
            data = str({"text_no": i, "text": ocr_item.text})
            user_prompt += data + "\n"
        return get_ocr_prompt(user_prompt)

    @abstractmethod
    async def _translate(self, inputs: str, image=None, *, model=None, system_prompt=None):
        raise NotImplementedError


class OpenAICompatibleTranslator(AsyncTranslatorBase):
    """Shared implementation for any OpenAI-compatible chat client (OpenRouter,
    Groq, ...) that exposes ``client.chat.completions.create``."""

    async def _translate(self, inputs: str, image=None, *, model=None, system_prompt=None):
        if image is not None:
            # Image must be a multimodal content part, not a raw data-URI string
            # in `content` (that gets tokenized as text -> ~2k-4k tokens/image).
            user_content = [
                {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
                {"type": "text", "text": inputs},
            ]
        else:
            user_content = inputs

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await self._call_with_retry(
            lambda: self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
        )
        return response.choices[0].message.content if response else None
