from abc import ABC, abstractmethod
import asyncio
import os

from typing import Optional

from ..utils.common import convert_img_to_base64
from .utils.common import parse_response
from .utils.prompt import get_sys_prompt, get_ocr_prompt, DEFAULT_GUIDELINE_PROMPT

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

    def _merge_guidelines(self, custom_instruction):
        """Effective guidelines for the system prompt: the base guidelines (this
        translator's, or the default set) plus any per-request custom instruction
        from the caller, marked as highest priority so the model heeds it."""
        base = self.guidelines or DEFAULT_GUIDELINE_PROMPT
        if not custom_instruction:
            return base
        return (f"{base}\n\nAdditional user instruction (highest priority, "
                f"override the above if they conflict): {custom_instruction}")

    def _system_prompt(self, from_lang, to_lang, custom_instruction=None):
        key = (from_lang, to_lang, custom_instruction or "")
        if key not in self._prompt_cache:
            self._prompt_cache[key] = get_sys_prompt(
                user_prompt=self.user_prompt,
                guidelines=self._merge_guidelines(custom_instruction),
                from_lang=from_lang, to_lang=to_lang,
            )
        return self._prompt_cache[key]

    async def translate(self, ocr_result: list[OCRResult], image=None, *,
                        model=None, from_lang=None, to_lang=None,
                        custom_instruction=None,
                        capture: dict | None = None) -> list[TranslationResult]:
        """``custom_instruction`` (optional): a free-text instruction from the
        caller appended to the localization guidelines for this call only.

        ``capture`` (optional): when a dict is passed it is filled with the raw
        request/response of this call — system prompt, the preprocessed user input,
        whether an image was attached, the raw model response, token usage, the
        parsed ``text_no`` map, and any error. Production callers omit it (no
        behaviour change); the eval harness passes a sink to log everything."""
        if not ocr_result:
            return []

        if image is not None:
            image = convert_img_to_base64(image)

        model = model or self.model
        system_prompt = self._system_prompt(from_lang or self.from_lang,
                                            to_lang or self.to_lang,
                                            custom_instruction=custom_instruction)

        inputs = self._preprocess(ocr_result)
        if capture is not None:
            capture.update(model=model, system_prompt=system_prompt,
                           user_input=inputs, image_attached=image is not None)
        # Only forward `capture` when asked, so a custom `_translate` override
        # that doesn't accept the kwarg keeps working (full backward-compat).
        extra = {"capture": capture} if capture is not None else {}
        try:
            response = await self._translate(inputs, image=image, model=model,
                                             system_prompt=system_prompt, **extra)
        except Exception as e:
            print(f"[translate] API call failed: {e}")
            response = None
            if capture is not None:
                capture["error"] = str(e)

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
        if capture is not None:
            capture["parsed"] = dict(by_no)

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
    async def _translate(self, inputs: str, image=None, *, model=None, system_prompt=None, capture=None):
        raise NotImplementedError


def _usage_to_dict(usage):
    """Best-effort conversion of a provider usage object to a plain dict."""
    if usage is None:
        return None
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    keys = ("prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_token_count", "candidates_token_count", "total_token_count")
    out = {k: getattr(usage, k) for k in keys if getattr(usage, k, None) is not None}
    return out or str(usage)


class OpenAICompatibleTranslator(AsyncTranslatorBase):
    """Shared implementation for any OpenAI-compatible chat client (OpenRouter,
    Groq, ...) that exposes ``client.chat.completions.create``."""

    async def _translate(self, inputs: str, image=None, *, model=None, system_prompt=None, capture=None):
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
        text = response.choices[0].message.content if response else None
        if capture is not None:
            capture["raw_response"] = text
            capture["usage"] = _usage_to_dict(getattr(response, "usage", None)) if response else None
        return text
