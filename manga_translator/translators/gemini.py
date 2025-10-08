import asyncio
from .translator import AsyncTranslatorBase
from google.genai.types import GenerateContentConfig

class AsyncGeminiTranslator(AsyncTranslatorBase):
    async def _translate(self, inputs: str, image=None):
        content = [inputs]

        if image is not None:
            content.insert(0, image)

        response = await self._call_api(content)
        return response

    async def _call_api(self, content):
        for attempt in range(self.max_retries):
            if self.RATE_LIMIT_FLAGS.is_set():
                await self.RATE_LIMIT_FLAGS.wait()
            try:
                async with self.semaphore:
                    response = await self.client.aio.models.generate_content(
                        model=self.model,
                        contents=content,
                        config=GenerateContentConfig(
                            system_instruction=self.system_prompt
                        )
                    )
                    return response.text
            except Exception as e:
                if "429" in str(e):
                    self.RATE_LIMIT_FLAGS.set()
                    wait_time = 2 ** attempt
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                    self.RATE_LIMIT_FLAGS.clear()
                else:
                    raise e
