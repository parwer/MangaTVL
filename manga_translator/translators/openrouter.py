from .translator import AsyncTranslatorBase
import asyncio

class AsyncOpenRouterTranslator(AsyncTranslatorBase):
    async def _translate(self, inputs: str, image=None):
        content = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": inputs}
        ]

        if image is not None:
            content.insert(1, {"role": "user", "content": image})

        response = await self._call_api(content)
        return response

    async def _call_api(self, content):
        for attempt in range(self.max_retries):
            if self.RATE_LIMIT_FLAGS.is_set():
                await self.RATE_LIMIT_FLAGS.wait()
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                                model=self.model,
                                messages=content,
                                max_tokens=8000,
                            )
                    return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e):
                    self.RATE_LIMIT_FLAGS.set()
                    wait_time = 2 ** attempt
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
                    self.RATE_LIMIT_FLAGS.clear()
                else:
                    raise e
