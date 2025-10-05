from .translator import AsyncTranslatorBase

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
        response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=content,
                    max_tokens=8000,
                )
        return response.choices[0].message.content