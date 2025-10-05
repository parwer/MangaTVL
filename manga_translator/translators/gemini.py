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
        response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=content,
                    config=GenerateContentConfig(
                        system_instruction=self.system_prompt
                    )
                )
        return response.text