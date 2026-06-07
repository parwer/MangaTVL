import base64

from google.genai import types
from google.genai.types import GenerateContentConfig

from .translator import AsyncTranslatorBase


class AsyncGeminiTranslator(AsyncTranslatorBase):
    async def _translate(self, inputs: str, image=None):
        contents = [inputs]
        if image is not None:
            # Send the image as a real Part, not a data-URI string (which would be
            # treated as text). `image` is "data:image/jpeg;base64,...." here.
            raw = base64.b64decode(image.split(",", 1)[1])
            contents.insert(0, types.Part.from_bytes(data=raw, mime_type="image/jpeg"))

        response = await self._call_with_retry(
            lambda: self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=GenerateContentConfig(system_instruction=self.system_prompt),
            )
        )
        return response.text if response else None
