import base64

from google.genai import types
from google.genai.types import GenerateContentConfig

from .translator import AsyncTranslatorBase


class AsyncGeminiTranslator(AsyncTranslatorBase):
    async def _translate(self, inputs: str, image=None, *, model=None, system_prompt=None, capture=None):
        contents = [inputs]
        if image is not None:
            # Send the image as a real Part, not a data-URI string (which would be
            # treated as text). `image` is "data:image/jpeg;base64,...." here.
            raw = base64.b64decode(image.split(",", 1)[1])
            contents.insert(0, types.Part.from_bytes(data=raw, mime_type="image/jpeg"))

        response = await self._call_with_retry(
            lambda: self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=GenerateContentConfig(system_instruction=system_prompt),
            )
        )
        text = response.text if response else None
        if capture is not None:
            from .translator import _usage_to_dict
            capture["raw_response"] = text
            capture["usage"] = _usage_to_dict(getattr(response, "usage_metadata", None)) if response else None
        return text
