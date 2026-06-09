import os

from pipeline import Pipeline
from manga_translator.utils.common import convert_img_to_base64
from manga_translator.rendering.fonts import list_fonts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import json
from pydantic import BaseModel
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_prompt = None

# Configurable via env so the same image runs on CPU or GPU without code edits.
DEVICE = os.getenv("DEVICE", "cpu")
PROVIDER = os.getenv("PROVIDER", "openrouter")
MODEL = os.getenv("MODEL", "google/gemini-3-flash-preview")

pipe = Pipeline(provider=PROVIDER, model=MODEL, device=DEVICE,
                user_prompt=user_prompt)


class TranslateRequest(BaseModel):
    images: list[str]          # image URLs collected client-side (e.g. a browser extension reading the rendered DOM)
    process_image: bool = False
    # Per-request overrides (fall back to the server's env/defaults when omitted)
    provider: str | None = None   # openrouter | gemini | groq
    api_key: str | None = None    # provider key from the client; env key used if omitted
    model: str | None = None
    from_lang: str | None = None
    to_lang: str | None = None
    font: str | None = None       # font key from /fonts/ (blank = per-language default)
    text_scale: float | None = None  # >1 = bigger text (fills more of the bubble); blank = server default
    upscale: float | None = None     # output upscale factor (1 = off, e.g. 2); blank = server default
    upscaler: str | None = None      # "lanczos" (fast) | "realesrgan" (AI, opt-in)


@app.post("/translate/")
async def translate_image(request: TranslateRequest):
    results = await pipe.run_batch(
        image_paths=request.images,
        process_image=request.process_image,
        provider=request.provider,
        api_key=request.api_key,
        model=request.model,
        from_lang=request.from_lang,
        to_lang=request.to_lang,
        font=request.font,
        text_scale=request.text_scale,
        upscale=request.upscale,
        upscaler=request.upscaler,
    )
    # Keep index alignment with the request (None = failed) so the client can
    # map each translated image back to its source image and swap it in place.
    images = [convert_img_to_base64(img) if img is not None else None for img in results]
    return {"images": images}


@app.get("/fonts/")
async def fonts():
    """List the fonts the renderer can use, for the client to populate a picker.
    Each item is {key, label, scripts}; pass a chosen `key` as `font` in a
    translate request (blank = the server's per-language default)."""
    return {"fonts": list_fonts()}


@app.post("/translate/stream/")
async def translate_image_stream(request: TranslateRequest):
    """Same inputs as /translate/, but streams one NDJSON line per image as soon
    as it finishes (completion order): {"index": <request index>, "image": <base64|null>}.
    Lets the client swap each page in place without waiting for the whole batch."""
    async def gen():
        async for index, img in pipe.run_batch_stream(
            image_paths=request.images,
            process_image=request.process_image,
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            from_lang=request.from_lang,
            to_lang=request.to_lang,
            font=request.font,
            text_scale=request.text_scale,
            upscale=request.upscale,
            upscaler=request.upscaler,
        ):
            b64 = convert_img_to_base64(img) if img is not None else None
            yield json.dumps({"index": index, "image": b64}) + "\n"

    # Disable proxy/CDN buffering so each NDJSON line reaches the client immediately.
    headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    return StreamingResponse(gen(), media_type="application/x-ndjson", headers=headers)