import asyncio
import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline import QualityPipeline  # noqa: E402
from manga_translator.schemas.interface import TranslationResult  # noqa: E402

VERSION = "0.2.0"
app = FastAPI(title="MangaTVL Companion")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"^chrome-extension://[a-p]{32}$|^moz-extension://.*$",
                   allow_methods=["*"], allow_headers=["*"])

PIPE: QualityPipeline | None = None
JOBS: dict[str, tuple[float, object]] = {}
OAUTH: dict[str, str] = {}
LOCK = asyncio.Lock()  # one GPU job at a time
RELAY_URL = os.getenv("MTVL_RELAY_URL", "http://127.0.0.1:8787")
TOKEN_FILE = Path(os.getenv("APPDATA", str(REPO))) / "MangaTVL" / "relay_token.txt"
OPENAI_COMPATIBLE = {  # provider -> (base url, default model)
    "relay": (RELAY_URL + "/v1", "google/gemini-2.5-flash-lite"),
    "local": ("http://127.0.0.1:11434/v1", "qwen3-vl:4b-instruct"),  # Ollama
}


@app.middleware("http")
async def only_extensions(request: Request, call_next):
    """Browser requests are accepted only from extensions (a web page can't drive the app)."""
    origin = request.headers.get("origin", "")
    if origin and not (origin.startswith("chrome-extension://") or origin.startswith("moz-extension://")):
        return HTMLResponse("forbidden origin", status_code=403)
    return await call_next(request)


def pipe() -> QualityPipeline:
    """The pipeline, loaded on first use: PyTorch when installed, else all-ONNX (DirectML)."""
    global PIPE
    if PIPE is None:
        try:
            import torch
            device = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            PIPE = QualityPipeline(provider="openrouter", device=device)
        except ImportError:  # packaged app: onnxruntime only
            from manga_translator.detection import build_detector
            models = REPO / "manga_translator" / "assets" / "models"
            device = os.getenv("DEVICE", "dml")
            det = build_detector(models / "best_diplom.onnx", models / "bubble_freetext_yolo26s.onnx", device=device)
            PIPE = QualityPipeline(provider="openrouter", device=device, det_model=det,
                                   lama_onnx=models / "lama_ft2_512.onnx")
    return PIPE


def ollama_ok() -> bool:
    try:
        return requests.get("http://127.0.0.1:11434/api/tags", timeout=0.5).ok
    except requests.RequestException:
        return False


def relay_token() -> str | None:
    """Anonymous device token for the owner's relay, created on first use."""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    try:
        tok = requests.post(RELAY_URL + "/v1/register", timeout=5).json()["token"]
    except Exception:
        return None
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(tok)
    return tok


def relay_ok() -> bool:
    try:
        return requests.post(RELAY_URL + "/v1/chat/completions", timeout=1).status_code in (401, 422)
    except requests.RequestException:
        return False


def translator_for(provider: str, api_key: str | None, model: str | None):
    """A translator for the provider; relay / local are OpenAI-compatible endpoints."""
    if provider in OPENAI_COMPATIBLE:
        from openai import AsyncOpenAI
        from manga_translator.translators import AsyncOpenRouterTranslator
        base, default = OPENAI_COMPATIBLE[provider]
        client = AsyncOpenAI(base_url=base, api_key=api_key or "none")
        return AsyncOpenRouterTranslator(client=client, model=model or default)
    return pipe()._get_translator(provider, pipe()._resolve_key(provider, api_key))


def decode(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1]))).convert("RGB")


def encode(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=92)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


class LLM(BaseModel):
    provider: str = "none"
    api_key: str | None = None
    model: str | None = None


class TranslateReq(BaseModel):
    image: str
    page_url: str | None = None
    from_lang: str = "auto"
    to_lang: str = "thai"
    quality: str = "best"
    llm: LLM = LLM()
    context: dict | None = None


class RenderReq(BaseModel):
    job: str
    translations: list[str]


@app.get("/v1/health")
def health():
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        import onnxruntime as ort
        gpu = "DmlExecutionProvider" in ort.get_available_providers()
    return {"version": VERSION, "gpu": gpu, "local_llm": ollama_ok(), "relay": relay_ok(),
            "providers": ["openrouter", "gemini", "relay", "local", "builtin"]}


@app.post("/v1/translate")
async def translate(req: TranslateReq):
    img = decode(req.image)
    src = None if req.from_lang == "auto" else req.from_lang
    async with LOCK:
        page = await asyncio.to_thread(pipe().analyze, img, src)
    if not page.ocr:
        return {"image": req.image, "context": req.context}
    prov, key = req.llm.provider, req.llm.api_key
    if prov == "relay" or (prov == "none" and relay_ok()):  # zero-account tier
        prov, key = "relay", relay_token()
    if prov in ("none", "builtin"):  # the extension translates with Chrome's on-device Translator
        for jid in [j for j, (t, _) in JOBS.items() if time.time() - t > 600]:
            JOBS.pop(jid, None)
        jid = uuid.uuid4().hex
        JOBS[jid] = (time.time(), (page, req.to_lang))
        return {"needs_client_translation": True, "job": jid, "from_lang": src or "english",
                "to_lang": req.to_lang, "bubbles": [{"n": n, "text": o.text} for n, o in enumerate(page.ocr)]}
    chapter = {"context": req.context}
    try:
        results = await pipe().translate_page(page, translator_for(prov, key, req.llm.model), chapter=chapter,
                                              model=req.llm.model, from_lang=src, to_lang=req.to_lang.title())
    except Exception as e:  # provider errors (bad key, quota) go to the badge
        if prov == "relay" and "429" in str(e):
            raise HTTPException(429, "โควตาฟรีวันนี้หมดแล้ว — กด Sign in with OpenRouter เพื่อแปลต่อ") from e
        raise HTTPException(502, f"{prov}: {e}") from e
    async with LOCK:
        out = await asyncio.to_thread(pipe().finish, page, results, req.to_lang)
    return {"image": encode(out), "context": chapter["context"],
            "bubbles": [{"n": n, "speaker": r.speaker, "text": r.translated_text} for n, r in enumerate(results)]}


@app.post("/v1/render")
async def render(req: RenderReq):
    _, (page, lang) = JOBS.pop(req.job, (0, (None, None)))
    if page is None:
        raise HTTPException(404, "job expired")
    results = [TranslationResult(ocr_result=o, translated_text=t) for o, t in zip(page.ocr, req.translations)]
    async with LOCK:
        out = await asyncio.to_thread(pipe().finish, page, results, lang)
    return {"image": encode(out)}


@app.get("/oauth/callback", response_class=HTMLResponse)
@app.get("/oauth/callback/{path_state}", response_class=HTMLResponse)
def oauth_callback(code: str, state: str = "", path_state: str = ""):
    # OpenRouter does not promise to echo `state`; the extension also puts it in the path
    OAUTH[path_state or state] = code
    return "<h3>เชื่อมต่อ OpenRouter สำเร็จ — ปิดแท็บนี้ได้เลย</h3><script>setTimeout(()=>window.close(),1500)</script>"


@app.get("/oauth/code")
def oauth_code(state: str):
    if state not in OAUTH:
        raise HTTPException(404)
    return {"code": OAUTH.pop(state)}


def main():
    if os.getenv("MTVL_PRELOAD", "1") == "1":
        pipe()
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("MTVL_PORT", "8765")), log_level="info")


if __name__ == "__main__":
    main()
