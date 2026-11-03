import asyncio
import base64
import heapq
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
from pipeline import QualityPipeline, glossary_words  # noqa: E402
from manga_translator.schemas.interface import TranslationResult  # noqa: E402

VERSION = "0.3.2"
MODEL = os.getenv("MODEL") or "google/gemini-3-flash-preview"
# MTVL_DEVICE, not DEVICE: the repo .env sets DEVICE=cpu for main.py, which put the companion's
# glyph mask and LaMa on the CPU (minutes per page, the whole PC froze)
DEVICE = os.getenv("MTVL_DEVICE")
app = FastAPI(title="MangaTVL Companion")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"^chrome-extension://[a-p]{32}$|^moz-extension://.*$",
                   allow_methods=["*"], allow_headers=["*"], allow_private_network=True)  # Chrome local-network preflight

PIPE: QualityPipeline | None = None
JOBS: dict[str, dict] = {}  # job id -> page state, translation results and the render task
OAUTH: dict[str, str] = {}
RENDER, ANALYZE = 0, 1  # GPU priorities: lettering a translated page goes before analysing the next


class GpuQueue:
    """One GPU job at a time, lowest priority number first (FIFO within a priority)."""

    def __init__(self):
        self._busy, self._wait, self._n = False, [], 0

    async def run(self, prio: int, fn, *args):
        if self._busy:
            fut = asyncio.get_running_loop().create_future()
            self._n += 1
            heapq.heappush(self._wait, (prio, self._n, fut))
            try:
                await fut  # handed over by the job before us
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():  # handed over, then cancelled: pass it on
                    self._release()
                raise
        self._busy = True
        try:
            return await asyncio.to_thread(fn, *args)
        finally:
            self._release()

    def _release(self):
        self._busy = False
        while self._wait:
            fut = heapq.heappop(self._wait)[2]
            if not fut.done():  # skip waiters whose request was cancelled
                self._busy = True
                fut.set_result(None)
                break


GPU = GpuQueue()
# Relay (apps/relay): the owner's OpenRouter key behind a free daily quota. Off until it is deployed somewhere.
# RELAY_URL = os.getenv("MTVL_RELAY_URL", "http://127.0.0.1:8787")
# TOKEN_FILE = Path(os.getenv("APPDATA", str(REPO))) / "MangaTVL" / "relay_token.txt"
OPENAI_COMPATIBLE = {  # provider -> (base url, default model)
    # "relay": (RELAY_URL + "/v1", "google/gemini-2.5-flash-lite"),
    "local": ("http://127.0.0.1:11434/v1", "qwen3-vl:4b-instruct"),  # Ollama; any other URL from the popup
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
            device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
            PIPE = QualityPipeline(provider="openrouter", model=MODEL, device=device)
        except ImportError:  # packaged app: onnxruntime only
            from manga_translator.detection import build_detector
            models = REPO / "manga_translator" / "assets" / "models"
            device = DEVICE or "dml"
            # detectors on CPU (0.22 s); on DirectML each new page size recompiles (0.55 s)
            det = build_detector(models / "best_diplom.onnx", models / "bubble_freetext_yolo26s.onnx", device="cpu")
            PIPE = QualityPipeline(provider="openrouter", model=MODEL, device=device, det_model=det,
                                   lama_onnx=models / "lama_ft2_512.onnx")
    return PIPE


def ollama_ok() -> bool:
    try:
        return requests.get("http://127.0.0.1:11434/api/tags", timeout=0.5).ok
    except requests.RequestException:
        return False


# Relay off (see RELAY_URL above).
# def relay_token() -> str | None:
#     """Anonymous device token for the owner's relay, created on first use."""
#     if TOKEN_FILE.exists():
#         return TOKEN_FILE.read_text().strip()
#     try:
#         tok = requests.post(RELAY_URL + "/v1/register", timeout=5).json()["token"]
#     except Exception:
#         return None
#     TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
#     TOKEN_FILE.write_text(tok)
#     return tok
#
#
# def relay_ok() -> bool:
#     try:
#         return requests.post(RELAY_URL + "/v1/chat/completions", timeout=1).status_code in (401, 422)
#     except requests.RequestException:
#         return False


def local_models(base_url: str) -> list[str]:
    """Model ids served by an OpenAI-compatible server (Ollama, LM Studio, llama.cpp, vLLM...)."""
    r = requests.get(base_url.rstrip("/") + "/models", timeout=3)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


VISION_HINTS = ("-vl", "vl:", "gemma3", "gemma4", "gemma-3", "gemma-4", "llava", "minicpm-v", "pixtral",
                "vision", "llama4", "mistral-small3")


def pick_local_model(models: list[str]) -> str | None:
    """An installed model for a blank Model field: one that reads images first, else the first one."""
    usable = [m for m in models if "embed" not in m.lower()]
    return next((m for m in usable if any(h in m.lower() for h in VISION_HINTS)), usable[0] if usable else None)


def translator_for(provider: str, api_key: str | None, model: str | None, base_url: str | None = None):
    """A translator for the provider; local (and the relay, when on) are OpenAI-compatible endpoints."""
    if provider in OPENAI_COMPATIBLE:
        from openai import AsyncOpenAI
        from manga_translator.translators import AsyncOpenRouterTranslator
        base, default = OPENAI_COMPATIBLE[provider]
        if provider == "local":
            base = (base_url or base).rstrip("/")
            if not model:  # blank: what this server really has (the default may not be installed)
                try:
                    model = pick_local_model(local_models(base))
                except Exception:
                    model = None
        client = AsyncOpenAI(base_url=base, api_key=api_key or "none")
        return AsyncOpenRouterTranslator(client=client, model=model or default)
    tr = pipe()._get_translator(provider, pipe()._resolve_key(provider, api_key))
    tr.model = tr.model or MODEL
    return tr


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
    base_url: str | None = None  # local: OpenAI-compatible server, e.g. LM Studio http://127.0.0.1:1234/v1


class TranslateReq(BaseModel):
    image: str
    page_url: str | None = None
    from_lang: str = "auto"
    to_lang: str = "thai"
    quality: str = "best"
    llm: LLM = LLM()
    context: dict | None = None
    send_image: bool = True  # False: the LLM gets the OCR text only


class AnalyzeReq(BaseModel):
    image: str
    from_lang: str = "auto"
    to_lang: str = "thai"
    quality: str = "best"  # best | fast (no foreign-text filter, Telea instead of LaMa)


class JobReq(BaseModel):
    job: str
    llm: LLM = LLM()
    context: dict | None = None  # None: translate the page on its own (no chapter context)
    send_image: bool = True  # False: the LLM gets the OCR text only (text-only models, fewer tokens)


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
    # relay off: "relay": relay_ok() and "relay" in providers
    # no GPU: each page takes about 30 s on the CPU and always runs at "fast" quality
    cpu_only = getattr(PIPE, "cpu_only", False) if PIPE is not None else (DEVICE == "cpu" or not gpu)
    return {"version": VERSION, "gpu": gpu, "cpu_only": cpu_only, "local_llm": ollama_ok(), "relay": False, "jobs": True,
            "providers": ["openrouter", "gemini", "local", "builtin"]}


def _expire():
    for jid in [j for j, v in JOBS.items() if time.time() - v["t"] > 1800]:
        JOBS.pop(jid, None)


def lang_label(name: str | None) -> str | None:
    """Language name for the LLM prompt ("chinese_traditional" -> "Chinese Traditional")."""
    return name.replace("_", " ").title() if name else None


async def _analyze(image: str, from_lang: str, to_lang: str, quality: str = "best") -> dict:
    src = None if from_lang == "auto" else from_lang
    fast = quality == "fast" or getattr(pipe(), "cpu_only", False)  # CPU: LaMa adds up to ~20 s a page
    page = await GPU.run(ANALYZE, pipe().analyze_gpu, decode(image), src, fast)
    # second OCR reading is CPU only: off the GPU queue so the next page's detection runs meanwhile
    page = await asyncio.to_thread(pipe().analyze_cpu, page, src)
    _expire()
    jid = uuid.uuid4().hex
    JOBS[jid] = {"t": time.time(), "page": page, "src": src, "to_lang": to_lang, "task": None}
    return {"job": jid, "regions": len(page.ocr), "from_lang": src or "english", "to_lang": to_lang,
            "bubbles": [{"n": n, "text": o.text} for n, o in enumerate(page.ocr)]}


def _job(jid: str) -> dict:
    if jid not in JOBS:
        raise HTTPException(404, "job expired")
    return JOBS[jid]


def _start_render(job: dict, results, keep_words=()):
    async def render():
        erased = await GPU.run(RENDER, pipe().erase, job["page"], results)
        # lettering is CPU only: off the GPU queue so the next page's analysis runs meanwhile
        out = await asyncio.to_thread(pipe().letter, job["page"], erased, job["to_lang"], None, None, keep_words)
        return await asyncio.to_thread(encode, out)
    job["task"] = asyncio.create_task(render())


async def _translate_job(job: dict, llm: LLM, context: dict | None, send_image: bool = True) -> dict:
    """LLM step only; lettering is queued on the GPU and fetched with /v1/result."""
    page = job["page"]
    if not page.ocr:
        job["task"] = None
        return {"context": context, "bubbles": []}
    prov, key = llm.provider, llm.api_key
    # relay off: zero-account tier
    # if prov == "relay" or (prov == "none" and relay_ok()):
    #     prov, key = "relay", relay_token()
    if prov in ("none", "builtin"):  # the extension translates with Chrome's on-device Translator
        return {"needs_client_translation": True, "context": context}
    chapter = {"context": context}
    try:
        tr = translator_for(prov, key, llm.model, llm.base_url)
        results = await pipe().translate_page(page, tr, chapter=chapter,
                                              model=llm.model, from_lang=lang_label(job["src"]), to_lang=lang_label(job["to_lang"]),
                                              send_image=send_image)
    except Exception as e:  # provider errors (bad key, quota) go to the badge
        # relay off: its daily quota
        # if prov == "relay" and "429" in str(e):
        #     raise HTTPException(429, "โควตาฟรีวันนี้หมดแล้ว — กด Sign in with OpenRouter เพื่อแปลต่อ") from e
        raise HTTPException(502, f"{prov}: {e}") from e
    _start_render(job, results, glossary_words(chapter["context"]))
    return {"context": chapter["context"],
            "bubbles": [{"n": n, "speaker": r.speaker, "text": r.translated_text} for n, r in enumerate(results)]}


@app.post("/v1/analyze")
async def analyze(req: AnalyzeReq):
    """Detect, mask and OCR a page; returns a job id for /v1/job."""
    return await _analyze(req.image, req.from_lang, req.to_lang, req.quality)


@app.post("/v1/job")
async def translate_job(req: JobReq):
    """Translate an analysed page; returns the new context as soon as the LLM answers."""
    return await _translate_job(_job(req.job), req.llm, req.context, req.send_image)


@app.get("/v1/result/{jid}")
async def result(jid: str):
    """The lettered page (waits for its GPU turn); no image when the page had no text."""
    job = _job(jid)
    if job["task"] is None:
        JOBS.pop(jid, None)
        return {"image": None}
    try:
        return {"image": await asyncio.shield(job["task"])}
    finally:
        if job["task"].done():
            JOBS.pop(jid, None)


@app.post("/v1/translate")
async def translate(req: TranslateReq):
    """One page in one call (analyze + job + result)."""
    a = await _analyze(req.image, req.from_lang, req.to_lang, req.quality)
    job = JOBS[a["job"]]
    if not a["regions"]:
        JOBS.pop(a["job"], None)
        return {"image": req.image, "context": req.context}
    t = await _translate_job(job, req.llm, req.context, req.send_image)
    if t.get("needs_client_translation"):
        return {**a, "needs_client_translation": True}
    out = await result(a["job"])
    return {"image": out["image"], "context": t["context"], "bubbles": t["bubbles"]}


@app.post("/v1/render")
async def render(req: RenderReq):
    """Letter a page with translations made by the extension (builtin flow)."""
    job = _job(req.job)
    results = [TranslationResult(ocr_result=o, translated_text=t) for o, t in zip(job["page"].ocr, req.translations)]
    _start_render(job, results)
    return await result(req.job)


@app.get("/oauth/callback", response_class=HTMLResponse)
@app.get("/oauth/callback/{path_state}", response_class=HTMLResponse)
def oauth_callback(code: str, state: str = "", path_state: str = ""):
    # OpenRouter does not promise to echo `state`; the extension also puts it in the path
    OAUTH[path_state or state] = code
    return "<h3>เชื่อมต่อ OpenRouter สำเร็จ — ปิดแท็บนี้ได้เลย</h3><script>setTimeout(()=>window.close(),1500)</script>"


@app.get("/v1/local-models")
def list_local_models(url: str = OPENAI_COMPATIBLE["local"][0]):
    """What a local OpenAI-compatible server offers (popup check before translating)."""
    try:
        return {"ok": True, "models": local_models(url)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "models": []}


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
