# MangaTVL

A FastAPI service that automatically translates manga / comic pages from **English(or any that llm support) → Thai** (configurable) through a 5-stage computer-vision + LLM pipeline. Its core idea: translation is done by a **VLM (vision-language model)** that receives the full page image alongside the OCR'd text, so it can use the surrounding art, panel layout, and visual context to translate more accurately — not just the raw text in isolation. The pipeline detects speech bubbles, reads the text, sends both the text and the page image to the LLM for context-aware translation, erases the original text, and renders the translation back onto the image.

---

## How it works

Each image flows through five swappable stages, orchestrated by `Pipeline` in [`pipeline.py`](pipeline.py):

```
Input image
  │
  ▼
[1] Detection ── find speech-bubble / text boxes (+ polygon)   → DetectionResult[]
  │              ONNX YOLO model, or ultralytics YOLO (YoloDetection)
  ▼
[2] OCR ─────── read text inside each bubble crop              → OCRResult[]
  │              EasyOCR (default) / manga-ocr (Japanese) / PaddleOCR
  │              chosen per source language (per-bubble crop + polygon clean)
  ▼
[3] Translation  send the whole page's text + the page image   → TranslationResult[]
  │              to a VLM at once for context-aware translation
  │              OpenRouter / Gemini / Groq (async, JSON output)
  ▼
[4] Inpainting ─ erase the original text from the image        → cleaned image
  │              OpenCV (telea) / Simple-LaMa; polygon-aware safe-fill
  ▼
[5] Rendering ── draw the translated text into the bubble    → output image
                 Pillow + font, auto font-size, language-aware line-breaking
                 (pythainlp for Thai, per-char for CJK, spaces otherwise)
```

The stages communicate through Pydantic models in [`manga_translator/schemas/interface.py`](manga_translator/schemas/interface.py):
`DetectionResult` → `OCRResult` → `TranslationResult`. The `DetectionResult` (with its bubble `bbox` and segmentation `polygon`) is threaded through to the end, so the inpaint and render stages can use the real bubble shape.

| Stage | Tech | Main file |
|-------|------|-----------|
| Detection | ONNX Runtime / ultralytics YOLO | [`detection/`](manga_translator/detection/) |
| OCR | EasyOCR / manga-ocr (Japanese) / PaddleOCR | [`ocr/`](manga_translator/ocr/) |
| Translation | OpenRouter / Gemini / Groq VLM (async, text + image) | [`translators/`](manga_translator/translators/) |
| Inpainting | OpenCV / Simple-LaMa | [`inpainting/`](manga_translator/inpainting/) |
| Rendering | balloon-shape-aware layout, HarfBuzz shaping (Thai marks), page-level size planning | [`rendering/`](manga_translator/rendering/) |
| Serving | FastAPI companion app | [`apps/companion/`](apps/companion/) |

### Quality mode (`QualityPipeline`)

`QualityPipeline` (used by the companion app) works on the letters instead of boxes:

```
detect (balloons + free text: HybridDetection)
  → glyph mask (masking.TextMasker, comic-text-detector) → region repairs (detection/refine.py)
  → OCR without double reads (ocr/region_ocr.py)
  → page translation with numbered tags, speakers and a running chapter context (translators.ContextTranslator)
  → erase only the lettering: flat fill or LaMa, verify / repair, revert regions that stay dirty (inpainting.TextEraser)
  → letter with the original glyph masks (rendering.TextRenderer)
```
`run_batch` / `run_batch_stream` treat the images as one chapter in reading order (pages run one after another
so each gets the context of the previous ones). It needs the model files below and a GPU (or DirectML) to be fast.

---

## Requirements

- **Python 3.12** (the project ships with an embedded interpreter at `../MangaTVL_ENV/python.exe`)
- Model files under `manga_translator/assets/models/` (**not** committed):

  | file | used by |
  |---|---|
  | `best_diplom.pt` (or `.onnx`) | balloon detector (all modes) |
  | `bubble_freetext_yolo26s.pt` (or `.onnx`) | free-text detector (`FREE_TEXT_MODEL=default`, quality mode) |
  | `comictextdetector.pt.onnx` | glyph mask (quality mode) |
  | `lama_ft2.pt` (or `lama_large_512px.ckpt`), `lama_ft2_512.onnx` | LaMa text eraser (quality mode; ONNX for the packaged app) |
  | `baberu/` | Japanese OCR without torch (packaged app) |
- Fonts are in `manga_translator/assets/fonts/` (Thai default: Mali Medium / Bold)
- One LLM provider API key (see Configuration)

> `requirement.txt` is incomplete. Expect to also install: `fastapi`, `uvicorn`, `python-dotenv`, `beautifulsoup4`, `requests`, `pillow`, `tqdm`, `pythainlp`, `ultralytics`, `simple-lama-inpainting`, and the LLM SDKs (`openai`, `groq`, `google-genai`).

---

## Installation

Run everything from the `MangaTVL/` directory.

```powershell
..\MangaTVL_ENV\python.exe -m pip install -r requirement.txt   # note: "requirement.txt" (no s)
..\MangaTVL_ENV\python.exe -m pip install fastapi uvicorn python-dotenv beautifulsoup4 requests pillow tqdm pythainlp ultralytics simple-lama-inpainting openai groq google-genai
```

---

## Configuration

Copy `.env.example` → `.env` and set **one** provider key:

```ini
OPENROUTER_API_KEY=...           # or GROQ_API_KEY / GOOGLE_API_KEY
MODEL=google/gemini-3-flash-preview
CONCURRENT_REQUESTS=4            # max concurrent LLM calls
```

> Note: the Gemini provider reads `GOOGLE_API_KEY` (not `GEMINI_API_KEY`).

The old hosted API (`main.py`, Docker images) and the Tampermonkey userscript (`tempomunkey.user.js`) are no
longer used; they are kept on the `archive/userscript-api` branch.

---

## Using the pipeline directly

```python
import asyncio
from pipeline import Pipeline
from manga_translator.detection.yolo_detection import YoloDetection

pipe = Pipeline(
    det_model=YoloDetection(task="segment"),   # polygon-aware detection
    provider="openrouter",
    model="google/gemini-3-flash-preview",
    device="cuda",
)

result = asyncio.run(pipe.run("path/or/url/to/page.jpg", process_image=True))
result.save("translated.jpg")
```

`Pipeline.run` handles a single image; `Pipeline.run_batch` processes many concurrently (API concurrency is throttled by `CONCURRENT_REQUESTS`).

---

## Browser extension + companion app (no API key for the user)

[`apps/`](apps/) is the end-user client: a Chrome MV3 extension (one button on any reader page) and a local
companion app that runs `QualityPipeline` on the user's GPU.

```powershell
python -m apps.companion.server          # http://127.0.0.1:8765, accepts requests from extensions only
# MTVL_DEVICE=cuda|dml|cpu (default: cuda with torch, dml without)
# MODEL=<openrouter model> (default google/gemini-3-flash-preview)
# Chrome -> chrome://extensions -> Developer mode -> Load unpacked -> apps/extension
```
Translation tiers, picked automatically by the extension: the owner's relay (no account), Sign in with
OpenRouter (OAuth PKCE, one click), a Gemini key, a local model (Ollama), or Chrome's on-device Translator
(the companion OCRs and erases; the extension translates; `/v1/render` letters the result).
Companion endpoints: `GET /v1/health`, `POST /v1/translate`, `POST /v1/render`, `GET /oauth/callback`, `GET /oauth/code`.

The relay ([`apps/relay/relay.py`](apps/relay/relay.py)) is a small FastAPI app the owner hosts with their
provider key (`OWNER_OPENROUTER_KEY`): anonymous device tokens, a daily page quota, OpenAI-compatible pass-through.

### Packaged app (no PyTorch, any Windows GPU)

```powershell
python packaging/stage.py --clean                       # -> packaging/build/app (code + ONNX models)
<torch-free venv>\python packaging/smoke_torchfree.py page.png   # full page, asserts torch is absent
<torch-free venv>\python packaging/build_exe.py         # -> packaging/dist/MangaTVL-Companion/
```
The torch-free venv has `onnxruntime-directml opencv-python-headless pillow pydantic fastapi uvicorn requests
openai python-dotenv pythainlp uharfbuzz freetype-py rapidocr pyinstaller`. Without torch the pipeline uses
ONNX detection, ONNX LaMa, RapidOCR (Baberu for Japanese) and DirectML.

---

## Project layout

```
MangaTVL/
├── pipeline.py                 # Pipeline + QualityPipeline orchestrators
├── manga_translator/
│   ├── detection/              # YOLO / ONNX detectors, HybridDetection, refine.py (region repairs)
│   ├── masking/                # TextMasker (glyph mask), per-region glyph masks
│   ├── ocr/                    # EasyOCR, manga-ocr, PaddleOCR, RapidOCR, Baberu, region_ocr.py
│   ├── translators/            # OpenRouter / Gemini / Groq, ContextTranslator
│   ├── inpainting/             # OpenCV / Simple-LaMa / LaMa (own arch), TextEraser
│   ├── rendering/              # TextRenderer, typeset, HarfBuzz shaper, fonts
│   ├── schemas/interface.py    # DetectionResult / OCRResult / TranslationResult
│   ├── utils/                  # image + geometry helpers, languages
│   └── assets/                 # fonts, models (not committed)
├── apps/                       # extension, companion app, relay
├── packaging/                  # torch-free companion build
└── tests/                      # pytest suite (no model files or API keys needed)
```

---

## Tests

```powershell
..\MangaTVL_ENV\python.exe -m pip install -r requirements-dev.txt
..\MangaTVL_ENV\python.exe -m pytest
```

The suite covers pure logic only (geometry helpers, LLM-response parsing, `text_no` translation mapping, detector output building, polygon cleaning, inpaint mask selection) — it does not require model weights or API keys.

---

## Notes

- Detection model weights (`*.onnx` / `*.pt`) are gitignored and must be supplied separately.
- Translation failures fall back to the original text per bubble rather than dropping the whole page.
