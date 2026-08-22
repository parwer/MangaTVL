# MangaTVL

A FastAPI service that automatically translates manga / comic pages from **English → Thai** (configurable) through a 5-stage computer-vision + LLM pipeline. It detects speech bubbles, reads the text, translates it with an LLM, erases the original text, and renders the translation back onto the image.

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
[3] Translation  send the whole page's text to an LLM at once  → TranslationResult[]
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
| Translation | OpenRouter / Gemini / Groq (async) | [`translators/`](manga_translator/translators/) |
| Inpainting | OpenCV / Simple-LaMa | [`inpainting/`](manga_translator/inpainting/) |
| Rendering | Pillow + language-aware line-breaking (pythainlp / per-char / spaces) | [`rendering/renderer.py`](manga_translator/rendering/renderer.py) |
| Serving | FastAPI | [`main.py`](main.py) |

---

## Requirements

- **Python 3.12** (the project ships with an embedded interpreter at `../MangaTVL_ENV/python.exe`)
- A detection model weight under `manga_translator/assets/models/` (e.g. `best_diplom.pt` for `YoloDetection`, or a `.onnx` for `ONNXDetection`) — model weights are **not** committed
- A Thai font at `manga_translator/assets/fonts/THSarabunNew.ttf`
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

Copy `.env.example` → `.env` and set **one** provider key plus the pipeline config:

```ini
OPENROUTER_API_KEY=...           # or GROQ_API_KEY / GOOGLE_API_KEY
DEVICE=cpu                       # cpu | cuda
PROVIDER=openrouter              # openrouter | gemini | groq
MODEL=google/gemini-3-flash-preview
CONCURRENT_REQUESTS=4            # max concurrent LLM calls
```

`main.py` reads `DEVICE`/`PROVIDER`/`MODEL` from the environment, so the same image runs on CPU or GPU without code changes.

> Note: the Gemini provider reads `GOOGLE_API_KEY` (not `GEMINI_API_KEY`).

---

## Run with Docker (recommended for deployment)

Model weight (`best_diplom.pt`) and the Thai font are baked into the image; only the API keys are passed at runtime via `.env`.

**CPU (default):**
```bash
cp .env.example .env        # fill in a provider key
docker compose up --build   # builds Dockerfile, serves on :8000
```
or manually:
```bash
docker build -t mangatvl-api .
docker run -p 8000:8000 --env-file .env mangatvl-api
```

**GPU (CUDA):** build with the GPU Dockerfile, set `DEVICE=cuda`, and pass `--gpus all`:
```bash
docker build -f Dockerfile.gpu -t mangatvl-api-gpu .
docker run --gpus all -e DEVICE=cuda -p 8000:8000 --env-file .env mangatvl-api-gpu
```

Once running, the API is at `http://localhost:8000` (interactive docs at `/docs`).

---

## Running the API (without Docker)

```powershell
..\MangaTVL_ENV\python.exe -m uvicorn main:app --reload
```

`main.py` configures the pipeline (provider, model, device) at the top — edit it to change providers or pass custom detector/OCR/inpainter/renderer instances.

### Endpoint

`POST /translate/`

```json
{
  "images": ["https://cdn.example.com/p1.jpg", "https://cdn.example.com/p2.jpg"],
  "process_image": false,
  "provider": "openrouter",
  "api_key": "sk-...",
  "model": "google/gemini-2.5-flash",
  "from_lang": "english",
  "to_lang": "thai",
  "font": "itim",
  "text_scale": 1.2,
  "upscale": 2,
  "upscaler": "lanczos",
  "custom_instruction": "keep character honorifics; use a polite tone"
}
```

- `images` (required): a **list of image URLs** collected client-side (e.g. a browser extension that reads the rendered manga-reader DOM). The full pipeline runs on each and returns base64-encoded translated images.
- The response list is **index-aligned** with the request (`null` for any image that failed), so the client can map each result back to its source image and swap it in place.
- `process_image: true` also sends the (downscaled) page to the LLM as **visual context** for better translation. Images are sent as proper multimodal parts, so the token cost stays low.
- `provider` / `api_key` / `model` / `from_lang` / `to_lang` are **optional per-request overrides**. Omit any of them to fall back to the server's defaults (and the env API key for that provider). `provider` ∈ `openrouter | gemini | groq`.
- `from_lang` also selects the **OCR engine**: `from_lang: "japanese"` switches OCR to **manga-ocr** (handles vertical text); other languages use **EasyOCR**. The mapping lives in [`manga_translator/assets/languages.json`](manga_translator/assets/languages.json) (`engine` field) and engines are cached per language, so the first Japanese request downloads the manga-ocr model (~400MB) and subsequent ones reuse it.
- `font` picks the lettering font (a `key` from `GET /fonts/`). Omit it to use the per-language default. A font is only usable for languages it has glyphs for (a Latin-only comic font can't draw Thai/CJK) — see **Fonts** below.
- `text_scale` makes the text bigger/smaller (default `1.2`, range `0.5–2.0`). The renderer auto-fits text to the bubble's inscribed rectangle, which is conservative for round bubbles and can leave the text small; `text_scale` grows the fit area toward the bubble's bounding box (capped to it) so text fills more of the bubble. Text is also supersampled (drawn at 2× then downscaled) for crisper edges.
- `upscale` enlarges the **returned page** by this factor (`1` = off, max `4`); `upscaler` picks how:
  - `lanczos` (default) — fast, no extra deps. Bigger page (our rendered text stays sharp); the source artwork is enlarged but not deblurred.
  - `realesrgan` — AI super-resolution (anime-tuned) that actually sharpens/deblurs blurry source art. Heavy and slow on CPU; needs `realesrgan`+`basicsr` (not installed by default — see `requirement.txt`). If unavailable it **falls back to LANCZOS**.
- `custom_instruction` is a free-text localization instruction appended to the translator's guidelines for that request (e.g. `"keep -san/-chan honorifics"`, `"use casual slang"`, `"translate SFX literally"`). It's marked highest-priority in the system prompt; omit for the default guidelines.

#### Fonts

`GET /fonts/` lists the available fonts: `{ "fonts": [{ "key", "label", "scripts" }] }`. Pass a `key` as `font` in a translate request. Several manga/comic fonts (OFL/Apache, from Google Fonts) ship in [`manga_translator/assets/fonts/`](manga_translator/assets/fonts/) and are described in [`manga_translator/assets/fonts.json`](manga_translator/assets/fonts.json):

| Script | Fonts |
|--------|-------|
| Thai (+Latin) | TH Sarabun (default), Itim, Mali, Sriracha, Charm |
| Latin (comic) | Bangers, Comic Neue, Permanent Marker, Patrick Hand |
| Japanese | Yusei Magic |
| Chinese | Zhi Mang Xing |
| Korean | Nanum Pen |

When `font` is omitted, the renderer uses the default for the target language (`defaults` in `fonts.json`) so CJK/Korean text gets a font with the right glyphs instead of tofu boxes. Add a font by dropping a `.ttf` into the folder and adding an entry to `fonts.json`.

> **Security:** `api_key` travels in the request body — use HTTPS in production and avoid logging it. The userscript stores the key in Tampermonkey's `GM` storage on the user's machine.

Response:

```json
{ "images": ["data:image/jpeg;base64,...", null, "..."] }
```

> The server fetches each URL directly, so the URLs must be reachable from the server. If the source CDN blocks hotlinking, have the client send already-fetched image bytes instead (future enhancement).

#### Streaming variant

`POST /translate/stream/` takes the **same body** but streams results as **NDJSON** (`application/x-ndjson`) — one line per image, emitted the moment that image finishes (completion order, not request order):

```
{"index": 2, "image": "data:image/jpeg;base64,..."}
{"index": 0, "image": null}
{"index": 1, "image": "data:image/jpeg;base64,..."}
```

- `index` is the image's position in the request `images` list, so the client can swap each page in place as soon as its line arrives instead of waiting for the whole batch.
- `image` is `null` for any page that failed.
- Use this for progressive UIs (the tempomunkey userscript uses it). The non-streaming `/translate/` is kept for simple one-shot callers.

---

## Browser userscript (tempomunkey)

[`tempomunkey.user.js`](tempomunkey.user.js) is a Tampermonkey userscript that lets you pick which manga page images to translate (or select all) right on the reader page, toggle visual context, and swap the translations in place.

**Setup**
1. Install [Tampermonkey](https://www.tampermonkey.net/), then add `tempomunkey.user.js`.
2. Edit the script header / config: set `@match` to the reader's domain and `API_BASE` to your running server (default `http://localhost:8000`).
3. Start the API (`docker compose up` or uvicorn) with a provider key in `.env`.

**Use** — open a chapter; a control bar appears at the bottom-right and a checkbox is overlaid on every large image:

| Control | What it does |
|---------|--------------|
| checkbox on an image | mark that page for translation |
| **Select all** / **Clear** | tick / untick every detected page |
| **Visual context** | also send the page to the LLM as an image for better translation (slower) |
| **Translate selected** | translate the ticked pages; each page is swapped in place as soon as it finishes (streamed), showing ✓ (done) or ✕ (failed) |
| **Interrupt** | cancel the in-flight translation request |
| **Revert all** | restore the original (untranslated) images |
| **Rescan images** | re-detect images (use after lazy-loaded pages appear) |
| **Settings** | per-request `provider` / `model` / **source language** / target language / **font** / **text size** / **upscale** (factor + fast/AI) / **custom instruction** / `api_key` (persisted on your machine; sent with each translate request, blank = server default). Set source language to `japanese` to use manga-ocr; raise **text size** (e.g. 1.4) if translations are too small; set **upscale** (e.g. 2) to enlarge low-res pages. The font list is fetched from the server's `GET /fonts/`. |

> Cross-origin/mixed-content (https page → http localhost) is handled via `GM_xmlhttpRequest`. Pages are streamed back via `/translate/stream/` and swapped in place one by one (the script parses the NDJSON in `onprogress`). **Interrupt** stops the client from waiting and frees the UI, but work the server already started keeps running in the background. If the source CDN blocks server-side hotlinking, translation returns `null` (✕) for that page (future: have the script upload image bytes instead of URLs).

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

## Scraping an evaluation set

[`scrape_images.py`](scrape_images.py) collects page images into a folder so you can build a test set for the translator. It drives a real (headless) browser via **Playwright**, scrolls to trigger lazy-loading, grabs `<img>`s ≥ `--min-width` in reading order, and downloads them through the browser session (cookies + referer) to get past hotlink protection. SVG/non-raster images are skipped.

```powershell
# one-time setup
..\MangaTVL_ENV\python.exe -m pip install playwright
..\MangaTVL_ENV\python.exe -m playwright install chromium

# scrape one or more pages -> eval/scraped/<page-slug>/001.jpg, 002.jpg, ...
..\MangaTVL_ENV\python.exe scrape_images.py "https://reader.example/chapter/1"
..\MangaTVL_ENV\python.exe scrape_images.py --urls-file urls.txt --min-width 500 --limit 20
```

It force-loads lazy images (promotes `data-src` → `src`, scrolls to the bottom) so chapters that defer off-screen pages still get fully captured.

Options: `--out` (default `eval/scraped`), `--min-width` (default 400), `--selector` (CSS, e.g. `img.comic-image`, for precision when a page has many non-content images), `--limit` per page, `--scroll-rounds`, `--delay`, `--headful` (show the window, e.g. to pass a captcha). Output under `eval/scraped/` is gitignored.

---

## Project layout

```
MangaTVL/
├── main.py                     # FastAPI app
├── pipeline.py                 # Pipeline orchestrator (the 5 stages)
├── manga_translator/
│   ├── detection/              # ONNXDetection, YoloDetection (DetectorBase)
│   ├── ocr/                    # PaddleOCREngine, EasyOCREngine, clean_poly
│   ├── translators/            # OpenRouter / Gemini / Groq (AsyncTranslatorBase)
│   ├── inpainting/             # OpenCV / Simple-LaMa (InpainterBase)
│   ├── rendering/              # TextRenderer
│   ├── schemas/interface.py    # DetectionResult / OCRResult / TranslationResult
│   ├── utils/common.py         # image + geometry helpers
│   └── assets/                 # fonts, models, sample images
├── tests/                      # pytest suite (pure-logic unit tests)
└── report.md                   # design / risk reference (Thai)
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
- `report.md` is maintained in Thai.
