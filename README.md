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
  │              PaddleOCR / EasyOCR (per-bubble crop + polygon clean)
  ▼
[3] Translation  send the whole page's text to an LLM at once  → TranslationResult[]
  │              OpenRouter / Gemini / Groq (async, JSON output)
  ▼
[4] Inpainting ─ erase the original text from the image        → cleaned image
  │              OpenCV (telea) / Simple-LaMa; polygon-aware safe-fill
  ▼
[5] Rendering ── draw the translated Thai text into the bubble → output image
                 Pillow + Thai font, auto font-size, pythainlp line-breaking
```

The stages communicate through Pydantic models in [`manga_translator/schemas/interface.py`](manga_translator/schemas/interface.py):
`DetectionResult` → `OCRResult` → `TranslationResult`. The `DetectionResult` (with its bubble `bbox` and segmentation `polygon`) is threaded through to the end, so the inpaint and render stages can use the real bubble shape.

| Stage | Tech | Main file |
|-------|------|-----------|
| Detection | ONNX Runtime / ultralytics YOLO | [`detection/`](manga_translator/detection/) |
| OCR | PaddleOCR / EasyOCR | [`ocr/`](manga_translator/ocr/) |
| Translation | OpenRouter / Gemini / Groq (async) | [`translators/`](manga_translator/translators/) |
| Inpainting | OpenCV / Simple-LaMa | [`inpainting/`](manga_translator/inpainting/) |
| Rendering | Pillow + pythainlp | [`rendering/renderer.py`](manga_translator/rendering/renderer.py) |
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

Copy `.env.example` → `.env` and set **one** provider key plus `CONCURRENT_REQUESTS`:

```ini
OPENROUTER_API_KEY=...      # or GROQ_API_KEY / GOOGLE_API_KEY
CONCURRENT_REQUESTS=4       # max concurrent LLM calls
```

> Note: the Gemini provider reads `GOOGLE_API_KEY` (not `GEMINI_API_KEY`).

---

## Running the API

```powershell
..\MangaTVL_ENV\python.exe -m uvicorn main:app --reload
```

`main.py` configures the pipeline (provider, model, device) at the top — edit it to change providers or pass custom detector/OCR/inpainter/renderer instances.

### Endpoint

`POST /translate/`

```json
{ "url": "https://example.com/manga-page", "process_image": false }
```

- Scrapes every `<img>` from the given URL, runs the full pipeline on each, and returns base64-encoded translated images.
- `process_image: true` also sends the (downscaled) page to the LLM as **visual context** for better translation. Images are sent as proper multimodal parts, so the token cost stays low.

Response:

```json
{ "images": ["data:image/jpeg;base64,...", "..."] }
```

---

## Using the pipeline directly

```python
import asyncio
from pipeline import Pipeline
from manga_translator.detection.yolo_detection import YoloDetection

pipe = Pipeline(
    det_model=YoloDetection(task="segment"),   # polygon-aware detection
    provider="openrouter",
    model="google/gemini-2.5-flash",
    device="cuda",
)

result = asyncio.run(pipe.run("path/or/url/to/page.jpg", process_image=True))
result.save("translated.jpg")
```

`Pipeline.run` handles a single image; `Pipeline.run_batch` processes many concurrently (API concurrency is throttled by `CONCURRENT_REQUESTS`).

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
├── CHANGELOG.md                # change log (Thai)
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
- `CHANGELOG.md` and `report.md` are maintained in Thai.
