# MangaTVL — รายงานการทำงานของ Pipeline และการวิเคราะห์ความเสี่ยง (Version 2)

> เอกสารฉบับที่ 2 อัปเดตหลังการ migration detection ไปเป็น **YOLO segmentation (ultralytics)** และการ refactor/แก้บั๊กครั้งใหญ่ทั้งระบบ (ดูประวัติใน [CHANGELOG.md](CHANGELOG.md))
>
> โครงสร้างเอกสารคล้ายฉบับแรก แต่เพิ่มส่วน "ปัญหาที่แก้แล้ว" กับ "ปัญหาที่ยังไม่แก้" แยกกันชัดเจน
>
> อัปเดต: 2026-06-08

## สารบัญ

1. [ภาพรวมการทำงานของ Pipeline (สรุป)](#1-ภาพรวมการทำงานของ-pipeline-สรุป)
2. [การทำงานของ Pipeline อย่างละเอียด](#2-การทำงานของ-pipeline-อย่างละเอียด)
3. [ปัญหา ความเสี่ยง และแนวทางแก้ไข](#3-ปัญหา-ความเสี่ยง-และแนวทางแก้ไข)

---

## 1. ภาพรวมการทำงานของ Pipeline (สรุป)

ระบบรับภาพหน้ามังงะ/คอมมิค แล้วแปลข้อความใน bubble คำพูดจากอังกฤษเป็นไทย (ปรับปลายทางได้) โดยวาดข้อความที่แปลกลับลงบนภาพเดิม ขั้นตอนเป็นท่อ (pipeline) 5 สเตจ:

```
ภาพเข้า
  │
  ▼
[1] Detection ─── ตรวจจับ bubble + polygon (segmentation)        → DetectionResult[]
  │                YOLO-seg (ultralytics) ผ่าน YoloDetection (default)
  ▼
[2] OCR ─────────  crop ต่อ bubble → ทาขาวนอก polygon → อ่าน    → OCRResult[]
  │                EasyOCR (default) / PaddleOCR (clean_poly ก่อน OCR)
  ▼
[3] Translation ── ส่งข้อความทั้งหน้าให้ LLM แปลทีเดียว           → TranslationResult[]
  │                OpenRouter / Gemini / Groq (async, JSON, map ด้วย text_no)
  ▼
[4] Inpainting ─── ลบข้อความเดิม (polygon safe-fill + box fallback) → ภาพที่ลบ text แล้ว
  │                OpenCV telea / Simple-LaMa
  ▼
[5] Rendering ──── วาดข้อความแปลในพื้นที่ bubble (polygon LIR)     → ภาพผลลัพธ์
                   Pillow + ฟอนต์ไทย, auto font-size, ตัดคำด้วย pythainlp
```

**จุดเชื่อมต่อหลัก:** [pipeline.py](pipeline.py) ที่ **project root** (ย้ายมาจาก `manga_translator/pipeline/` แล้ว) คือ `Pipeline` ที่ร้อยทุกสเตจ (`run` ภาพเดียว, `run_batch` หลายภาพ) และ [main.py](main.py) เปิดเป็น FastAPI service: รับ URL → scrape `<img>` → แปลทั้งหมด → คืน base64

**สแต็กเทคโนโลยี:**

| สเตจ | เทคโนโลยี | ไฟล์หลัก |
|------|-----------|----------|
| Detection | ultralytics YOLO-seg (default) | [detection/yolo_detection.py](manga_translator/detection/yolo_detection.py) — ONNX เป็น legacy เลิกใช้แล้ว |
| OCR | EasyOCR (default) / PaddleOCR | [ocr/easyocr_engine.py](manga_translator/ocr/easyocr_engine.py), [ocr/paddleocr_engine.py](manga_translator/ocr/paddleocr_engine.py) |
| Translation | OpenRouter / Gemini / Groq (async) | [translators/](manga_translator/translators/) |
| Inpainting | OpenCV / Simple-LaMa | [inpainting/](manga_translator/inpainting/) |
| Rendering | Pillow + pythainlp | [rendering/renderer.py](manga_translator/rendering/renderer.py) |
| Schema | Pydantic | [schemas/interface.py](manga_translator/schemas/interface.py) |
| Serving | FastAPI | [main.py](main.py) |
| Orchestrator | — | [pipeline.py](pipeline.py) (root) |

**การเปลี่ยนแปลงสำคัญจาก v1:** detection ย้ายจาก ONNX box-only → **YOLO-seg ที่ให้ polygon** ของ bubble; polygon ถูกพาผ่านทุกสเตจและถูกใช้จริงใน OCR (clean), inpaint (safe-fill), render (LIR) ซึ่งเป็นหัวใจของการแก้ปัญหา OCR overlap ข้ามฟองที่เป็นโจทย์ตั้งต้นใน v1

---

## 2. การทำงานของ Pipeline อย่างละเอียด

### 2.0 โครงสร้างข้อมูลที่ไหลผ่าน pipeline

[schemas/interface.py](manga_translator/schemas/interface.py) — เพิ่มฟิลด์ `segmentation` ใน `DetectionResult` (จุดต่างหลักจาก v1):

```python
class DetectionResult(BaseModel):
    bbox: list[int]                          # [x1, y1, x2, y2]
    segmentation: list[list[int]] | None = None   # polygon [[x,y], ...] จาก YOLO-seg
    form_type: str                           # "bubble" / "free_text"

class OCRResult(BaseModel):
    text: str
    boxes: list[list[int]]
    detection_result: DetectionResult        # พา bbox + polygon ไปด้วย

class TranslationResult(BaseModel):
    ocr_result: OCRResult
    translated_text: str
```

`DetectionResult` (พร้อม `segmentation`) ถูกฝังต่อเนื่องไปจนถึงสเตจสุดท้าย → render/inpaint เข้าถึง polygon ของ bubble จริงได้

### 2.1 Pipeline orchestrator

[pipeline.py](pipeline.py) (root) — หลัง refactor: รวม logic ของ `run`/`run_batch` ไว้ใน `_run_on_image` ตัวเดียว, ยุบ `_init_translator` 3 branch เป็น kwargs dict เดียว; **default detector = `YoloDetection`, default OCR = `EasyOCREngine`**

```python
async def _run_on_image(self, image, process_image=False):
    detection_result = self.det_model.detect(image)
    if not detection_result:
        return image
    ocr_result = self.ocr_engine.get_ocr(image, detection_result)
    context_image = self.resize_image(image, max_size=self.resize_max) if process_image else None
    translation_result = await self.translator.translate(ocr_result=ocr_result, image=context_image)
    inpainted_image = self.inpainter.inpaint(image, translation_result)
    return self.renderer.render(inpainted_image, translation_result)
```

### 2.2 สเตจ 1 — Detection

**`DetectorBase`** ([detection/detector.py](manga_translator/detection/detector.py)) — กำหนด contract `.detect(image) -> list[DetectionResult]` ร่วมของทุก backend + helper `build_result`/`build_results` ที่รับ output แบบ ultralytics (xyxy + class ids + polygons จาก `result.masks.xy`) แล้วแปลงเป็น schema

**`YoloDetection`** ([detection/yolo_detection.py](manga_translator/detection/yolo_detection.py)) — backend หลัก/default ปัจจุบัน:

```python
result = self.model(image, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
boxes = result.boxes.xyxy.cpu().numpy()
cls_ids = result.boxes.cls.cpu().numpy()
polygons = result.masks.xy if result.masks is not None else None
return self.build_results(boxes, cls_ids, polygons)
```

- ใช้ `ultralytics.YOLO` รองรับทั้ง `.pt` และ `.onnx` ในตัวเดียว, ทำงานได้ทั้ง task `detect` และ `segment`
- ถ้าเป็นโมเดล seg → `result.masks` มีค่า → เก็บ polygon ลง `segmentation`; ถ้า detect ล้วน → `segmentation = None`
- default `model_path` = `assets/models/best_diplom.pt` (resolve แบบ package-relative)

**`ONNXDetection`** — backend เดิม (box-only) **เลิกใช้แล้ว** เก็บไว้เป็น legacy option เท่านั้น ไม่ใช่ default ของ pipeline อีกต่อไป

### 2.2.1 Model & Dataset (ใหม่)

- **Base model:** `yolo26n-seg.pt` (ต้องเป็นรุ่น **-seg** ไม่ใช่ detect-only — ไม่งั้นโมเดลจะไม่มี seg head และ `result.masks` จะเป็น None ทำให้ไม่ได้ polygon)
- **Dataset:** [Roboflow — diplom-uhct7/manga-6puie v4](https://universe.roboflow.com/diplom-uhct7/manga-6puie/dataset/4)
  - `nc: 1`, `names: ['bubble']` — **มี class เดียว** (ไม่มี free_text/SFX)
  - polygon labels (segmentation format)
  - train 1,304 / valid 189 / test 103 ภาพ
  - มี negative samples (label ว่าง ไม่มี bubble) ~15% — ช่วยลด false positive
- **ผลที่ได้:** detect bubble แม่น แต่ **ไม่รู้จัก SFX/free_text** (ดูปัญหาข้อ 3.2)

### 2.3 สเตจ 2 — OCR

ยุทธวิธีปัจจุบัน: **crop ต่อ bubble → ทาขาวนอก polygon (clean_poly) → OCR** — แก้ปัญหา OCR overlap ข้ามฟองที่เป็นโจทย์ตั้งต้นใน v1

`clean_poly` ([ocr/ocr_engine.py](manga_translator/ocr/ocr_engine.py)) บน base `OCREngine`:

```python
def clean_poly(self, cropped_image, poly, offset=(0,0), fill=(255,255,255)):
    if poly is None or len(poly) == 0:
        return cropped_image                 # detect-only → คืนเดิม
    pts = np.array(poly) - np.array(offset)  # shift polygon เข้ากรอบ crop
    mask = np.zeros(...); cv2.fillPoly(mask, [pts], 255)
    arr[mask == 0] = fill                    # นอก polygon = ขาว
    return Image.fromarray(arr)
```

**`EasyOCREngine` คือ default/ตัวหลักปัจจุบัน** ([ocr/easyocr_engine.py](manga_translator/ocr/easyocr_engine.py)); `PaddleOCREngine` เป็นทางเลือก ทั้งคู่ใช้ pipeline เดียวกัน (crop ต่อ bubble → `clean_poly` → OCR → map box กลับ):

```python
for det in detection_results:
    x1, y1, x2, y2 = det.bbox
    cropped = image.crop((x1, y1, x2, y2))
    cropped = self.clean_poly(cropped, det.segmentation, offset=(x1, y1))  # ลบ noise นอกฟอง
    text, boxes = self.ocr(cropped)
    mapped_boxes = [[xmin+x1, ymin+y1, xmax+x1, ymax+y1] for ... in boxes]
    ocr_results.append(OCRResult(text=text, boxes=mapped_boxes, detection_result=det))
```

- `EasyOCREngine` เขียนใหม่ให้ใช้ per-bubble crop pipeline แบบเดียวกับ PaddleOCR (เดิมเคย OCR ทั้งภาพ + matcher)
- ถ้าสลับไปใช้ `PaddleOCREngine` ต้องส่ง `enable_mkldnn=False` (กัน Paddle 3.3 PIR/oneDNN crash)

### 2.4 สเตจ 3 — Translation

[translators/translator.py](manga_translator/translators/translator.py) — หลัง optimize:

- `AsyncTranslatorBase` — base + `_call_with_retry` (rate-limit + backoff + semaphore ที่เดียว) + `translate()` ที่ robust
- `OpenAICompatibleTranslator` — implement `_translate` แบบ multimodal สำหรับ client ที่มี `chat.completions.create`; `AsyncOpenRouterTranslator` / `AsyncGroqTranslator` เป็น subclass ว่าง (dedup)
- `AsyncGeminiTranslator` — ส่งรูปเป็น `Part.from_bytes`

**Map ผลแปลด้วย `text_no` (ไม่ใช่ positional zip):**

```python
by_no = {int(r["text_no"]): r["translated_text"] for r in parse_response(response) if ...}
for i, ocr_item in enumerate(ocr_result):
    translated = by_no.get(i, ocr_item.text)   # ไม่มี → fallback ข้อความเดิม
    results.append(TranslationResult(ocr_result=ocr_item, translated_text=translated))
```

**รูปส่งเป็น image part จริง** (ไม่ใช่ base64 ใน text) — `{"type":"image_url","image_url":{"url": data_uri, "detail":"low"}}`

### 2.5 สเตจ 4 — Inpainting

[inpainting/inpainter.py](manga_translator/inpainting/inpainter.py) — **border-uniformity safe-fill** (adapt จาก PanelCleaner):

```python
def inpaint(self, image, inputs):
    inputs = self.parse_inputs(inputs)
    image, demoted = self._safe_fill_polygons(image, inputs)  # bubble ที่ polygon ปลอดภัย → ทาขาว
    if not demoted:
        return image
    mask = self._masking(image, demoted)                       # ที่เหลือ → box-based
    return self._inpaint(image, mask, inputs=demoted)
```

`_pick_safe_polygon_mask` — erode polygon เข้าทีละ step (adaptive ตาม diagonal), วัด std ของ pixel ที่ขอบ mask, เลือกตัวที่ std ต่ำสุด (= ขอบอยู่ในของ bubble interior ขาว ไม่ทาบเส้นขอบดำ); ถ้า std ยังเกิน `max_std=25` → คืน None (skip, ตกไป box-based) — แก้ปัญหา YOLO-seg polygon overshoot นอกขอบ bubble

### 2.6 สเตจ 5 — Rendering

[rendering/renderer.py](manga_translator/rendering/renderer.py) — ใช้ **Largest Inscribed Rectangle (LIR)** ของ polygon เป็นพื้นที่วาด:

- `_compute_box(det, image_shape, pad)` priority: polygon LIR → `det.bbox` inset → OCR-union fallback
- `inscribed_rect` ([utils/common.py](manga_translator/utils/common.py)) — rasterize polygon → erode → หา LIR ด้วย largest-rectangle-in-histogram O(H×W)
- stroke scale ตาม font (`max(1, fs//12)`) + นับ stroke เข้า fit check
- fallback ไม่ overflow: shrink ต่ำกว่า min_font → truncate + "…"

---

## 3. ปัญหา ความเสี่ยง และแนวทางแก้ไข

### 3.1 ปัญหาที่แก้แล้ว (ใน session ปัจจุบัน)

| ปัญหา | วิธีแก้ | อ้างอิง |
|-------|--------|---------|
| **OCR overlap ข้ามฟอง** (crop สี่เหลี่ยมกินฟองข้างเคียง) | `clean_poly` ทาขาวนอก polygon ก่อน OCR | [2026-06-07 14:45](CHANGELOG.md) |
| **Token ระเบิด ~1M/รูป** เมื่อส่งรูปให้ VLM | รูปถูกส่งเป็น base64 text → เปลี่ยนเป็น image_url multimodal part (ประหยัด ~50-400x); วัดจริงด้วย tiktoken | [2026-06-07 19:57](CHANGELOG.md), [2026-06-07 20:27](CHANGELOG.md) |
| **คำแปลลงผิด bubble** (positional zip) | map ด้วย `text_no` + fallback ข้อความเดิม + ไม่กลืน exception | [2026-06-07 16:45](CHANGELOG.md), [2026-06-07 20:27](CHANGELOG.md) |
| **ข้อความล้นนอกเส้น bubble** | render area = polygon LIR + stroke scaling + truncate fallback | [2026-06-07 15:53](CHANGELOG.md) |
| **Inpaint smear/กินขอบ bubble** (polygon overshoot) | border-uniformity safe-fill (adaptive erode + std threshold) | [2026-06-07 18:40](CHANGELOG.md), [2026-06-07 19:21](CHANGELOG.md) |
| **font path Linux hardcode** (พังบน Windows) | resolve เป็น package-relative / ให้ renderer ใช้ default เอง | [2026-06-07 20:17](CHANGELOG.md) |
| **PaddleOCR crash** (oneDNN/PIR บน Paddle 3.3) | `enable_mkldnn=False` | [2026-06-07 14:54](CHANGELOG.md) |
| **Translator ซ้ำ 3 ไฟล์** | ยุบ OpenRouter/Groq เป็น OpenAI-compatible base + shared retry | [2026-06-07 20:27](CHANGELOG.md) |
| **pipeline.py อ่านยาก + อยู่ลึก** | ย้ายไป root + refactor (รวม run/run_batch, ตัด dead code) | [2026-06-07 20:17](CHANGELOG.md) |
| **ไม่มี test** | เพิ่ม pytest suite 45 tests (pure logic) | [2026-06-07 20:40](CHANGELOG.md) |

### 3.2 ปัญหาที่ยังไม่แก้ / ความเสี่ยงคงค้าง

**🔴 Tier 1 — สำคัญ**
- **SFX / ตัวอักษรวาดมือ OCR ไม่ออก** — printed text อ่านได้ดีแล้ว แต่ SFX (เช่น "NOOOO", "HAA HAA") ออกมาเป็น gibberish ทุก OCR engine (EasyOCR/PaddleOCR เทรนจาก font พิมพ์) → translator เดาสุ่มเป็น SFX ไทยมั่ว
  - *แนวทาง:* filter ด้วย OCR confidence → skip หรือส่ง crop ให้ vision LLM อ่าน

**🟠 Tier 2 — คุณภาพ/ความถูกต้อง**
- **Dataset มี class เดียว (`nc:1` bubble)** — ไม่มี free_text/SFX → SFX ไม่ถูก detect แยก
  - *แนวทาง:* label เพิ่ม + retrain `nc:2`

**🟡 Tier 3 — สถาปัตยกรรม/สเกล**
- **"async" ปลอม** — detect/ocr/inpaint/render เป็น sync แต่เรียกใน coroutine → บล็อก event loop (CV ของหน้าอื่นไม่ overlap จริง)
  - *แนวทาง:* `run_in_executor` / process pool
- **`run_batch` gather ไม่จำกัด** — หลายภาพพร้อมกัน เสี่ยง OOM
- **`concurrent_limit` default = 1** — ถ้าไม่ตั้ง `CONCURRENT_REQUESTS` → API call serialize ทั้งหมด (run_batch ดูขนานแต่จริงไม่ขนาน)
- **SSRF ใน [main.py](main.py)** — โหลด URL ใดก็ได้ที่ผู้ใช้ส่ง → allowlist domain + จำกัดขนาด/จำนวน
- **ไม่มี cache** — รูปเดิมแปลซ้ำเสียค่า API ทุกครั้ง

**🟢 เล็กน้อย**
- `OCRResult.__post_init__` ([interface.py](manga_translator/schemas/interface.py)) เป็น dead code — pydantic `BaseModel` ไม่เรียก `__post_init__` (เป็นของ dataclass)
- `use_angle_cls=True` ใน PaddleOCR 3.x เปลี่ยนชื่อเป็น `use_textline_orientation` แล้ว — ค่าปัจจุบันถูก `**kwargs` กลืน ไม่มีผล

### 3.3 ลำดับแนะนำถัดไป

1. จัดการ SFX: confidence filter → skip/route ไป vision LLM
2. ตั้ง `CONCURRENT_REQUESTS` + ใส่ image-level semaphore ใน run_batch
3. (ระยะยาว) retrain dataset `nc:2` เพิ่ม free_text, ห่อ sync stages ด้วย executor, เพิ่ม cache
