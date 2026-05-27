# MangaTVL — รายงานการทำงานของ Pipeline และการวิเคราะห์ความเสี่ยง

> เอกสารฉบับนี้อธิบายการทำงานของระบบแปลมังงะ/คอมมิคอัตโนมัติในโปรเจกต์นี้ ตั้งแต่ภาพรวม ลงลึกถึงระดับ source code และตัวอย่างการทำงาน พร้อมวิเคราะห์ปัญหา/ความเสี่ยงที่พบในปัจจุบันและแนวทางแก้ไข
>
> อัปเดต: 2026-05-28

## สารบัญ

1. [ภาพรวมการทำงานของ Pipeline (สรุป)](#1-ภาพรวมการทำงานของ-pipeline-สรุป)
2. [การทำงานของ Pipeline อย่างละเอียด](#2-การทำงานของ-pipeline-อย่างละเอียด)
3. [ปัญหา ความเสี่ยง และแนวทางแก้ไข](#3-ปัญหา-ความเสี่ยง-และแนวทางแก้ไข)

---

## 1. ภาพรวมการทำงานของ Pipeline (สรุป)

ระบบรับภาพหน้ามังงะ/คอมมิค แล้วแปลข้อความใน bubble คำพูดเป็นภาษาปลายทาง (ค่า default คือไทย) โดยวาดข้อความที่แปลกลับลงบนภาพเดิม ขั้นตอนหลักเป็นท่อ (pipeline) 5 สเตจต่อเนื่องกัน:

```
ภาพเข้า
  │
  ▼
[1] Detection ─── ตรวจจับกล่อง (bbox) ของ bubble คำพูด/ข้อความ  → DetectionResult[]
  │                (ONNX YOLO-style model, class: bubble / free_text)
  ▼
[2] OCR ─────────  อ่านข้อความในแต่ละกล่อง                   → OCRResult[]
  │                (EasyOCR ทั้งภาพ + matcher  หรือ  PaddleOCR แบบ crop ต่อกล่อง)
  ▼
[3] Translation ── ส่งข้อความทั้งหน้าให้ LLM แปลทีเดียว        → TranslationResult[]
  │                (Gemini / OpenRouter / Groq, ตอบกลับเป็น JSON)
  ▼
[4] Inpainting ─── ลบข้อความเดิมออกจากภาพ (mask จากกล่อง OCR) → ภาพที่ลบ text แล้ว
  │                (OpenCV telea  หรือ  Simple-LaMa)
  ▼
[5] Rendering ──── วาดข้อความที่แปลแล้วลงในพื้นที่ bubble          → ภาพผลลัพธ์
                   (PIL + ฟอนต์ไทย, auto font-size, ตัดคำด้วย pythainlp)
```

**จุดเชื่อมต่อหลัก:** ไฟล์ [manga_translator/pipeline/pipeline.py](manga_translator/pipeline/pipeline.py) เป็นตัวร้อยทุกสเตจเข้าด้วยกัน (`run` สำหรับภาพเดียว, `run_batch` สำหรับหลายภาพ) และ [main.py](main.py) เปิดเป็น FastAPI service ที่รับ URL → scrape `<img>` → แปลทั้งหมด → คืน base64

**สแต็กเทคโนโลยี:**

| สเตจ | เทคโนโลยี | ไฟล์หลัก |
|------|-----------|----------|
| Detection | ONNX Runtime (YOLO-style) | [detection/onnx_detection.py](manga_translator/detection/onnx_detection.py) |
| OCR | EasyOCR / PaddleOCR | [ocr/easyocr_engine.py](manga_translator/ocr/easyocr_engine.py), [ocr/paddleocr_engine.py](manga_translator/ocr/paddleocr_engine.py) |
| Translation | Gemini / OpenRouter / Groq (async) | [translators/](manga_translator/translators/) |
| Inpainting | OpenCV / Simple-LaMa | [inpainting/](manga_translator/inpainting/) |
| Rendering | Pillow + pythainlp | [rendering/renderer.py](manga_translator/rendering/renderer.py) |
| Schema | Pydantic | [schemas/interface.py](manga_translator/schemas/interface.py) |
| Serving | FastAPI | [main.py](main.py) |

---

## 2. การทำงานของ Pipeline อย่างละเอียด

### 2.0 โครงสร้างข้อมูลที่ไหลผ่าน pipeline

ทุกสเตจสื่อสารกันผ่าน schema 3 ตัวใน [schemas/interface.py](manga_translator/schemas/interface.py):

```python
class DetectionResult(BaseModel):
    bbox: list[int]       # [x1, y1, x2, y2]
    form_type: str        # "bubble" หรือ "free_text"

class OCRResult(BaseModel):
    text: str                       # ข้อความที่อ่านได้ (รวมแล้ว)
    boxes: list[list[int]]          # กล่องของแต่ละกลุ่มข้อความจาก OCR
    detection_result: DetectionResult   #  bubble ที่ข้อความนี้สังกัด

class TranslationResult(BaseModel):
    ocr_result: OCRResult
    translated_text: str
```

`DetectionResult` ถูกฝังต่อเนื่องไปจนถึงสเตจสุดท้าย (ผ่าน `OCRResult.detection_result` → `TranslationResult.ocr_result.detection_result`) ทำให้สเตจ render/inpaint ยังเข้าถึงกล่อง bubble ต้นทางได้

### 2.1 Entry point และตัวร้อย Pipeline

[main.py](main.py) สร้าง `Pipeline` หนึ่งตัวเป็น global แล้วเปิด endpoint `POST /translate/`:

```python
pipe = Pipeline(provider="openrouter", model="google/gemini-2.5-flash",
                device="cuda", user_prompt=user_prompt)

@app.post("/translate/")
async def translate_image(request: TranslateRequest):
    res = requests.get(request.url)
    soup = BeautifulSoup(res.text, "html.parser")
    imgs = [img.get("src").split("?")[0] for img in soup.find_all("img") ...]
    results = await pipe.run_batch(image_paths=imgs, process_image=request.process_image)
    return {"images": [convert_img_to_base64(img) for img in results if img]}
```

ตัว `Pipeline.run` ([pipeline.py:97](manga_translator/pipeline/pipeline.py#L97)) เรียงสเตจตรงไปตรงมา:

```python
detection_result = self.det_model.detect(image)          # [1]
ocr_result       = self.ocr_engine.get_ocr(image, detection_result)  # [2]
translation_result = await self._translate(ocr_result, image_for_translate)  # [3]
inpainted_image  = self.inpainter.inpaint(image, translation_result) # [4]
rendered_image   = self.renderer.render(inpainted_image, translation_result) # [5]
```

ค่า default ที่ประกอบใน `__init__` ([pipeline.py:29-58](manga_translator/pipeline/pipeline.py#L29-L58)): detector = `ONNXDetection`, OCR = `EasyOCREngine(language="en")`, inpainter = `OpenCVInpainter`, renderer = `TextRenderer`

### 2.2 สเตจ 1 — Detection

[detection/onnx_detection.py](manga_translator/detection/onnx_detection.py) โหลดโมเดล ONNX แบบ YOLO (2 คลาส: `0=bubble`, `1=free_text`) แล้วทำ inference + decode เอง

ขั้นตอนภายใน `detect`:
1. **preprocess** — resize เป็นขนาด input ของโมเดล, HWC→CHW, normalize /255
2. **run** ONNX session
3. **postprocess** — กรองด้วย confidence (`conf_threshold=0.6`), แปลงจาก `(cx,cy,w,h)` → `(x1,y1,x2,y2)`, rescale กลับขนาดภาพเดิม
4. **NMS** เอง (`MNS`, `iou_threshold=0.5`)

```python
def detect(self, image):
    boxes, confidences = self.run_inference(image)
    results = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2, class_id = box
        form_type = self.cls_names.get(int(class_id), "unknown")
        results.append(DetectionResult(bbox=[int(x1), int(y1), int(x2), int(y2)],
                                        form_type=form_type))
    return results
```

ผลลัพธ์เป็น `list[DetectionResult]` — กล่องสี่เหลี่ยมรอบ bubble/ข้อความแต่ละจุด

### 2.3 สเตจ 2 — OCR

มี 2 engine ที่ใช้ยุทธวิธีต่างกัน:

**(ก) EasyOCREngine — OCR ทั้งภาพ แล้วค่อย match เข้า bubble** ([ocr/easyocr_engine.py](manga_translator/ocr/easyocr_engine.py))

```python
def get_ocr(self, image, detection_results):
    result = self.ocr_model.readtext(pil2cv(image), detail=1, paragraph=True)
    result = [(res[1], self.poly_to_xyxy(res[0])) for res in result]
    matched = match_ocr_to_bubbles(detection_results, result)
    return matched
```

จากนั้น [ocr/bubble_ocr_matcher.py](manga_translator/ocr/bubble_ocr_matcher.py) จับคู่กล่อง OCR แต่ละอันเข้ากับ bubble ที่ "เข้ากันมากสุด" โดยให้คะแนน = `max(IoU, overlap%, center_in_box)` แล้วรวมข้อความของ bubble เดียวกันเป็นก้อนเดียว:

```python
scores[i, j] = max(iou, overlap_pct_of_b, center_flag)
...
best_i = int(np.argmax(scores[:, j]))   # เลือก bubble ที่คะแนนสูงสุดให้ OCR box j
...
combined_text = " ".join(t for t in texts if t)  # รวมข้อความทุกอันใน bubble
```

**(ข) PaddleOCREngine — crop ต่อ bubble** ([ocr/paddleocr_engine.py](manga_translator/ocr/paddleocr_engine.py)) — ยุทธวิธีที่ใช้งานจริงในปัจจุบัน

```python
def get_ocr(self, image, detection_results):
    ocr_results = []
    for det in detection_results:
        x1, y1, x2, y2 = det.bbox
        cropped_img = image.crop((x1, y1, x2, y2))   # crop เป็นสี่เหลี่ยมรอบ bubble
        text, boxes = self.ocr(cropped_img)
        mapped_boxes = [[xmin+x1, ymin+y1, xmax+x1, ymax+y1] for (xmin,ymin,xmax,ymax) in boxes]
        ocr_results.append(OCRResult(text=text, boxes=mapped_boxes, detection_result=det))
    return ocr_results
```

ทั้งสองแบบให้ผลเป็น `list[OCRResult]` ที่ 1 รายการ = 1  bubble (พร้อมข้อความและกล่อง)

### 2.4 สเตจ 3 — Translation

[translators/translator.py](manga_translator/translators/translator.py) เป็น base แบบ async โดยรวมข้อความ**ทุก bubble ในหน้า**เข้าเป็น prompt เดียวแล้วยิงให้ LLM ครั้งเดียว:

```python
def _preprocess(self, inputs):
    user_prompt = ""
    for i, ocr_item in enumerate(inputs):
        user_prompt += str({"text_no": i, "text": ocr_item.text}) + "\n"
    return get_ocr_prompt(user_prompt)
```

System prompt ([translators/utils/prompt.py](manga_translator/translators/utils/prompt.py)) สั่งให้ LLM คืน JSON `[{"text_no", "translated_text"}, ...]` แล้ว parse กลับด้วย `parse_response` ([translators/utils/common.py](manga_translator/translators/utils/common.py)) ที่รองรับทั้ง JSON/YAML และ code fence จากนั้น map ผลกลับเข้า OCRResult:

```python
response_sorted = sorted(response_json, key=lambda x: x['text_no'])
for ocr_item, res_item in zip(ocr_result, response_sorted):
    translation_results.append(TranslationResult(ocr_result=ocr_item,
                                                  translated_text=res_item['translated_text']))
```

มี 3 provider ที่ logic เหมือนกันต่างแค่ SDK: [gemini.py](manga_translator/translators/gemini.py), [openrouter.py](manga_translator/translators/openrouter.py), [groq.py](manga_translator/translators/groq.py) ทุกตัวมี retry + จัดการ HTTP 429 ด้วย exponential backoff และ semaphore จำกัด concurrency

### 2.5 สเตจ 4 — Inpainting

[inpainting/inpainter.py](manga_translator/inpainting/inpainter.py) (base) สร้าง mask จาก**กล่องข้อความ (OCR boxes)** ของทุก bubble แล้วส่งให้ backend ลบ:

```python
def _masking(self, image, inputs, expand_margin=2):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for item in inputs:
        for box in item.boxes:
            x1, y1, x2, y2 = self.expand_box(box, image.shape, expand_margin)
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
    return mask
```

backend มี 2 ตัว: [opencv_inpainter.py](manga_translator/inpainting/opencv_inpainter.py) (`cv2.inpaint` telea — เบา/เร็ว) และ [simple_lama_inpainter.py](manga_translator/inpainting/simple_lama_inpainter.py) (Simple-LaMa — คุณภาพสูงกว่าแต่หนัก) ค่า default ของ pipeline ใช้ OpenCV

### 2.6 สเตจ 5 — Rendering

[rendering/renderer.py](manga_translator/rendering/renderer.py) วาดข้อความที่แปลแล้วลงบนภาพที่ลบ text เดิมไปแล้ว ขั้นตอนสำคัญ:

1. หาพื้นที่วาด = **union ของกล่อง OCR** ของ bubble นั้น ([rendering/extract_text_box.py](manga_translator/rendering/extract_text_box.py) → `combine_bbox`)
2. ตัดคำไทยด้วย pythainlp (`newmm`) แล้ว **binary search หา font size ที่ใหญ่สุดที่ยังพอดีกล่อง**
3. วาดข้อความจัดกึ่งกลาง พร้อม stroke ขาว

```python
def render(self, image, inputs):
    draw = ImageDraw.Draw(image_copy)
    for item in inputs:
        text = item.translated_text or item.ocr_result.text
        box = extract_text_box(image_copy, item)             # = union ของ OCR boxes
        det_box = item.ocr_result.detection_result.bbox      # กล่อง bubble (ส่งต่อแต่ไม่ได้ใช้)
        self._render_single(draw, text, box, det_box)
    return image_copy
```

```python
def wrap_extraction(self, draw, text, box, det_box):
    low, high = self.min_font_size, self.max_font_size
    while low <= high:                          # binary search font size
        mid = (low + high) // 2
        fits, wrap_text = self._fits_in_box(draw, mid, tokenized_text,
                                            target_width, target_height, det_box)
        if fits: best_font_size = mid; low = mid + 1
        else:    high = mid - 1
```

---

## 3. ปัญหา ความเสี่ยง และแนวทางแก้ไข

จัดลำดับตามความรุนแรง/ผลกระทบต่อความถูกต้องของผลลัพธ์

### 3.0 ปัญหาหลักที่เป็นโจทย์ตั้งต้น — OCR overlap ข้าม bubble

**อาการ:**  bubble คำพูดสองอันที่อยู่ติด/ซ้อนกัน ทำให้พื้นที่ render คำนวณผิดและการแปลผิด

![alt text](report_image.png "Overlap Bubbles")

**สาเหตุ:** การ crop  bubble ในสเตจ OCR ([paddleocr_engine.py:36](manga_translator/ocr/paddleocr_engine.py#L36)) ใช้กล่อง **สี่เหลี่ยม** ของ `det.bbox` แต่ bubble จริงเป็นวงรี → กล่องสี่เหลี่ยมของ bubble หนึ่งกินพื้นที่ของ bubble ข้างเคียง → OCR อ่าน text ของ bubble อื่นปนเข้ามา → ข้อความเพี้ยน + กล่องบวม + แปลผิด

**แนวทางแก้ (ตามที่ตกลงไว้):** เทรน detector ใหม่เป็น **YOLOv8-seg** เพื่อให้ได้ instance mask รูป bubble จริง แล้วนำ mask มา **ทาขาวพื้นที่นอก bubble** ในภาพ crop ก่อนส่ง OCR —  bubble ข้างเคียงจะอยู่นอก mask เสมอจึงถูกลบทิ้ง รายละเอียดและความเสี่ยงของแผนนี้อยู่ในหัวข้อ 3.4

### 3.1 🔴 Tier 1 — บั๊กที่ทำให้ผลลัพธ์ผิดแบบเงียบ ๆ (ควรแก้ก่อน)

| # | ปัญหา | ตำแหน่ง | ผลกระทบ | แนวทางแก้ |
|---|-------|---------|---------|-----------|
| 1 | **Map คำแปลด้วยลำดับ `zip` ไม่ใช่ `text_no`** | [translator.py:52-56](manga_translator/translators/translator.py#L52-L56) | ถ้า LLM ตอบขาด/เกิน/สลับ → คำแปลลงผิด bubble ทั้งหมดหลังจุดนั้น โดยไม่มี error | ทำ dict `{text_no: translated_text}` แล้ว lookup ตาม index ของ ocr_item; key หาย → fallback เป็น text เดิม + log |
| 2 | **Parse fail = ทั้งหน้าหายเงียบ** | [translators/utils/common.py:54](manga_translator/translators/utils/common.py#L54), [translator.py:64](manga_translator/translators/translator.py#L64) | `parse_response` คืน `[]` แล้ว `except Exception: return []` กลืน error ทุกชนิด → หน้าไม่ถูกแปลโดยไม่รู้สาเหตุ | retry เมื่อ parse fail, บังคับ structured output, log payload ที่พัง, ไม่กลืน exception เงียบ |
| 3 | **Hardcode path แบบ Linux บนเครื่อง Windows** | [onnx_detection.py:10](manga_translator/detection/onnx_detection.py#L10), [pipeline.py:43](manga_translator/pipeline/pipeline.py#L43), [renderer.py:13](manga_translator/rendering/renderer.py#L13) | `/home/parwer/...` ไม่มีบน Windows → โมเดลโหลดไม่ขึ้น และฟอนต์ fallback วาดไทยไม่ได้ (กลายเป็นกล่อง tofu) | ใช้ path สัมพัทธ์จาก package (`Path(__file__).parent/...`) หรือ env/config + ตรวจ `exists` ตอน init |
| 4 | **device string ไม่ตรง → รันบน CPU เงียบ ๆ** | [main.py:23](main.py#L23) ส่ง `"cuda"` vs [easyocr_engine.py:13](manga_translator/ocr/easyocr_engine.py#L13) เช็ค `"gpu"` | EasyOCR ได้ `gpu=False` เสมอ; `ONNXDetection` ไม่รับ device เลย (CPU ตลอด) | normalize ค่า device จุดเดียว, ส่ง `CUDAExecutionProvider` ให้ ONNX |
| 5 | **`cv2.resize` สลับ w/h (latent)** | [onnx_detection.py:42](manga_translator/detection/onnx_detection.py#L42) | `input_size=(h,w)` แต่ cv2 รับ `(w,h)` — ยังไม่พังเพราะ input สี่เหลี่ยมจัตุรัส แต่จะบิดเมื่อเปลี่ยนเป็น non-square (เช่นตอนทำ seg) ทำให้ mask เพี้ยน | ส่ง `(input_size[1], input_size[0])` + พิจารณา letterbox แทน stretch |

### 3.2 🟠 Tier 2 — คุณภาพและความทนทาน

| # | ปัญหา | ตำแหน่ง | แนวทางแก้ |
|---|-------|---------|-----------|
| 6 | **OCR ต่อข้อความไม่เรียง reading order** | [paddleocr_engine.py:29](manga_translator/ocr/paddleocr_engine.py#L29) | sort box ตาม y แล้ว x ก่อน `join` |
| 7 | **NMS อาจลบ bubble ซ้อนที่ถูกต้องทิ้ง** (เกี่ยวกับ overlap โดยตรง) | [onnx_detection.py:163](manga_translator/detection/onnx_detection.py#L163) `iou=0.5` | ใช้ per-class NMS / Soft-NMS; เมื่อมี seg ใช้ mask-IoU แทน box-IoU |
| 8 | **JSON จาก LLM เปราะ + ทั้ง batch พังพร้อมกัน** | [translator.py:67-72](manga_translator/translators/translator.py#L67-L72) | บังคับ response schema ของ provider, validate ด้วย pydantic, retry เฉพาะ item ที่หาย |
| 9 | **prompt บอก "sort right-to-left (manga)" แต่เนื้อหาเป็นคอมมิค LTR + เปิดช่องให้ LLM จัดเรียงเอง** | [prompt.py:7](manga_translator/translators/utils/prompt.py#L7) | กำหนดทิศตามภาษาจริง, สั่งห้ามเพิ่ม/ลด/เรียง item ใหม่ ต้องคืน `text_no` ครบ |
| 10 | **Render เปราะหลายจุด:** stroke=5 คงที่ (ฟอนต์เล็กเป็นก้อนทึบ), fallback min-font ล้นกล่อง, `det_box` ส่งเข้าไปแต่ไม่ถูกใช้ | [renderer.py:67](manga_translator/rendering/renderer.py#L67), [renderer.py:91-103](manga_translator/rendering/renderer.py#L91-L103) | stroke ∝ font_size, fallback ที่การันตีอยู่ในกล่อง, ใช้ mask/det_box เป็น render area |
| 11 | **ความกำกวมของ OCR engine** — default คือ EasyOCR (ทั้งภาพ) แต่ logic crop อยู่ใน PaddleOCR → มี dead path | [pipeline.py:46](manga_translator/pipeline/pipeline.py#L46) | เลือก engine เดียวเป็นทางการ, ให้ทั้งสองมี interface crop เหมือนกัน, ลบ/ทำเครื่องหมาย experimental |

### 3.3 🟡 Tier 3 — สถาปัตยกรรม / สเกล / ความปลอดภัย

| # | ปัญหา | ตำแหน่ง | แนวทางแก้ |
|---|-------|---------|-----------|
| 12 | **"async" ปลอม** — `detect/get_ocr/inpaint/render` เป็น sync CPU/GPU-bound แต่เรียกใน coroutine → บล็อก event loop, งานรูปอื่นไม่ overlap จริง; `run_batch` gather ทุกภาพไม่จำกัด → กิน RAM | [pipeline.py:157-176](manga_translator/pipeline/pipeline.py#L157-L176) | ห่อ sync ด้วย `run_in_executor`/process pool, ใส่ semaphore จำกัดจำนวนภาพ |
| 13 | **Pipeline instance เดียว global + rate-limit Event เป็น global + logic set→sleep→clear เปราะ** | [main.py:23](main.py#L23), [gemini.py:29-37](manga_translator/translators/gemini.py#L29-L37) | worker pool/lock รอบโมเดล, ทำ rate-limit เป็น token-bucket ต่อ provider |
| 14 | **config เปราะ** — `concurrent_limit=os.getenv(...)` ผูกตอน import; ไม่ตั้ง env → `int(None)` crash | [translator.py:23](manga_translator/translators/translator.py#L23), [translator.py:35](manga_translator/translators/translator.py#L35) | อ่าน env ใน body พร้อม default: `int(os.getenv("CONCURRENT_REQUESTS", "4"))` |
| 15 | **SSRF / input ไม่จำกัด** — ดึง URL ที่ผู้ใช้ส่งแล้วโหลด `<img>` ทุกตัว | [main.py:37-45](main.py#L37-L45) | allowlist domain, จำกัดขนาด/จำนวน/timeout, บล็อก IP ภายใน |
| 16 | **ไม่มี cache** — รูปเดิมแปลซ้ำเสียค่า API/เวลาทุกครั้ง | — | cache ด้วย hash ของรูป+config |

### 3.4 🔵 Tier 4 — ความเสี่ยงเฉพาะของแผน YOLOv8-seg (ดู 3.0)

| ความเสี่ยง | ผลกระทบ | แนวทางแก้ |
|------------|---------|-----------|
| **ต้นทุน/คุณภาพ annotation** | mask ที่ auto-label (เช่นจาก SAM) อาจผิดที่หาง bubble/รอยต่อ → garbage-in | bootstrap mask จากกล่องเดิมด้วย SAM, QA สุ่มตรวจ + แก้ subset ก่อนเทรน |
| **Domain shift** | ใช้กับสไตล์/สีอื่นแล้ว detect พลาด | dataset หลากสไตล์, เก็บ hard case มา fine-tune |
| **Mask หยาบ** (proto ~160²) |  bubble เล็กขอบบล็อก → clip ตัวอักษร | dilate ชดเชยตอน clean หรือ refine ขอบด้วย CV ใน crop |
| **free_text ไม่มี bubble** | mask ไร้ความหมายสำหรับ SFX | เช็ค `form_type` → ข้าม clean / ใช้กล่องเต็ม |
| ** bubble แชร์เส้นขอบ** | instance seg ยังอาจ merge | center prior + เลือก component ที่ใกล้ศูนย์กล่องสุด |
| **ONNX decode** | `det[4:].argmax()` จะกิน mask coefficients → class เพี้ยน | แยกชัดเจน: `cls=det[4:4+nc]`, `coeffs=det[4+nc:]` |
| **Inference ช้าบน CPU** | seg หนักกว่า detect | แก้ Tier 1 #4 (ใช้ GPU จริง), เลือก `yolov8n-seg` |

**กลยุทธ์ลดความเสี่ยงของแผน seg:** แยก "interface" ออกจาก "mask source" — (1) เพิ่ม field `polygon` ใน `DetectionResult` + เขียน path `clean crop → OCR → reuse ที่ render/inpaint` ให้ครบก่อน, (2) เสียบ mask ราคาถูก (SAM/CV) เป็น stand-in เพื่อทดสอบทั้งสาย, (3) เทรน YOLOv8-seg แล้ว swap แค่ mask source — ถ้า stand-in ดีพออาจไม่ต้องเทรนเลย

### 3.5 ลำดับการลงมือที่แนะนำ

1. **Tier 1 ก่อน** (ถูก/เร็ว/กันผลลัพธ์ผิดเงียบ) — โดยเฉพาะ #1, #2, #3 จะกู้ความถูกต้องคืนมามากสุดทันที
2. **Tier 2** — #6, #7 เกี่ยวกับ overlap โดยตรง
3. **Tier 4 (แผน seg)** — งานยาว ทำตามกลยุทธ์ลดความเสี่ยงข้างต้น
4. **Tier 3** — ปรับเมื่อจะ scale ขึ้น production

> **ข้อสังเกตสำคัญ:** หลายข้อใน Tier 1-2 (โดยเฉพาะ **#1 zip misalign**) ทำให้ "การแปลผิด" ได้พอ ๆ กับปัญหา OCR overlap — บางส่วนของอาการแปลผิดที่เห็นอาจมาจากบั๊กเหล่านี้โดยไม่เกี่ยวกับ overlap เลย
