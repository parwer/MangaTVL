# Changelog

บันทึกการเปลี่ยนแปลงของโปรเจกต์นี้ทั้งหมด รายการล่าสุดอยู่ด้านบนสุด

---

## [2026-06-08 15:31] เพิ่ม test_pipeline.ipynb สำหรับทดสอบ pipeline แบบ interactive

**ประเภท:** เพิ่ม feature (test)

**รายละเอียด:**
- สร้าง `test_pipeline.ipynb` ที่ root สำหรับทดสอบ pipeline แบบ interactive ตามที่ผู้ใช้ขอ — ไม่ทับ `notebook.ipynb` เดิมที่ผู้ใช้ใช้เป็น scratch
- 5 ส่วน: (1) setup imports + `DEVICE`/`IMAGE`; (2) **Detection** — `YoloDetection(task="segment").detect()` + print form_type/bbox/จำนวน polygon points + `show_image_with_polygons`; (3) **OCR** — `EasyOCREngine.get_ocr()` per-bubble + print text; (4) **Full pipeline** — `Pipeline(det_model=YoloDetection(task="segment"), provider="openrouter", model="google/gemini-2.5-flash")` + `await pipe.run(process_image=True)` + `show_images` เทียบ original/translated; (5) **Batch** — `await pipe.run_batch` หลายภาพ
- ออกแบบให้สเตจ detection/OCR รันได้โดยไม่ต้องมี API key (เทสต์ส่วน CV ได้เลย); สเตจเต็มต้องมี `.env` provider key
- validate ipynb JSON ผ่าน; ไม่แตะ source code

**ไฟล์ที่แก้ไข:**
- `test_pipeline.ipynb` — (ไฟล์ใหม่) notebook ทดสอบ pipeline ทีละสเตจ + รันเต็ม + batch

---

## [2026-06-08 15:23] เพิ่ม resize_max default 256 → 1280 (ภาพ visual context ให้ VLM)

**ประเภท:** แก้ไข

**รายละเอียด:**
- ภาพที่ส่งให้ translator (VLM) เป็น visual context ตอน `process_image=True` ถูก resize ด้วย `resize_max`; เดิม **256px เล็กเกินไป** ตัวอักษรเล็ก (SFX, ข้อความในกรอบเล็ก) อ่านไม่ครบ
- เปลี่ยน default เป็น **1024** ตาม sweet spot ที่อ้างอิง (ด้านยาว ~1100-1300px = จุดสมดุลความแม่นยำ vs token, ~1300-1600 tokens/หน้า)
- ทำได้เพราะหลังแก้บั๊ก base64 [2026-06-07 19:57] รูปถูกส่งเป็น image_url part ที่นับ token แบบ tile-based → เพิ่มเป็น 1024 ยังถูก (~2-3 tiles) ต่างจากเดิมที่ถ้าใหญ่จะระเบิด token

**ไฟล์ที่แก้ไข:**
- `pipeline.py` — `resize_max` default 256 → 1024

---

## [2026-06-07 20:42] ลบ manga_translator/tests/ (standalone scripts เก่า)

**ประเภท:** ลบ

**รายละเอียด:**
- ลบ `manga_translator/tests/` ทั้ง dir — standalone test scripts แบบรันมือเก่า (`test_detection.py`, `test_ocr.py`, `test_translator.py`) ที่ถูกแทนที่ด้วย pytest suite ใหม่ที่ `tests/` (root) แล้ว [20:40]
- `test_translator.py` มี hardcoded Groq API key — การลบช่วยเอา leaked secret ออกจาก working tree
- ⚠️ secret ยังคงอยู่ใน git history เก่า — ถ้าต้องการเอาออกจริงต้อง **rotate key ที่ Groq** + ทำ `git filter` แยก (การลบไฟล์ไม่ลบจาก history)
- ใช้ `git rm -r` (stage การลบไว้แล้ว)

**ไฟล์ที่แก้ไข:**
- `manga_translator/tests/test_detection.py` — (ลบไฟล์)
- `manga_translator/tests/test_ocr.py` — (ลบไฟล์)
- `manga_translator/tests/test_translator.py` — (ลบไฟล์, มี leaked Groq key)

---

## [2026-06-07 20:40] เพิ่ม pytest unit test suite สำหรับ pure logic (45 tests)

**ประเภท:** เพิ่ม feature (test)

**รายละเอียด:**
- โปรเจกต์ยังไม่เคยมี unit test จริง (เดิมมีแค่ standalone scripts ใน `manga_translator/tests/` ที่รันมือ + มี hardcoded key) → เพิ่ม pytest suite ล็อก behavior ของ pure logic ที่ refactor/แก้บั๊กรอบนี้
- **Setup:** ติดตั้ง `pytest` (9.0.3); `requirements-dev.txt` (ไฟล์ใหม่); `pytest.ini` ตั้ง `pythonpath=.`, `testpaths=tests` (**สำคัญ — กัน pytest ไปเก็บ standalone เดิมที่มี hardcoded key**), filterwarnings
- สร้าง `tests/` ที่ root **6 ไฟล์ 45 เทสต์** เน้น pure logic ที่ deterministic (ไม่เทสต์ model inference / API call เพราะต้อง weight/key + ไม่ deterministic):
  - `test_parse_response.py` — `parse_response` (JSON / ```json fenced / generic fenced / YAML / garbage→[] / scalar→[])
  - `test_translator_mapping.py` — `translate()` map ด้วย text_no ผ่าน FakeTranslator + `asyncio.run` (ไม่ต้อง pytest-asyncio): complete/reordered/missing(fallback ต้นฉบับ)/extra(ข้าม)/garbage/None/empty/count
  - `test_utils_common.py` — xyxy↔xywh, combine_bbox, inset_bbox, inscribed_rect (LIR), refine_unit_value_type, img_pattern, pil2cv↔cv2pil, base64 roundtrip, resize_image
  - `test_detector.py` — `DetectorBase.build_result/build_results` (bbox→int, cls_names map, polygon→int)
  - `test_clean_poly.py` — `OCREngine.clean_poly` (fill นอก polygon, offset, None)
  - `test_inpainter.py` — `expand_box`, `_masking`, `_pick_safe_polygon_mask` (uniform→mask, noisy→None, อยู่ในกรอบ)
- verify: **45 passed**; ยืนยัน collect เฉพาะ `tests/` ไม่แตะ `manga_translator/tests/`; ไม่แตะ source code ใดๆ
- รัน: `..\MangaTVL_ENV\python.exe -m pytest`

**ไฟล์ที่แก้ไข:**
- `pytest.ini` — (ไฟล์ใหม่) config pytest
- `requirements-dev.txt` — (ไฟล์ใหม่) dev deps (pytest)
- `tests/test_parse_response.py`, `tests/test_translator_mapping.py`, `tests/test_utils_common.py`, `tests/test_detector.py`, `tests/test_clean_poly.py`, `tests/test_inpainter.py` — (ไฟล์ใหม่) unit tests

---

## [2026-06-07 20:27] Optimize translator ทุก provider (dedup + robustness + performance)

**ประเภท:** refactor / แก้ bug

**รายละเอียด:**
- **Robustness (base `translator.py`):**
  - `translate()` เปลี่ยนจาก map คำแปลด้วย `zip(ocr_result, response_sorted)` (positional) เป็น **map ด้วย `text_no`** (`by_no[int(text_no)] = translated_text`) → ถ้า LLM ตอบขาด/เกิน/สลับ ไม่เลื่อนคำแปลลงผิด bubble ทั้งหมดอีกต่อไป
  - item ที่ LLM ไม่ตอบ → **fallback เป็น `ocr_item.text` เดิม** (แทนทั้งหน้าหาย)
  - เลิกกลืน exception เงียบ (`except Exception: return []`) → log error + fallback; parse fail / None response → fallback ต้นฉบับ ไม่ crash
- **Dedup:**
  - เพิ่ม `_call_with_retry(do_call)` ที่ base — รวม rate-limit (`RATE_LIMIT_FLAGS`) + exponential backoff + semaphore ไว้ที่เดียว แทน `_call_api` ที่ copy ซ้ำ 3 ไฟล์
  - เพิ่มคลาส `OpenAICompatibleTranslator(AsyncTranslatorBase)` implement `_translate` แบบ multimodal (image_url part + detail:low) สำหรับ client ที่มี `chat.completions.create`
  - `openrouter.py` + `groq.py` ยุบเหลือ subclass ว่าง (`pass`) ของ `OpenAICompatibleTranslator` → **groq ได้ image fix อัตโนมัติ** (เดิมยังส่งรูปเป็น base64 text)
- **Image fix `gemini.py`:** ส่งรูปเป็น `types.Part.from_bytes` (base64 decode จาก data-URI) แทน data-URI string ที่ถูกตีเป็น text; ใช้ `_call_with_retry` ร่วม
- **Performance:** `max_tokens` เป็น ctor param (default 8000) แทน hardcode 3 ที่
- คงชื่อคลาส `AsyncOpenRouterTranslator`/`AsyncGroqTranslator`/`AsyncGeminiTranslator` ในไฟล์เดิม → `pipeline.py` import path เดิมใช้ได้, `translators/__init__.py` ไม่ต้องแก้
- verify: `py_compile` 4 ไฟล์ + import test (MRO/ชื่อคลาส/`_call_with_retry` inherited ครบ) + unit test การ map ด้วย text_no (mock `_translate` 6 เคส: complete/reordered/missing/extra/garbage/none — map ถูกช่อง + fallback ทำงาน ไม่ crash; ลบ test script แล้ว)

**ไฟล์ที่แก้ไข:**
- `manga_translator/translators/translator.py` — text_no mapping + fallback + log; `_call_with_retry`; `max_tokens` param; `OpenAICompatibleTranslator`
- `manga_translator/translators/openrouter.py` — ยุบเหลือ subclass ของ `OpenAICompatibleTranslator`
- `manga_translator/translators/groq.py` — ยุบเหลือ subclass (ได้ image fix)
- `manga_translator/translators/gemini.py` — image เป็น `Part.from_bytes` + shared retry

---

## [2026-06-07 20:17] ย้าย pipeline.py ไป project root + refactor ให้อ่านง่าย

**ประเภท:** refactor

**รายละเอียด:**
- ย้าย `manga_translator/pipeline/pipeline.py` → `pipeline.py` ที่ project root (ข้างๆ main.py) ตามที่ผู้ใช้ขอ; เปลี่ยน relative imports (`from ..xxx`) เป็น absolute (`from manga_translator.xxx`); ลบ package `manga_translator/pipeline/` ทั้ง dir
- อัปเดต importers 3 จุด: `main.py` → `from pipeline import Pipeline`; `manga_translator/__init__.py` ลบบรรทัด `from .pipeline.pipeline import Pipeline` (คง `import torch` กัน WinError 127 + คอมเมนต์ path ใหม่)
- **refactor:**
  - รวม logic ซ้ำของ `run()` กับ `run_batch()` — เดิม `process_single` ใน run_batch ซ้ำกับ run เกือบทั้งหมด → สร้าง `_run_on_image(image, process_image)` ตัวเดียว, `run_batch` เหลือ `[self.run(p) for p] + gather`
  - ยุบ `_init_translator` ที่ซ้ำ 3 branch → สร้าง kwargs dict ครั้งเดียวแล้วเลือก client+class ตาม provider
  - ตัด dead code ใน run_batch — เดิม detect notebook/terminal แล้ว import `_tqdm` คนละตัว แต่สุดท้ายใช้ `tqdm_asyncio.gather` เฉยๆ (ไม่ใช้ `_tqdm` เลย) → เหลือ `show_progress` flag เลือก `tqdm_asyncio.gather` / `asyncio.gather`
  - แก้ `font_path` default ที่ hardcode เป็น Linux path `/home/parwer/...` (พังบน Windows) → `None` แล้วให้ `TextRenderer` ใช้ `DEFAULT_FONT_PATH` ของตัวเอง
  - ตัด import ไม่ใช้ (`Any`/`Dict`, `resize_image` จาก utils ที่ซ้ำ method, `show_images`, `DetectionResult`, `SimpleLamaInpainter` type hint) และ wrapper `_load`/`_translate` ที่เป็น passthrough (inline `load_image` / `translator.translate`)
- verify: `py_compile` + import test จริง `from pipeline import Pipeline` ผ่าน, torch-before-paddle ordering ยังทำงาน
- ⚠️ **ผู้ใช้ต้องแก้ notebook:** เปลี่ยน import เป็น `from pipeline import Pipeline`

**ไฟล์ที่แก้ไข:**
- `pipeline.py` — (ไฟล์ใหม่) ย้ายมาจาก package + refactor
- `manga_translator/pipeline/pipeline.py` — (ลบไฟล์)
- `manga_translator/pipeline/__init__.py` — (ลบไฟล์)
- `manga_translator/__init__.py` — ลบ import Pipeline, คง torch
- `main.py` — `from pipeline import Pipeline`

---

## [2026-06-07 19:57] แก้บั๊กส่งรูปเป็น base64 text → VLM translation กิน token ~50-400x

**ประเภท:** แก้ bug

**รายละเอียด:**
- เมื่อเปิด `process_image=True` ส่งรูปเป็น visual context ให้ translator → token พุ่งเกือบ 1M/รูป; root cause ไม่ใช่รูปใหญ่ แต่เป็นบั๊ก: รูปถูกส่งเป็น **base64 data-URI string ธรรมดา** ลงใน message `content` → OpenAI-compatible API (OpenRouter) ตีความเป็น **ข้อความ** → tokenize base64 ทั้งก้อนเป็น text token
- วัดจริงด้วย `tiktoken` (o200k) บน 17.jpg: full-res = **173,960 token**, resize256 = **14,627 token** ต่อ 1 รูป (ไม่ใช่ ~1000 อย่างที่คาด); รูปใหญ่ระดับ 1600px ~388,796 token → รวมหลายรูป/หน้าสูงก็แตะ ~1M ได้

- **อธิบายเพิ่ม — ทำไม token ระเบิด, กลไกทำงานยังไง, ทำไมลดลงได้:**
  - **โมเดลนับ token ของ "ข้อความ" กับ "รูป" คนละวิธี:**
    - *ข้อความ* → ผ่าน tokenizer (BPE) ซอย string เป็น sub-word tokens
    - *รูป* → โมเดล multimodal ไม่ได้นับจากความยาวข้อมูลรูป แต่นับจาก **ขนาด/จำนวน tile** (เช่น Gemini ~258 token ต่อ tile 768px; OpenAI detail:"low" คิดแบบ flat ~85 token) — ไม่เกี่ยวกับว่าไฟล์ใหญ่แค่ไหน
  - **ทำไมระเบิด:** เดิมเราแปลงรูปเป็น base64 data-URI string (`"data:image/jpeg;base64,/9j/4AAQ..."`) แล้ววางเป็น `content` ที่เป็น **string ธรรมดา** → API ไม่มีทางรู้ว่าเป็นรูป จึงปฏิบัติกับมันเป็น **ข้อความ** → เอา base64 ทั้งก้อนไปเข้า tokenizer
  - **ทำไมตัวเลขสูงมาก:** base64 เป็นสตริงตัวอักษร/ตัวเลขแบบสุ่ม BPE ยุบเป็น sub-word ที่ใช้ซ้ำไม่ได้ → อัตราส่วนแย่มากที่ ~**0.68 token/char** (1 token ต่อ ~1.46 ตัวอักษร) ดังนั้นยิ่ง base64 ยาว (รูปใหญ่/quality สูง) token ยิ่งพุ่งเชิงเส้น
  - **ทำไมลดลงได้:** พอย้าย data-URI **ตัวเดิม** ไปไว้ใน image content part `{"type":"image_url",...}` → API decode มันเป็นรูปจริง แล้วคิดเงินแบบ **image token (tile-based)** แทนที่จะ tokenize base64 เป็น text → รูปเดียวกันเป๊ะ แต่จาก ~14k–174k+ เหลือ ~258–1,000 (~50–400x) — **ลดเพราะเปลี่ยน "วิธีแนบรูป" ไม่ใช่เพราะย่อรูป**
  - **ตัวช่วยเสริม (รอง):** `detail:"low"` cap tile ของฝั่ง OpenAI-style, resize_max=256 ให้รูปเหลือ ~1 tile, JPEG quality=60 ลด payload — ทั้งหมดนี้ลดเพิ่มเล็กน้อย แต่ตัวที่แก้ root cause จริงคือการแนบเป็น image part

- **(1) `openrouter.py`** — เปลี่ยน `_translate` จากใส่ data-URI string ใน `content` ตรงๆ เป็น proper multimodal content part `{"type":"image_url","image_url":{"url": image, "detail":"low"}}` รวมกับ text ใน user message เดียว → API นับเป็น **image token** (gemini-2.5-flash via OpenRouter tile-based ~258-1000 token) ประหยัด ~50-400x; `detail:"low"` เป็น hint ประหยัดเพิ่ม
- **(2) `common.py`** — `convert_img_to_base64(image, quality=60)` เพิ่ม param quality ส่งเข้า `image.save(format="JPEG", quality=...)` ทั้ง branch pil/cv2 → ลด payload ~58% (q95 40,907 → q60 17,219 chars); ไม่ลด image-token count โดยตรงแต่ payload เล็กลง upload เร็วขึ้น; param มี default ไม่กระทบ caller เดิม (main.py encode ภาพ output)
- ไม่แตะ `gemini.py`/`groq.py` (ไม่ได้ใช้จริง — main.py ใช้ openrouter), `translator.py` base (data-URI flow เดิมใช้กับ image_url ได้), `pipeline.resize_max=256` (เหมาะกับ "ประหยัดสุด" อยู่แล้ว)
- verify ด้วย `py_compile` + วัด token เทียบจริง; ติดตั้ง `tiktoken` ใน MangaTVL_ENV ไว้สำหรับวัด

**ไฟล์ที่แก้ไข:**
- `manga_translator/translators/openrouter.py` — ส่งรูปเป็น image_url content part + detail:low
- `manga_translator/utils/common.py` — `convert_img_to_base64` เพิ่ม `quality=60`

---

## [2026-06-07 19:36] เขียน EasyOCREngine ใหม่ให้ใช้ pipeline เดียวกับ PaddleOCR

**ประเภท:** refactor

**รายละเอียด:**
- เขียน `EasyOCREngine` ใหม่ใน `manga_translator/ocr/easyocr_engine.py` ให้ mirror โครงสร้าง `PaddleOCREngine` เพื่อใช้เป็น drop-in alternative เทียบคุณภาพ OCR — ผู้ใช้อยากดูว่า EasyOCR จะอ่านดีกว่า PaddleOCR ไหม
- เดิม EasyOCREngine ถูกมาร์ค "No Longer Used" + ใช้วิธีคนละแบบ (OCR ทั้งภาพด้วย `readtext(paragraph=True)` แล้ว match เข้า bubble ผ่าน `match_ocr_to_bubbles`) ซึ่ง overlap ข้าม bubble ได้ง่าย
- เขียนใหม่: เพิ่ม `ocr(cropped_images)` รัน `readtext(detail=1, paragraph=True)` บน crop คืน `(text, boxes)` พิกัด crop-local; `get_ocr()` loop detection แต่ละ bubble → `crop(bbox)` → `clean_poly` ลบนอก polygon → `ocr` → map boxes กลับ full-image → 1 `OCRResult` ต่อ bubble (เหมือน PaddleOCR เป๊ะ)
- เลิก dependency กับ `bubble_ocr_matcher` / `match_ocr_to_bubbles`; `clean_poly` สืบทอดจาก base `OCREngine`
- `poly_to_xyxy` คืน `int` (ตรง schema `OCRResult.boxes: list[list[int]]`) แทน float เดิม; device check รับทั้ง `"gpu"` และ `"cuda"` (เดิมรับแค่ `"gpu"`)
- verify ด้วย `py_compile` ผ่าน
- วิธีสลับมาเทียบ: `Pipeline(ocr_engine=EasyOCREngine(language="en", device="cuda"))`

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/easyocr_engine.py` — เขียนใหม่ทั้งไฟล์ให้ใช้ per-bubble crop pipeline เหมือน PaddleOCR

---

## [2026-06-07 19:21] ปรับ border-uniformity mask selector ให้ adaptive (pass rate 36%→73-100%)

**ประเภท:** แก้ bug

**รายละเอียด:**
- border-uniformity selector ที่ implement ตอน [18:40] ทำงานถูกแต่ pass rate ต่ำ — diagnostic ด้วย model จริงพบว่าบนภาพ 17.jpg ผ่าน safety check แค่ 4/11 bubbles (36%) → ส่วนใหญ่ตกไป box-based fallback ที่ยังมี artifact เดิม (เป็นเหตุที่ debug print "All items safely inpainted..." ไม่เคย fire — เพราะ `if not demoted:` ต้องการ 100% pass)
- root cause 3 ข้อ: (1) `erode_steps=(0,2,4,6,8,10)` คงที่ ตื้นเกิน — bubble ที่ผ่านเพิ่งผ่านที่ e=10, ตัวที่ fail std ยังลดลงเรื่อยๆ ที่ e=10 (ยังไม่ถึงก้น) เพราะ YOLO-seg polygon overshoot บางที 30+ px; (2) bubble เล็ก over-erode → std พุ่งกลับขึ้น; (3) `max_std=15` strict ไป
- **(a) Adaptive erode depth** — คำนวณ `max_erode = max(6, int(diag * 0.12))` จาก polygon bbox diagonal กระจาย `n_steps=8` ขั้น แทน tuple คงที่ → bubble ใหญ่ (diag ~489) erode ลึกถึง ~58px, bubble เล็ก (diag ~125) แค่ ~15px ไม่ over-erode
- **(b) `max_std` 15→25** — จาก diagnostic: bubble ที่ขอบยังอยู่ในของ bubble จอดที่ std 16-25 (std สูงเพราะ glyph ในของ ไม่ใช่ขอบ); เคสขอบทาบเส้น bubble จริง std 50-110 ยังถูก reject สบาย
- **(c) Over-erode guard** — ข้าม candidate ที่ erode จน area < 30% ของ base (`min_area_frac=0.3`) กันเลือก mask เล็กจิ๋วที่ border บังเอิญ uniform
- **(d) Orchestrator log** — แทน `print("All items...")` ที่แทบไม่ fire ด้วย `[inpaint] polygon safe-fill: n/total, box-fallback: m` ที่ gate ด้วย `show_log`
- ผลทดสอบ pass rate: 17.jpg 4→8/11 (73%), 5.jpg 8→10/11 (91%), 13.jpg 4→7/7 (100%), 111.jpg 3→5/5 (100%); 3 ตัวที่ยัง fail ใน 17.jpg เป็น SFX hand-drawn + bubble ที่ border ชนตัวอักษรจริง = true reject (ควรตก box-based ถูกต้อง ไม่ใช่ false negative)
- verify ด้วย `py_compile` + diagnostic script (เขียนชั่วคราว รันด้วย YoloDetection จริง แล้วลบทิ้ง ไม่ commit); `_masking` box-based fallback คงเดิมไม่แตะ
- parameters ที่ปรับได้ภายหลัง: `max_erode_frac` (0.12), `max_std` (25), `min_area_frac` (0.3), `n_steps` (8)

**ไฟล์ที่แก้ไข:**
- `manga_translator/inpainting/inpainter.py` — `_pick_safe_polygon_mask` adaptive erode + max_std 25 + over-erode guard; `inpaint()` log via show_log

---

## [2026-06-07 18:51] Revert OCR y-sort กลับเป็นลำดับธรรมชาติของ PaddleOCR

**ประเภท:** Revert

**รายละเอียด:**
- ใน `PaddleOCREngine.ocr()` ตอน [16:45] เพิ่ม sort `(text, box)` pairs ตาม `(y_center, x_center)` ก่อน join เพราะคิดว่าจะแก้เคส "bubble multi-line ลำดับสลับ" — ผู้ใช้รายงานจากภาพทดสอบจริงว่า **ผลตรงข้าม**: y-sort ทำให้บาง bubble เพี้ยน เพราะ PaddleOCR คืน `rec_boxes` มาตามลำดับ "ที่เป็นธรรมชาติของ detector" ซึ่งสำหรับ bubble ที่เทสต์ดูจะตรงกับ reading order จริงอยู่แล้ว → การไป force sort ดิบๆ ทำลายลำดับที่ใช้ได้
- เคสตัวอย่างที่ผู้ใช้ส่งให้: bubble `bbox=[72, 302, 164, 498]` (อ่าน top-to-bottom ปกติของ Western comic ไม่ใช่ artistic choice) — หลัง y-sort ได้ `"PEOPLE?! ARE YOU THE HELL NING! WHO I'M RUN- COURSE OF"` (ลำดับสลับจาก reading order) → translator แปลเป็น "คนเหรอ?! พวกแกน่ะเหรอที่ฉันกำลังวิ่งหนีอยู่เนี่ย!" ซึ่ง semantic เลื่อนจาก "OF COURSE I'M RUNNING! WHO THE HELL ARE YOU PEOPLE?!"
- กลับมาใช้ `texts.extend / boxes.extend` ตามที่ PaddleOCR ส่งมา (เหมือนก่อน [16:45])
- verify ด้วย `python -m py_compile` ผ่าน
- บทเรียน: ลำดับธรรมชาติของ PaddleOCR ฉลาดกว่าการ sort ตาม geometric heuristic ง่ายๆ — ถ้าอนาคตเจอ bubble ที่ลำดับเพี้ยนจริงๆ ค่อยพิจารณา reading-order detection ที่ฉลาดกว่า (เช่น cluster boxes เป็นบรรทัดก่อนแล้วเรียงในบรรทัด) ไม่ใช่ raw sort ตาม y

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/paddleocr_engine.py` — `ocr()` คืนลำดับ append เดิม (ไม่ sort)

---

## [2026-06-07 18:40] Inpainter ใช้ border-uniformity mask selection แบบ PanelCleaner

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- หลังลองใช้ polygon เป็น inpaint mask 2 ครั้งแล้วล้มเหลวเพราะ YOLOv8-seg polygon ของเรา overshoot นอกขอบ bubble 3-8 px ([16:45], [18:00]) → revert กลับเป็น box-based ([18:09]) → ครั้งนี้ adopt อัลกอริทึม `pick_best_mask` จาก `pcleaner/image_ops.py` ของ [VoxelCubes/PanelCleaner](https://github.com/VoxelCubes/PanelCleaner) มา เพราะ PanelCleaner ออกแบบมาเพื่อปัญหานี้พอดี — เลือก mask candidate ที่ขอบ "วางบนพื้นสีเดียวกันสม่ำเสมอ" → ขอบ bubble line ปลอดภัยโดยอัตโนมัติ
- ความแตกต่างจาก PanelCleaner: ของเขา**dilate ออก** หา safe boundary (เพราะ detector ของเขาให้ mask tight); ของเรา**erode เข้า** (เพราะ polygon ของเรา loose/overshoot) — algorithm สมมาตร ทิศทางกลับกัน
- **(1) `_pick_safe_polygon_mask(image_gray, polygon, erode_steps=(0,2,4,6,8,10), max_std=15.0)`** — สร้าง candidate mask หลายเวอร์ชันด้วย `cv2.fillPoly` + `cv2.erode` ตาม `erode_steps`; วัด standard deviation ของ pixel ที่ขอบ mask (เก็บขอบด้วย `cv2.morphologyEx(MORPH_GRADIENT)`); เลือก candidate ที่ std ต่ำสุด — std ต่ำ = ขอบอยู่ในของ bubble interior ขาวสม่ำเสมอ, std สูง = ขอบทาบเส้น bubble ดำ; ถ้า std ของตัวที่ดีที่สุดยัง > `max_std` คืน `None` ตามปรัชญา PanelCleaner "ทำไม่ได้ → ไม่ทำ"
- **(2) `_safe_fill_polygons(image, inputs) -> (image, demoted)`** — ลูปทุก item; ถ้ามี polygon และผ่าน safety check → ทาสีขาว `(255,255,255)` ตรงๆ ลง image (อาศัย insight ว่า bubble interior = ขาวอยู่แล้ว ไม่ต้องเรียก inpaint algorithm); item ที่ไม่มี polygon หรือ skip → ใส่ `demoted` list
- **(3) `inpaint()` orchestrator** — เรียก `_safe_fill_polygons` ก่อน; ถ้า `demoted` ว่าง → return image ที่ทาขาวแล้ว ไม่ต้องเรียก `_inpaint` algorithm เลย; ถ้ามี demoted → `_masking` + `_inpaint` เฉพาะ demoted (free_text, detect-only model, หรือ bubble ที่ skip จาก safety) ใช้ box-based + cv2.inpaint เป็น fallback
- `_masking()` คงเดิม (box-based ตาม state revert) → `show_masks`/`get_masks` ใน notebook ไม่กระทบ
- Parameters ที่อาจต้อง tune หลังเทสต์: `erode_steps` (ขยายช่วงถ้าบาง bubble overshoot > 10 px), `max_std` (ลด → strict ขึ้น, skip เยอะขึ้น; เพิ่ม → ผ่อนคลาย)
- verify ด้วย `python -m py_compile` ผ่าน; visual verify รอผู้ใช้รัน notebook บนหน้าทดสอบเดิม

**ไฟล์ที่แก้ไข:**
- `manga_translator/inpainting/inpainter.py` — เพิ่ม `_pick_safe_polygon_mask` + `_safe_fill_polygons`; ปรับ `inpaint()` orchestrator ให้ pre-process polygon items ด้วย safe-fill ก่อนตกไปยัง box-based fallback

---

## [2026-06-07 18:09] Revert inpainter กลับเป็น box-based mask อย่างเดียว (polygon overshoot)

**ประเภท:** Revert

**รายละเอียด:**
- เลิกใช้ polygon ทั้งจาก [16:45] (polygon เป็น cv2.inpaint mask) และ [18:00] (ทาขาวตามรอย polygon) — สาเหตุเดียวกัน: **polygon ที่ YOLO-seg ตรวจมาไม่ tight พอ มัก overshoot นอกขอบ bubble** → ทั้งสองวิธีไป erase พิกเซลขอบ bubble บางส่วน ทำให้ bubble border ขาด/รู้สึกแปลก (เห็นชัดในภาพทดสอบล่าสุดที่ผู้ใช้ส่งมา bubble ใหญ่ฝั่งขวาขอบหายเป็นช่วงๆ)
- ใช้ `git checkout 0f72aa8 -- manga_translator/inpainting/inpainter.py` restore เป็น state ก่อน d4702ad — `_masking` กลับมาใช้ box-based ของ OCR boxes + `expand_margin=2` + `cv2.rectangle` แบบดั้งเดิม ไม่มี polygon branch และไม่มี `_fill_polygons_white`
- การเปลี่ยนนี้เป็น **revert ใน working tree (uncommitted)** — commit `d4702ad` ใน history ยังคงมี polygon mask อยู่ ถ้าจะ persist revert ต้อง commit revert แยก
- verify ด้วย `python -m py_compile` ผ่าน
- ขั้นต่อไป: รอผู้ใช้เลือกวิธี CV ใหม่ที่ทน polygon overshoot ได้ (เช่น erode polygon + threshold dark pixels) — ยังไม่ implement ในรอบนี้

**ไฟล์ที่แก้ไข:**
- `manga_translator/inpainting/inpainter.py` — restore เป็น box-based mask ของ `0f72aa8`

---

## [2026-06-07 18:00] เปลี่ยน inpaint ของ bubble polygon เป็นทาสีขาวตรงๆ (กัน cv2.inpaint smear)

**ประเภท:** แก้ bug

**รายละเอียด:**
- รอบที่แล้ว ([16:45]) เปลี่ยน `_masking` ให้ใช้ `cv2.fillPoly` ของ bubble polygon เป็น inpaint mask แล้วส่งเข้า `cv2.inpaint` — ตั้งใจให้ลบทั้ง bubble interior สะอาด แต่พบว่า OpenCV `cv2.inpaint` (telea/NS) ออกแบบมาสำหรับ **หลุมเล็กๆ** ไม่ใช่พื้นที่ใหญ่; พอบอกให้ reconstruct ทั้ง bubble interior มันดึง texture จากนอก bubble (พื้นเข้ม/screentone/artwork) มา fill → smearing หนัก + **ทำลายเส้นขอบ bubble** (เห็นชัดในภาพทดสอบที่ผู้ใช้ส่งมา bubble ใหญ่ฝั่งขวาเส้นขอบหายเกลี้ยง)
- แก้โดยใช้ความจริงที่ว่า bubble interior เป็นสีขาวอยู่แล้ว — แค่ "ทาสีขาวทับ" ตรงๆ ก็พอ ไม่ต้องเรียก algorithm reconstruction
- **(1) เพิ่ม `_fill_polygons_white(image, inputs)`** ใน `InpainterBase` — ใช้ `cv2.fillPoly` เติม `(255,255,255)` ทับ bubble polygon ทั้งวงโดยตรงบนภาพต้นทาง flat fill, ไม่มี smear, เร็วกว่ามาก
- **(2) ปรับ `inpaint()` orchestrator** เรียก `_fill_polygons_white` ก่อน `_masking` → ภาพที่ส่งต่อให้ `_inpaint` มี bubble interior เป็นสีขาวสะอาดอยู่แล้ว
- **(3) ปรับ `_masking()`** ให้ `continue` สำหรับ item ที่มี `segmentation` (เพราะถูกจัดการในขั้นทาขาวแล้ว) — เหลือเฉพาะ free_text / detect-only model ที่ยังใช้ box-based mask + `cv2.inpaint` ตามเดิม
- ไม่ใช่ revert ของ [16:45] — แนวคิด "ใช้ polygon ลบทั้ง bubble" ยังคงอยู่ แค่เปลี่ยน mechanism จาก inpaint algorithm → direct fill ซึ่งถูกกับลักษณะของ bubble manga (พื้นใน = ขาวล้วน)
- verify ด้วย `python -m py_compile` ผ่าน; visual verify รอผู้ใช้รัน notebook เพื่อยืนยัน bubble ขอบครบ + interior สะอาด

**ไฟล์ที่แก้ไข:**
- `manga_translator/inpainting/inpainter.py` — เพิ่ม `_fill_polygons_white`; `inpaint()` orchestrator เรียกก่อน `_masking`; `_masking()` skip item ที่มี polygon

---

## [2026-06-07 16:45] แก้ OCR ลำดับสลับ + กำจัด inpaint artifact ด้วย bubble polygon

**ประเภท:** แก้ bug

**รายละเอียด:**
- แก้ 2 ปัญหาในงานเดียวกัน (จากการวิเคราะห์ผลทดสอบจริงที่เห็นใน `TranslationResult[]` ของ pipeline):
- **(1) OCR multi-line bubble join ผิดลำดับ** — `PaddleOCREngine.ocr()` ใน [paddleocr_engine.py:22-39](manga_translator/ocr/paddleocr_engine.py#L22-L39) เดิม `" ".join(texts)` ตามลำดับที่ PaddleOCR คืนมา ซึ่งไม่ใช่ reading order → multi-line bubble ได้ string สลับ → translator แปลเพี้ยน semantic เช่นเคสจริง `"PEOPLE?! ARE YOU THE HELL NING! WHO I'M RUN- COURSE OF"` ที่จริงคือ `"OF COURSE I'M RUNNING! WHO THE HELL ARE YOU PEOPLE?!"`; เปลี่ยนเป็น zip `(text, box)` แล้ว sort ด้วย key `(y_center, x_center)` ก่อน join — top-to-bottom เป็นหลัก, left-to-right ในบรรทัดเดียวกัน
- **(2) Inpaint ลบตัวอักษรเก่าไม่หมด** — `InpainterBase._masking()` ใน [inpainter.py:35-49](manga_translator/inpainting/inpainter.py#L35-L49) เดิมสร้าง mask จาก `item.boxes` (OCR boxes) เท่านั้น → ถ้า OCR fail (โดยเฉพาะ SFX bubble) หรือ box ตึงเกินขอบ glyph → เหลือซากตัวเก่าหลัง inpaint ปนกับ text แปลใหม่; เพิ่ม branch ใช้ `cv2.fillPoly(mask, [polygon], 255)` ของ `item.detection_result.segmentation` เมื่อมี — ลบทั้ง bubble interior สะอาดเอี่ยม **ไม่ขึ้นกับ OCR เห็นตัวอักษรหรือไม่**; ตกกลับใช้ box-based เดิมเมื่อ segmentation = None (free_text / detect-only model)
- **Translator defensive fix** — `translator.py:23` `concurrent_limit=os.getenv("CONCURRENT_REQUESTS") or 1` กัน `int(None)` crash เมื่อไม่ได้ตั้ง env

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/paddleocr_engine.py` — `ocr()` sort pairs ตาม `(y_center, x_center)` ก่อน join, return signature เดิม
- `manga_translator/inpainting/inpainter.py` — `_masking()` เพิ่ม polygon branch (cv2.fillPoly) เมื่อมี segmentation; box-based fallback เมื่อไม่มี
- `manga_translator/translators/translator.py` — fallback `or 1` กัน int(None) crash

---

## [2026-06-07 15:53] แก้ TextRenderer ไม่ให้ข้อความล้นออกนอกเส้น bubble โดยใช้ polygon LIR

**ประเภท:** แก้ bug

**รายละเอียด:**
- ปัญหาเดิม: renderer ใช้กรอบสี่เหลี่ยม (`combine_bbox` ของ OCR boxes) เป็นพื้นที่ layout แต่ bubble จริงเป็นรูปไข่/อิสระ — มุมและส่วนโค้งของ bubble อยู่ใน bbox แต่ "นอก bubble" → ข้อความที่ผ่านการ fit เข้า bbox + auto-shrink font แล้วยังคงล้นออกนอกเส้น bubble (เห็นชัดในภาพทดสอบล่าสุด)
- แนวทางแก้: ใช้ `DetectionResult.segmentation` (polygon จาก YOLOv8-seg ที่ pipeline พามาถึง renderer ครบอยู่แล้ว) คำนวณ **Largest Inscribed Rectangle (LIR)** — สี่เหลี่ยม axis-aligned ที่ใหญ่ที่สุดที่ฝังในรูปทรง bubble จริง — แล้วใช้แทนกรอบสี่เหลี่ยมเดิม
- **(1) Geometry helpers ใน `utils/common.py`:** เพิ่ม `inscribed_rect(polygon, image_shape, padding=0)` ที่ใช้ `cv2.fillPoly` rasterize polygon เป็น binary mask (downsample ถ้า bubble ใหญ่กว่า 200px เพื่อ bound เวลา) → `cv2.erode` ตาม padding เผื่อ stroke/margin → หา LIR ด้วย row-wise largest-rectangle-in-histogram + monotonic stack O(H×W) → scale กลับเป็นพิกัดภาพ; และเพิ่ม `inset_bbox(bbox, pad)` สำหรับ bbox fallback
- **(2) Render area priority ใน `renderer.py`:** เพิ่ม `_compute_box(det, image_shape, pad)` เลือกตามลำดับ: polygon LIR → `det.bbox` inset → fallback `extract_text_box` เดิม — แก้ปัญหาที่ `det_box` ถูก thread เข้าทุก method แต่ไม่เคยถูกใช้
- **(3) Stroke ที่ scale ตาม font_size:** เดิม `stroke_width=5` คงที่ — ฟอนต์เล็กกลายเป็นก้อนสีดำและ stroke ดันออกนอก box แม้ glyph จะ fit; เปลี่ยนเป็น `_stroke_for_font(fs) = max(1, fs//12)` และส่ง `stroke_width=stroke` เข้า `multiline_textbbox` ใน `_fits_in_box` เพื่อให้ fit check วัดรวมขอบ stroke (ไม่งั้น glyph fit แต่ stroke ทะลุ)
- **(4) Fallback ที่ไม่ overflow:** เดิมเมื่อ binary search ไม่เจอ font ที่ fit จะ dump text ทุก 4 token ที่ `min_font_size` → ล้นเสมอ; เปลี่ยนเป็น 2 ชั้น — (ก) shrink font ต่ำกว่า `min_font_size` ลงไปถึง `ABSOLUTE_MIN_FONT_SIZE=10` ก่อน, (ข) ถ้ายังไม่ fit ใช้ `_truncate_to_fit` binary search หาจำนวน token สูงสุดที่ fit แล้วเติม `"…"` — การันตี"ไม่ล้น"ทุกเคส แม้จะเป็น truncated/อ่านไม่ครบ
- ไม่แตะ `paddleocr_engine.py`, `translator.py`, `schemas/interface.py`, `yolo_detection.py` — polygon ถูกพาผ่านมาอยู่แล้ว ไม่ต้องแก้

**ไฟล์ที่แก้ไข:**
- `manga_translator/utils/common.py` — เพิ่ม `inscribed_rect` และ `inset_bbox`
- `manga_translator/rendering/renderer.py` — เพิ่ม `_compute_box` / `_truncate_to_fit` / `_stroke_for_font`; ปรับ `render`, `_render_single`, `wrap_extraction`, `_fits_in_box` ให้ใช้ polygon LIR + stroke scaling + fallback ใหม่

---

## [2026-06-07 14:54] แก้ PaddleOCR crash (oneDNN/PIR) ด้วย enable_mkldnn=False

**ประเภท:** แก้ bug

**รายละเอียด:**
- `PaddleOCR.predict` ระเบิดตอน text detection ด้วย `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>] (...onednn_instruction.cc:118)` — error อยู่ลึกใน Paddle inference ไม่เกี่ยวกับโค้ด pipeline (การ crop/clean_poly ทำงานปกติ แล้วค่อยพังตอนเรียก `self.ocr(...)`)
- สาเหตุ: เวอร์ชันที่ติดตั้งคือ **paddleocr 3.5.0 / paddle 3.3.1** ซึ่ง PIR executor ตัวใหม่ของ Paddle 3.3 แปลง attribute แบบ `Double`-array ผ่าน path ของ oneDNN (MKLDNN) บน CPU ไม่ได้ → detection model พังทันที เป็น bug ฝั่ง backend ของ Paddle ไม่ใช่ของเรา
- แก้โดยส่ง `enable_mkldnn=False` เข้า constructor ของ `PaddleOCR` (ผ่าน `**kwargs` → PaddleX) ให้ inference วิ่งผ่าน CPU kernel ปกติแทน oneDNN — ทดสอบแล้ว `predict` รันผ่าน คืนผลลัพธ์ได้ ไม่ raise
- หมายเหตุ: ต้อง restart kernel ของ notebook ก่อน re-run เพราะ object `ocr_engine` เดิมถูกสร้างตอน MKLDNN ยังเปิดอยู่
- หมายเหตุเพิ่ม: `use_angle_cls=True` เป็นชื่อพารามิเตอร์ของ PaddleOCR 2.x — ใน 3.x เปลี่ยนเป็น `use_textline_orientation` ตอนนี้ค่านี้ถูก `**kwargs` กลืนเฉยๆ ไม่มีผล (ยังไม่แก้ รอยืนยันว่าต้องการ angle classification จริงไหม)

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/paddleocr_engine.py` — เพิ่ม `enable_mkldnn=False` ใน `PaddleOCR(...)` พร้อมคอมเมนต์อธิบายสาเหตุ

---

## [2026-06-07 14:45] implement clean_poly สำหรับลบ pixel นอก segmentation polygon ก่อน OCR

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- implement `clean_poly` ใน `OCREngine` (เดิมเป็น `pass` ว่างเปล่า) — ลบ (fill ขาว) ทุก pixel ที่อยู่นอก polygon ของ bubble เพื่อไม่ให้ OCR อ่าน bubble/artwork ข้างเคียงที่ติดมาในกรอบ crop สี่เหลี่ยม ทำให้ผล OCR สะอาดขึ้น
- ใช้ `cv2.fillPoly` สร้าง mask แล้ว fill นอก polygon; รองรับทั้งภาพ RGB / RGBA / grayscale (pad/trim ค่า fill ตามจำนวน channel)
- รับพารามิเตอร์ `offset=(x1, y1)` เพราะ `det.segmentation` เก็บพิกัดแบบ full-image แต่ภาพที่ส่งเข้ามาถูก crop ที่ bbox แล้ว — ต้อง shift polygon ให้ตรงกรอบ crop
- คืนภาพเดิมทันทีเมื่อ `poly` เป็น `None`/ว่าง (รองรับ detect-only model ที่ `segmentation = None`)
- แก้ลำดับการเรียกใน `PaddleOCREngine.get_ocr` — เดิมเรียก `clean_poly` **หลัง** OCR และทิ้งค่าที่คืนมา ทำให้ OCR ไม่ได้ใช้ภาพที่ clean เลย; เปลี่ยนเป็น clean ก่อนแล้วป้อนผลเข้า `self.ocr(...)` พร้อมส่ง `offset=(x1, y1)`

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/ocr_engine.py` — implement `clean_poly` (เพิ่ม import `cv2`, `numpy`)
- `manga_translator/ocr/paddleocr_engine.py` — clean ก่อน OCR และส่ง offset เข้า `clean_poly`

---

## [2026-06-07 14:27] เพิ่ม show_image_with_polygons สำหรับ visualize segmentation mask

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เพิ่มฟังก์ชัน `show_image_with_polygons` ใน `manga_translator/utils/common.py` สำหรับวาด polygon outline ลงบนภาพ — ใช้ตรวจดูผลของ `DetectionResult.segmentation` ที่ได้จาก seg model (เช่น YoloDetection ที่โหลด `.pt`/`.onnx` แบบมี seg head) ระหว่างทดสอบใน notebook
- ตาม pattern เดียวกับ `show_image_with_boxes` ที่มีอยู่ — รับ image (PIL/cv2) + list ของ polygons (แต่ละอันคือ list ของ `[x,y]` points), optional cls_text/color/fig_size; ใช้ `cv2.polylines` วาดเส้นปิด
- ข้าม polygon ที่เป็น `None` หรือว่างเปล่าโดยอัตโนมัติ — รองรับเคสที่ DetectionResult บางตัวมาจากโมเดล detect-only (`segmentation = None`) ผสมกับโมเดล seg ใน batch เดียวกัน

**ไฟล์ที่แก้ไข:**
- `manga_translator/utils/common.py` — เพิ่มฟังก์ชัน `show_image_with_polygons`

---

## [2026-06-07 13:05] Implement YoloDetection โดยสืบจาก DetectorBase

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- แทนที่ stub เดิมของ `yolo_detection.py` (ที่ instantiate `YOLO()` ระดับ module + เรียก `predict()` ไม่มี args) ด้วยคลาส `YoloDetection(DetectorBase)` ที่ใช้งานได้จริงและเข้ากับ contract เดียวของ `DetectorBase` — pipeline สลับมาใช้ตัวนี้แทน `ONNXDetection` ได้โดยไม่กระทบ calling code
- ใช้ `ultralytics.YOLO` ซึ่งรองรับทั้งไฟล์ `.pt` และ `.onnx` ในตัวเดียว และทำงานได้ทั้งโมเดล detect และ seg — เก็บ polygon ลง `DetectionResult.segmentation` เมื่อ `result.masks` มีค่า, ถ้าเป็นโมเดล detect ล้วน polygon = None ตามปกติ
- ส่ง output ของ ultralytics (`result.boxes.xyxy`, `result.boxes.cls`, `result.masks.xy`) เข้า `build_results` ของ base ตรง ๆ ไม่ต้องเขียน loop แปลงเอง
- Default `model_path` resolve แบบ package-relative ผ่าน `Path(__file__).parent.parent / "assets/models/bast_diplom.onnx"` แทนการใช้ relative path ที่ขึ้นกับ cwd — กันปัญหาเดิมแบบเดียวกับที่เคยเจอใน hardcoded path ของ `ONNXDetection`
- เปิด `conf` / `iou` / `device` เป็น ctor param ให้ปรับได้ และตั้ง `verbose=False` ตอน inference เพื่อไม่ให้ ultralytics spam log ตอน batch

**ไฟล์ที่แก้ไข:**
- `manga_translator/detection/yolo_detection.py` — แทนที่ stub ด้วยคลาส `YoloDetection(DetectorBase)` ที่ใช้งานได้จริง

---

## [2026-06-07 12:49] เพิ่ม DetectorBase รวม contract ของ detection backends

**ประเภท:** refactor

**รายละเอียด:**
- แทนที่ `Detector` placeholder เดิม (inherit ABC ไม่ครบและ logic ไม่สมบูรณ์) ด้วย `DetectorBase` เพื่อกำหนด contract `.detect(image) -> list[DetectionResult]` ที่ทั้ง `ONNXDetection` และ `YoloDetection` (ผ่าน `ultralytics`) จะ subclass ได้ ทำให้ `pipeline.py` สลับ backend ได้โดยไม่ต้องเปลี่ยน calling code
- เพิ่ม helper `build_result` / `build_results` ที่ออกแบบให้รับ output แบบ ultralytics โดยตรง (xyxy boxes + class ids + polygons จาก `result.masks.xy`) แล้วแปลงเป็น `DetectionResult` ตาม schema รวมฟิลด์ `segmentation` ที่เพิ่งเพิ่มเข้ามา — subclass ของ YOLO จึงเขียนสั้นได้
- ยังไม่แก้ `onnx_detection.py` ตามที่ผู้ใช้กำกับ จะ retrofit ทีหลังเมื่อพร้อมย้ายมาใช้ `DetectorBase`

**ไฟล์ที่แก้ไข:**
- `manga_translator/detection/detector.py` — แทนที่ `Detector` placeholder ด้วย `DetectorBase` + helper สำหรับ ultralytics output

---

## [2026-06-07] เริ่มย้ายไป YOLO (ultralytics) สำหรับ detection พร้อม segmentation และเลิกใช้โมดูลเก่า

**ประเภท:** ปรับโครงสร้าง (refactor) / เตรียม feature

**รายละเอียด:**
- เพิ่มไฟล์ `yolo_detection.py` เป็นจุดเริ่มต้นสำหรับ detection ด้วย `ultralytics` (YOLOv8-seg) แทน ONNX ตามแผนแก้ปัญหา OCR overlap ข้ามฟองใน report.md
- เพิ่มฟิลด์ `segmentation` (list ของ polygon) ใน `DetectionResult` เพื่อรองรับผลลัพธ์แบบ segmentation จาก YOLO
- ทำเครื่องหมายโมดูลเดิมที่เลิกใช้แล้วเป็น "No Longer Used": `onnx_detection.py`, `bubble_ocr_matcher.py`, `easyocr_engine.py`
- แก้ path ที่ hardcode เป็น Linux absolute path ให้เป็น relative path ที่ใช้งานได้บน Windows: font ใน `renderer.py` และ default `model_path` ของ `ONNXDetection`

**ไฟล์ที่แก้ไข:**
- `manga_translator/detection/yolo_detection.py` — (ไฟล์ใหม่) จุดเริ่มต้น YOLO/ultralytics detection
- `manga_translator/schemas/interface.py` — เพิ่มฟิลด์ `segmentation` ใน `DetectionResult`
- `manga_translator/detection/onnx_detection.py` — มาร์คเลิกใช้ + แก้ default `model_path`
- `manga_translator/ocr/bubble_ocr_matcher.py` — มาร์คเลิกใช้
- `manga_translator/ocr/easyocr_engine.py` — มาร์คเลิกใช้
- `manga_translator/rendering/renderer.py` — แก้ font path เป็น relative path
- `CHANGELOG.md` — เพิ่มรายการนี้

---

## [2026-05-28 05:23] เพิ่มเอกสาร report.md อธิบายการทำงานของ pipeline และวิเคราะห์ความเสี่ยง

**ประเภท:** เพิ่ม feature (เอกสาร)

**รายละเอียด:**
- สร้างรายงานสรุปการทำงานของระบบแปลมังงะทั้งหมด เพื่อใช้เป็นเอกสารอ้างอิงของโปรเจกต์
- ครอบคลุม 3 ส่วน: (1) ภาพรวม pipeline 5 สเตจ (detection → OCR → translation → inpainting → rendering), (2) รายละเอียดระดับ source code พร้อมตัวอย่างของแต่ละสเตจ, (3) การวิเคราะห์ปัญหา/ความเสี่ยงจัดลำดับเป็น 4 tier พร้อมแนวทางแก้
- รวมการวิเคราะห์ปัญหาหลัก OCR overlap ข้ามฟอง และแผนแก้ด้วย YOLOv8-seg ที่หารือกันไว้
- เป็นเอกสารอย่างเดียว ยังไม่มีการแก้ไข logic ของระบบ

**ไฟล์ที่แก้ไข:**
- `report.md` — (ไฟล์ใหม่) รายงานการทำงานของ pipeline และการวิเคราะห์ความเสี่ยง
- `CHANGELOG.md` — (ไฟล์ใหม่) สร้างไฟล์บันทึกการเปลี่ยนแปลงจาก template

---
