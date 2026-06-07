# Changelog

บันทึกการเปลี่ยนแปลงของโปรเจกต์นี้ทั้งหมด รายการล่าสุดอยู่ด้านบนสุด

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
