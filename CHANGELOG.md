# Changelog

บันทึกการเปลี่ยนแปลงของโปรเจกต์นี้ทั้งหมด รายการล่าสุดอยู่ด้านบนสุด

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
