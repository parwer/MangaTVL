# MangaTVL — เว็บไซต์อธิบายโปรเจกต์

เว็บไซต์ static (HTML/CSS/JS ล้วน ไม่ต้อง build/install) ที่อธิบายโปรเจกต์ MangaTVL
แบ่งเป็นหมวดหมู่ตาม pipeline พร้อม **visualization** ของแต่ละ approach

## เปิดดู

เปิด `index.html` ด้วย browser ได้เลย (double-click) — ทำงานแบบ `file://` ได้
หรือเสิร์ฟด้วย static server ก็ได้:

```powershell
# จาก D:\WorkSpace\MangaTVL\website
..\..\MangaTVL_ENV\python.exe -m http.server 5500
# เปิด http://localhost:5500
```

## หมวดหมู่ (หน้า)

| ไฟล์ | หมวด | visualization |
|------|------|---------------|
| `index.html` | ภาพรวม & ที่มาที่ไป | แผนผัง pipeline 5 สเตจ (กดไปแต่ละสเตจได้) |
| `detection.html` | Stage 1 · Detection | toggle box vs polygon (เห็นการกินกันของ bbox) |
| `ocr.html` | Stage 2 · OCR | step-through crop → clean_poly → OCR |
| `translation.html` | Stage 3 · Translation | การ map ด้วย `text_no` (round-trip) |
| `inpainting.html` | Stage 4 · Inpainting | slider หด polygon + วัด std (safe-fill) |
| `rendering.html` | Stage 5 · Rendering | toggle bbox / LIR / ข้อความ |
| `architecture.html` | สถาปัตยกรรม & API | schema flow, endpoints, config |
| `eval.html` | **Evaluation** | before/after slider, 4-stage gallery, text-only vs image-process, metric bars |
| `roadmap.html` | ปัญหา & อนาคต | ปัญหาที่แก้แล้ว/ค้าง + สรุป eval |

ภาพตัวอย่างของหน้า Evaluation คัดลอกมาจาก
`eval/results/20260626-140148-compare/images/...` มาไว้ที่ `assets/eval/page009/`
และ `assets/eval/page013/` (original / overlay / inpainted / rendered ทั้ง 2 variant)

## ที่มาของเนื้อหา

อ้างอิงและตรวจสอบจาก `report.md`, `README.md`, `CHANGELOG.md` และอ่านไฟล์ source จริง
(`pipeline.py`, `schemas/interface.py`, `translators/utils/prompt.py` ฯลฯ) เพื่อความถูกต้อง

## โครงสร้าง

```
website/
├── index.html
├── detection.html  ocr.html  translation.html
├── inpainting.html rendering.html
├── architecture.html  roadmap.html
└── assets/
    ├── css/style.css   # ธีม + component ทั้งหมด
    └── js/site.js      # mobile nav + stepper helper
```

> หมวด **Evaluation** ตั้งใจเว้นเป็น placeholder ไว้ (ดู `roadmap.html`) จะเติมรายละเอียด
> + visualization เมื่อระบบ eval พร้อม
