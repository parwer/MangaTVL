# MangaTVL — Reference-free Evaluation

วัดผลสเตจ **3 (Translation) / 4 (Inpainting) / 5 (Rendering)** บนภาพมังงะดิบ
โดย **ไม่ต้องมี ground-truth / คำแปลอ้างอิง / ภาพ clean** — ทุก metric ดึงจาก
output กลางของ pipeline เอง หรือจาก **segmentation polygon** ของ bubble

> สเตจ 1 (Detection) / 2 (OCR) ต้องใช้ ground-truth ถึงวัดแบบมาตรฐานได้ (mAP / CER)
> จึงยังไม่รวมที่นี่ — ดู report.md ส่วนการวัดผล

## รันยังไง

```powershell
# จาก D:\WorkSpace\MangaTVL
# วัดเต็ม (สเตจ 3 ต้องมี API key ของ provider ใน .env)
..\MangaTVL_ENV\python.exe -m eval.run_eval --dir eval/scraped --limit 10

# วัดเฉพาะ 4/5 โดยไม่เสียค่า API (ข้ามการแปล ใช้ข้อความ OCR แทน)
..\MangaTVL_ENV\python.exe -m eval.run_eval --dir eval/scraped --limit 10 --fake-translate

# เพิ่ม LLM-as-judge วัดคุณภาพการแปลเชิงความหมาย (เสีย API เพิ่ม 1 call/หน้า)
..\MangaTVL_ENV\python.exe -m eval.run_eval --dir eval/scraped --limit 10 --judge
# ใช้ judge คนละรุ่น (เลี่ยง self-bias):
..\MangaTVL_ENV\python.exe -m eval.run_eval --dir eval/scraped --limit 10 --judge --judge-model openai/gpt-4o-mini

# เก็บภาพ intermediate ไว้ตรวจด้วยตา (ต้นฉบับ/หลัง inpaint/หลัง render/text-mask/overlay)
..\MangaTVL_ENV\python.exe -m eval.run_eval --dir eval/scraped --limit 5 --save-images
```

### เทียบ text-only vs image-process (สเตจ 3)
`process_image` มีผลเฉพาะสเตจ Translation — โหมดนี้แปลแต่ละหน้า **2 รอบ** (มีรูป/ไม่มีรูป)
แล้ว judge **ทั้งคู่โดยแนบ original + rendered ของแต่ละ variant** (self-bias ของ judge คงที่ → Δ เทียบได้):
```powershell
..\MangaTVL_ENV\python.exe -m eval.compare_eval --dir eval/scraped
```
ผลที่ `eval/results/<ts>-compare/`: `compare.md` (ตาราง Δ คุณภาพ + token/cost),
`pairs.jsonl` (จับคู่ต่อ bubble: source / text-only / image / Δ), `raw/<id>.json` (raw req/resp/usage ทั้ง 4 call).
รันจริง 34 ภาพ (255 bubbles): image-process ให้ adequacy/fluency สูงกว่าเล็กน้อยแต่ใช้ token ~2 เท่า

## โครงสร้างผลลัพธ์ (เป็น 1 run ต่อการรัน — ตรวจย้อนกลับได้)

ผลเขียนที่ `eval/results/<run_id>/` (`run_id` = timestamp + slug ของ dataset):

| ไฟล์ | เนื้อหา |
|------|---------|
| `config.json` | param ทั้งหมด + **JSON schema** ของ DetectionResult/OCRResult/TranslationResult + git commit + รายชื่อไฟล์ |
| `manifest.json` | ต่อรูป: `image_id`, status (ok/skipped/error), จำนวน bubble, timing |
| `corpus.json` | metric รวมระดับ corpus (micro-average) |
| `summary.md` | ตารางสรุปไทย |
| `bubbles.jsonl` | **1 บรรทัด/bubble** ข้ามทุกรูป — ตารางหลักไว้ audit (ดูล่าง) |
| `images/<id>.json` | เรคคอร์ดเต็มต่อรูป: param + **raw ทุกสเตจ** + per-bubble metric |
| `images/<id>/*.png` | เฉพาะ `--save-images` |

### raw log / การ audit
ทุก metric ย้อนกลับไปดู input/output จริงได้:
- **`bubbles.jsonl`** แต่ละแถวมี `run_id`/`image_id`/`bubble_index` (provenance) + `ocr_text`,
  `source_text`, `translated_text` + metric ของทุกสเตจในแถวเดียว → สาวกลับได้ว่าเลขมาจาก bubble ไหน
- **`images/<id>.json` → `translation_raw`** เก็บ raw ของ LLM: `system_prompt`, `user_input`,
  `raw_response` (string ดิบจากโมเดล), `usage` (token + cost), `parsed` (map `text_no→คำแปล`)
  → ตรวจได้ว่า `translated_text` = ผล parse จาก `raw_response` จริง
- **`judge_raw`** เก็บ req/resp ของ judge เช่นเดียวกัน
- raw LLM เก็บ **ครั้งเดียวต่อรูป** (1 call/หน้า) แต่ละ bubble อ้างด้วย `text_no`

> raw มาจาก capture hook ใน production `translator.translate(..., capture=dict)` — code path
> เดียวกับ production จริง (ไม่ใช่ re-implement) ค่า default `capture=None` = production ไม่กระทบ

## Metric ที่วัด

### สเตจ 3 — Translation (reference-free)
จับ failure mode ที่พบบ่อยได้ แต่ **ไม่ตัดสินความหมาย** (อันนั้นต้องมี reference หรือ LLM-judge)

| metric | ความหมาย | ทิศที่ดี |
|--------|----------|---------|
| `fallback_rate` | translator ตกกลับไปใช้ข้อความเดิม (API fail / ไม่มี key) | ต่ำ |
| `untranslated_rate` | output ไม่ใช่ภาษาเป้าหมาย (model echo/refuse) | ต่ำ |
| `mean_target_script_ratio` | สัดส่วนอักษรภาษาเป้าหมาย (ไทย) ใน output | สูง |
| `mean_len_ratio` | ความยาวแปล / ต้นฉบับ | ~1 |

#### LLM-as-judge (opt-in `--judge`)
call ที่ 2 แยกต่างหาก *หลัง* แปลเสร็จ — เอาคู่ `source → translated` ให้ LLM ให้คะแนน
(ไม่ได้แปลใหม่) prompt เป็น rubric แยก ไม่ใช่ prompt แปล

judge แนบ **2 รูปของหน้าเดียวกัน** เป็นบริบท: IMAGE 1 = หน้า **ต้นฉบับ** (ข้อความในฟอง), IMAGE 2 = หน้า **ที่ render แล้ว** (คำแปลวาดกลับลงฟอง) — ช่วยให้คะแนน adequacy/fluency ตรงบริบทจริงขึ้น (ยังให้คะแนนแค่ 2 แกนนี้ ไม่ตัดสิน typesetting)
- **ต้องใช้ vision model** (default `google/gemini-3-flash-preview` รองรับ) — ถ้า `--judge-model` เป็น text-only (เช่น deepseek/qwen-text) ใส่ `--judge-no-images` เพื่อ judge แบบข้อความล้วน
- รูปถูก resize (`resize_max`) + `detail:low` เพื่อคุม token

| metric | ความหมาย | ทิศที่ดี |
|--------|----------|---------|
| `mean_adequacy` | ความหมายตรงต้นฉบับ (1–5) | สูง |
| `mean_fluency` | ภาษาไทยเป็นธรรมชาติ (1–5) | สูง |
| `low_adequacy_rate` | สัดส่วน bubble ที่ adequacy ≤ 2 | ต่ำ |

> ⚠️ **self-bias:** ถ้า `--judge-model` = รุ่นเดียวกับ translator (default) โมเดลมักให้คะแนน
> งานตัวเองสูงเกินจริง (มักได้ 5/5 รวด) — อ่านเป็นสัญญาณ *เชิงเปรียบเทียบ* เท่านั้น
> ใช้ `--judge-model` รุ่นอื่นเพื่อให้คะแนนน่าเชื่อถือ · summary จะ flag `self_judged` ไว้

### สเตจ 4 — Inpainting (วัดภายใน polygon)
erode polygon เข้าเล็กน้อยเพื่อตัดเส้นขอบ bubble แล้ววัดความสะอาดของพื้นในฟอง

| metric | ความหมาย | ทิศที่ดี |
|--------|----------|---------|
| `mean_interior_std_after` | std ของ grayscale ในฟองหลังลบ | ต่ำ (เรียบ) |
| `mean_std_reduction` | (std_before − std_after)/std_before | สูง |
| `mean_ink_residual_after` | สัดส่วน pixel เข้ม (text) ที่ค้าง | ต่ำ |
| `mean_ink_removed` | สัดส่วนหมึกที่ลบออกได้จริง | สูง |

> หมายเหตุ: ink-residual สมมติ text เข้มบนพื้นอ่อน — ฟองพื้นเข้ม/ภาพรกให้ดู
> `std_reduction` แทน (ไม่ขึ้นกับสีพื้น) · bubble ที่ไม่มี polygon ถูกข้าม (รายงานเป็น coverage)

### สเตจ 5 — Rendering
**spill/fill** วัดจาก diff ภาพ (หลัง render − หลัง inpaint = pixel ตัวอักษร) เทียบกับ polygon ·
**fit** เรียกตรรกะ fit จริงของ renderer ซ้ำต่อ bubble

| metric | ความหมาย | ทิศที่ดี |
|--------|----------|---------|
| `mean_spill_ratio` | สัดส่วน text ที่ล้น *นอก* รูปทรงฟอง | ต่ำ |
| `mean_fill_ratio` | text ใช้พื้นที่ใน polygon แค่ไหน | กลางๆ (สูงไป=แน่น) |
| `truncated_rate` | ถูกตัดด้วย `…` ที่ font ต่ำสุด (ไม่พอที่จริง) | ต่ำ |
| `mean_font_size` / `mean_lines` | ขนาดฟอนต์ / บรรทัดเฉลี่ยต่อฟอง | — |

## โครงสร้าง
- `harness.py` — รัน pipeline 1 หน้า เก็บ intermediate ทุกสเตจ (`Capture`) แบบไม่แตะ `pipeline.py`
- `metrics.py` — ฟังก์ชันคำนวณ metric ต่อสเตจ
- `run_eval.py` — driver: วน batch, รวมผล (micro-average), เขียน JSON + summary.md
