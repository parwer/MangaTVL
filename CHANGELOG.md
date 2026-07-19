# Changelog

บันทึกการเปลี่ยนแปลงของโปรเจกต์นี้ทั้งหมด รายการล่าสุดอยู่ด้านบนสุด

---

## [2026-07-02 22:05] tempomunkey ส่ง custom instruction (ต่อยอด guidelines) ต่อ request

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- ให้ผู้ใช้ส่ง "custom instruction" (เช่น คงคำ honorific, โทนสุภาพ, แปล SFX ตรงตัว) ไปมีผลกับการแปลต่อ request — ต่อยอดจาก `DEFAULT_GUIDELINE_PROMPT`
- `translator.py`: `translate(..., custom_instruction=None)` — เพิ่ม `_merge_guidelines()` เอา base guidelines (ของ translator หรือ default) + custom instruction (มาร์กเป็น highest-priority) ; `_system_prompt(from_lang, to_lang, custom_instruction)` เปลี่ยน cache key เป็น 3-tuple `(from_lang, to_lang, custom_instruction)` → prompt ต่าง instruction ไม่ปนกัน (build_system_prompt เดิม cache ตามภาษาเท่านั้น)
- thread ผ่าน `pipeline` (run/run_batch/run_batch_stream/_run_on_image) → `main.py` field `custom_instruction` ใน TranslateRequest (ทั้ง `/translate/` + `/translate/stream/`)
- `tempomunkey.user.js`: เพิ่ม `<textarea>` "custom instruction" ใน Settings, persist `mtvl_custom_instruction`, ส่งใน body เมื่อกรอก
- backward-compat: ไม่ส่ง → guidelines เดิม (prompt เท่าเดิม)
- verify: `pytest` 78 passed (เพิ่มเคส custom_instruction เปลี่ยน prompt+cache key; อัปเดตเคส cache key เดิมเป็น 3-tuple) + `node --check` + `py_compile`

**ไฟล์ที่แก้ไข:**
- `manga_translator/translators/translator.py` — `_merge_guidelines` + `custom_instruction` ใน translate/_system_prompt
- `pipeline.py`, `main.py` — thread `custom_instruction`
- `tempomunkey.user.js` — textarea + persist + ส่ง
- `tests/test_translator_mapping.py` — เคสใหม่ + อัปเดต cache-key assertion
- `README.md` — doc field + userscript

---

## [2026-07-02 22:04] judge (eval) เห็น 2 รูป (original + rendered) + แก้บั๊ก prompt .format

**ประเภท:** เพิ่ม feature / แก้ bug

**รายละเอียด:**
- ให้ LLM-as-judge เห็น **2 รูปหน้าเดียวกัน**: ORIGINAL (ข้อความต้นฉบับในฟอง) + RENDERED (คำแปลที่วาดกลับ) เป็นบริบท → คะแนน adequacy/fluency ตรงบริบทจริงขึ้น (คง 2 แกนเดิม ไม่เพิ่ม typesetting → ไม่แตะ metrics/schema)
- `judge.py`: `_call(..., images=None)` รับ list `(label, b64)` แนบหลายรูป (openrouter: text+image_url parts, gemini: Part.from_text+from_bytes); `run_judge(..., original_image, rendered_image)` แทน `image` เดียว; prompt อธิบาย 2 รูป + ย้ำห้ามให้คะแนน layout
- **แก้บั๊กเดิม**: `JUDGE_SYSTEM` มีตัวอย่าง JSON `{"text_no": 0,...}` วงเล็บเดี่ยว → `.format()` ตีความเป็น field → `KeyError` ทุกครั้งที่เรียก judge; escape เป็น `{{ }}`
- `run_eval.py` / `compare_eval.py`: ส่ง original+rendered (resize + detail:low คุม token) เข้า judge; `compare_eval` **ย้าย judge มาหลัง render** (live) เพื่อมีรูป rendered ต่อ variant (reuse path โหลด judge จาก cache เหมือนเดิม); เพิ่ม flag `--judge-no-images` สำหรับ judge model แบบ text-only
- verify: `pytest tests/test_judge.py` (FakeClient) — 2 image parts + label + parse ถูก, ไม่มีรูป=string content, 1 รูป=1 part; suite รวม 78 passed

**ไฟล์ที่แก้ไข:**
- `eval/judge.py` — multi-image `_call`, `run_judge` original/rendered, prompt 2 รูป, แก้ `.format` brace bug
- `eval/run_eval.py`, `eval/compare_eval.py` — ส่ง 2 รูป + `--judge-no-images` (+ compare: render-before-judge)
- `tests/test_judge.py` (ไฟล์ใหม่) — FakeClient assert 2 labeled images + parsing
- `eval/README.md` — doc judge 2 รูป + vision-model note

---

## [2026-06-26 14:17] โหมดเทียบ: เพิ่มคะแนนสเตจ 4/5 + เซฟรูป intermediate (log สมบูรณ์)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- `eval/compare_eval.py` เพิ่มการวัด **สเตจ 4 (Inpainting) + 5 (Rendering)** เข้า compare.md (เดิมมีแค่ translation+judge)
- **Inpaint วัดครั้งเดียวต่อหน้า** เพราะไม่ขึ้นกับคำแปล (`InpainterBase.inpaint` ใช้แค่ ocr box/polygon, `parse_inputs` ทิ้ง translated_text) → รายงานเป็นคอลัมน์เดียว "เท่ากันทั้ง 2 variant"
- **Render วัดต่อ variant** เพราะข้อความต่างความยาว → เทียบ spill/fill/truncate/font/lines เป็น Δ
- เซฟรูป intermediate **7 ไฟล์/หน้า** ลง `images/<id>/`: original, overlay (polygon+bbox), inpainted, rendered_text_only, rendered_image, text_mask_text_only, text_mask_image — เปิด default (มี `--no-images` ปิด); reuse `eval/artifacts.py` (`_save`/`polygon_overlay`)
- เพิ่มโหมด `--reuse <run_dir>`: replay คำแปล+judge จาก cache (ไม่เรียก API) คำนวณเฉพาะ inpaint/render — ไว้เติมสเตจ 4/5 ให้ run เดิมโดยไม่เสียค่า LLM
- pairs.jsonl เพิ่ม field `inpaint{std_reduction,ink_removed,...}` + render (spill/fill/font/truncated) ต่อ variant
- **ผลรันสด 34 ภาพ (255 bubbles):** image-process: adequacy +0.13 (4.79→4.92), fluency +0.106, fallback −0.027, token ~2 เท่า; inpaint std_reduction 0.90/ink_removed 0.95; render font เฉลี่ยใกล้กัน (image เล็กกว่าเล็กน้อยเพราะคำแปลยาวกว่า)

**ไฟล์ที่แก้ไข:**
- `eval/compare_eval.py` — เพิ่มสเตจ 4/5, เซฟรูป (`--no-images`), โหมด `--reuse`
- `eval/judge.py` — `_call` คืน `(text, usage)` เพื่อเก็บ token/cost ของ judge (เดิม cost=0)

---

## [2026-06-26 13:35] เพิ่มโหมดเทียบ text-only vs image-process (สเตจ Translation)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เพิ่ม `eval/compare_eval.py` วัดว่าการส่งรูปหน้าให้ตัวแปล (`process_image`) ช่วยคุณภาพการแปลจริงไหม — ต่อหน้า: detect+ocr รอบเดียว แล้วแปล **2 รอบ** (text-only / image-process) + judge ทั้งคู่
- ตามที่ผู้ใช้กำหนด: **judge แนบรูปหน้าเสมอทั้ง 2 variant** → judge ตัดสินจากบริบทภาพจริง และ self-bias คงที่ทั้งสองฝั่งจึงหักล้างกันใน Δ (เทียบได้ แม้ judge ใช้ model เดียวกับ translator)
- ให้ `eval/judge.py` แนบรูปได้ (`_call`/`run_judge` รับ `image=`): openai-compatible ส่ง multimodal `image_url`, gemini ส่ง `Part.from_bytes` — เลียนแบบ path ของ translator; เก็บ `image_attached` + `usage` (token/cost) ใน raw
- ผลที่ `eval/results/<ts>-compare/`: `compare.md` (ตาราง metric|text-only|image|Δ + token/cost), `pairs.jsonl` (จับคู่ต่อ bubble สาวกลับได้), `compare.json`, `raw/<id>.json` (raw req/resp/usage ของ translate+judge ทั้ง 4 call ต่อรูป)
- reuse ของเดิมทั้งหมด (translator capture hook, translation_metrics, run_judge, aggregate) ไม่แตะ run_eval/harness/metrics/translator
- **ผลรันจริง 34 ภาพ (255 bubbles, 31 ok/3 skip):** image-process ดีกว่าเล็กน้อยทุก metric — adequacy +0.117 (4.77→4.89), fluency +0.071, fallback −0.019 — แต่ใช้ token ~2 เท่า (993→2092/หน้า) cost แปล $0.034→$0.053; พบ insight: หน้าเลขหน้า "6" ฝั่ง image ตีความเป็น "ตอนที่ 29" จากบริบทภาพ (over-interpret)

**ไฟล์ที่แก้ไข:**
- `eval/compare_eval.py` (ไฟล์ใหม่) — driver เทียบ 2 variant + เขียน pairs/compare/raw
- `eval/judge.py` — `_call`/`run_judge` รองรับ `image=` (multimodal) + คืน/เก็บ `usage`
- `eval/README.md` — เพิ่มหัวข้อโหมดเทียบ

---

## [2026-06-26 13:02] รื้อ eval เป็น batch + raw-data logging ที่ตรวจสอบย้อนกลับได้

**ประเภท:** refactor + เพิ่ม feature

**รายละเอียด:**
- รื้อ eval ให้รันทั้ง dataset เป็น **1 run** ที่ตรวจย้อนกลับได้ (เดิมเป็น for-loop เก็บ metric สรุปอย่างเดียว สาวกลับไม่ได้)
- **capture hook ใน production translator** (backward-compatible 100%): `translate(..., capture: dict|None=None)` + `_translate(..., capture=None)` — ถ้าส่ง dict จะเก็บ `system_prompt`, `user_input`, `raw_response` (string ดิบ), `usage` (token+cost), `parsed` (map text_no→คำแปล); ค่า default `None` → production (main.py/pipeline.py) ไม่กระทบ และ **forward capture เฉพาะเมื่อไม่ None** เพื่อให้ `_translate` override ที่ไม่มี param นี้ยังทำงาน (test เดิม 74 ตัวผ่านครบ)
- โครงสร้างผลลัพธ์ใหม่ `eval/results/<run_id>/`: `config.json` (param + JSON schema ของ 3 data contract + git commit + รายชื่อไฟล์), `manifest.json` (status/bubble/timing ต่อรูป), `corpus.json`, `summary.md`, **`bubbles.jsonl`** (1 บรรทัด/bubble มี provenance + raw text + metric ทุกสเตจ — ตาราง audit หลัก), **`images/<id>.json`** (raw ทุกสเตจต่อรูป รวม raw LLM req/resp/token + judge raw)
- `Capture` เก็บ `image_id` (จาก path relative, sanitize), `params` snapshot (detector conf/iou, ocr engine, inpainter/renderer), `translation_raw`/`judge_raw`, + `to_record()` serialize JSON-safe
- ทุก per-bubble metric ใส่ `bubble_index` เป็น provenance (aggregate ข้าม field นี้)
- flag `--save-images`: เซฟ original/inpainted/rendered/text_mask/polygon-overlay ต่อรูป (default ปิด)
- ตรวจแล้ว: audit `target_script_ratio` ด้วยมือตรงกับ log, `parsed[0]==translated_text`, real run เก็บ token+cost ($0.0019/หน้า) ครบ

**ไฟล์ที่แก้ไข:**
- `manga_translator/translators/translator.py` — capture hook ใน `translate`/`_translate` + helper `_usage_to_dict`
- `manga_translator/translators/gemini.py` — capture ใน `_translate` (usage_metadata)
- `eval/harness.py` — `Capture` ขยาย (image_id/params/raw/to_record) + ส่ง capture sink
- `eval/run_eval.py` — เขียน driver/output layout ใหม่ (run_id, manifest, bubbles.jsonl, images/<id>.json, config+schema)
- `eval/metrics.py` — เติม `bubble_index` ทุก per-record
- `eval/judge.py` — คืน `raw` (req/resp/usage) + `bubble_index`
- `eval/artifacts.py` (ไฟล์ใหม่) — เซฟภาพ intermediate + polygon overlay
- `eval/README.md` — อธิบาย layout ใหม่ + raw log + การ audit

---

## [2026-06-26 12:35] เพิ่ม LLM-as-judge (opt-in) วัดคุณภาพการแปลเชิงความหมาย

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เพิ่ม flag `--judge` ใน eval: เป็น API call **ที่ 2 แยกต่างหาก** รัน *หลัง* แปลเสร็จ — เอาคู่ `source → translated` ที่ translator ผลิตแล้วส่งให้ LLM ให้คะแนน ไม่ได้แปลใหม่
- prompt ของ judge เป็น **rubric แยก** (ไม่ใช่ SYSTEM_PROMPT ของการแปล): ให้คะแนน `adequacy` (ความหมายตรง 1–5) + `fluency` (ภาษาเป้าหมายเป็นธรรมชาติ 1–5) คืน JSON array map ด้วย `text_no`
- เพิ่ม metric `mean_adequacy`, `mean_fluency`, `low_adequacy_rate` เข้า summary; bubble ที่ judge ไม่คืนคะแนนถูกข้าม (ไม่ลงโทษ)
- รองรับ `--judge-provider` / `--judge-model` แยกจาก translator; dispatch ได้ทั้ง openrouter/groq (chat.completions) และ gemini (genai)
- **self-bias guard:** ถ้า judge ใช้ model เดียวกับ translator → set flag `self_judged=true` และ summary.md ขึ้น ⚠️ เตือนว่าคะแนนเอนสูง (ทดสอบจริงได้ 5.0/5.0 รวด = สะท้อน self-preference bias ตามคาด) — ปัจจุบัน default ใช้ model เดียวกันไปก่อนตามที่ตั้งใจ

**ไฟล์ที่แก้ไข:**
- `eval/judge.py` (ไฟล์ใหม่) — rubric prompt + dispatch ต่อ provider + parse/aggregate คะแนน
- `eval/run_eval.py` — เพิ่ม flag `--judge`/`--judge-provider`/`--judge-model`, เรียก judge ต่อหน้า, รวมผล + section ใน summary.md
- `eval/README.md` — เพิ่มหัวข้อ LLM-as-judge + คำเตือน self-bias + ตัวอย่างคำสั่ง

---

## [2026-06-26 12:21] เพิ่มชุดวัดผล reference-free (eval/) สำหรับสเตจ 3/4/5

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- วัดผล Translation / Inpainting / Rendering บนภาพมังงะดิบโดย **ไม่ต้องมี ground-truth / คำแปลอ้างอิง / ภาพ clean** — ทุก metric ดึงจาก output กลางของ pipeline หรือจาก segmentation polygon ของ bubble
- **สเตจ 3 (Translation):** `fallback_rate` (ตกกลับใช้ข้อความเดิม), `target_script_ratio` (สัดส่วนอักษรเป้าหมาย), `untranslated_rate`, `len_ratio` — จับ failure mode (API fail / model echo) โดยไม่ตัดสินความหมาย
- **สเตจ 4 (Inpainting):** วัดในขอบเขต polygon ที่ erode เข้า — `interior_std` (ก่อน/หลัง + reduction) และ `ink_residual` (หมึกค้าง) เพื่อเช็คว่าลบ text ในฟองสะอาดแค่ไหน robust กับสีพื้นด้วย std_reduction
- **สเตจ 5 (Rendering):** `spill_ratio`/`fill_ratio` วัดจาก diff ภาพ (หลัง render − หลัง inpaint = pixel ตัวอักษร) เทียบ polygon ว่าข้อความล้นนอกฟองไหม + `truncate_rate`/`font_size` โดยเรียกตรรกะ fit จริงของ `TextRenderer` ซ้ำ
- harness เลียนแบบ `Pipeline._run_on_image` แบบไม่แตะ `pipeline.py` (ข้าม upscale เพื่อให้ภาพ render ขนาดตรงกับ polygon), รองรับ `--fake-translate` วัด 4/5 โดยไม่เสียค่า API
- รวมผลแบบ micro-average ข้ามหน้า เขียน `metrics.json` (รายละเอียดต่อ bubble) + `summary.md` (ตารางไทย); validate จริงด้วย openrouter (`google/gemini-3-flash-preview`): fallback 0.05, อักษรไทย 90%, ink_removed 0.97, spill 0.0
- **แก้บั๊ก:** `_ensure_provider_key` เช็ค env ก่อน `load_dotenv()` ของ pipeline ทำงาน → เห็น key ว่าง → ตั้ง `eval-dummy` แล้ว `load_dotenv` ไม่ override → 401 ทุก call; แก้โดย `load_dotenv()` ที่ต้น `run_eval.py`

**ไฟล์ที่แก้ไข:**
- `eval/harness.py` (ไฟล์ใหม่) — รัน pipeline 1 หน้า เก็บ intermediate ทุกสเตจเป็น `Capture` แบบไม่แตะ pipeline
- `eval/metrics.py` (ไฟล์ใหม่) — ฟังก์ชัน metric ต่อสเตจ (translation/inpainting/rendering spill+fit)
- `eval/run_eval.py` (ไฟล์ใหม่) — CLI driver: วน batch, รวมผล, เขียน JSON + summary.md
- `eval/README.md` (ไฟล์ใหม่) — อธิบายนิยาม metric และวิธีรัน
- `eval/__init__.py` (ไฟล์ใหม่) — docstring ของแพ็กเกจ
- `.gitignore` — เพิ่ม `eval/results/` (output ที่ generate)

---

## [2026-06-26 11:44] แก้ scrape_images.py เก็บรูป lazy-load ไม่ครบ (ได้ 6 จาก 32)

**ประเภท:** แก้ bug

**รายละเอียด:**
- อาการ: roliascan.com (Grand Blue ch29) มี 32 หน้า แต่ scrape ได้แค่ 6
- debug จากหน้าจริง: หน้า 022–032 เป็น lazy — `src` เป็น data-URI placeholder (naturalWidth=24) ส่วน URL จริงอยู่ใน `data-src` → โดน skip (`data:`) + กรองด้วย min-width; และ `autoscroll` เดิมหยุดเร็วเพราะ placeholder จอง height ไว้ scrollHeight เลยนิ่งตั้งแต่ต้น → โหลดจริงแค่ ~6 หน้าแรก
- แก้: 
  - เพิ่ม **force-load lazy** (`_FORCE_LAZY_JS`): เซ็ต `img.src` จาก `data-src`/`data-lazy-src`/`data-original`/`data-lazy`/`data-url` + `loading=eager` → รูปโหลดจริงแม้อยู่นอก viewport
  - เปลี่ยน `autoscroll` → `load_all_images`: scroll จนถึงล่างจริง + force-load ทุกรอบ, หยุดเมื่อจำนวนรูปที่โหลด (naturalWidth>1) นิ่ง (ไม่ใช่ scrollHeight)
  - `collect_images`: เลือก URL จริงจาก `data-*` ด้วย (ไม่เอา data-URI), เก็บรูปที่วัดขนาดไม่ได้ (w=0 จาก lazy) ไว้แทนที่จะตัดทิ้ง
  - เพิ่ม `--selector` (เช่น `img.comic-image`) เลือกเฉพาะรูปเนื้อหาเพื่อความแม่นยำ
- verify: รันจริง URL เดิม → เจอ **32 รูป** (จาก 6), ดาวน์โหลด WEBP เปิดด้วย PIL ได้

**ไฟล์ที่แก้ไข:**
- `scrape_images.py` — force-load lazy + load_all_images + collect ใช้ data-src + `--selector`
- `README.md` — อธิบาย lazy-load handling + `--selector`

---

## [2026-06-26 11:36] เพิ่ม scrape_images.py — scraper รูปจากเว็บ (Playwright) สำหรับชุด evaluate

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- ทำ standalone scraper ดึงรูปจากหน้าเว็บมาเก็บเป็นชุดทดสอบ translator ("จะเอามา evaluate")
- ใช้ **Playwright (headless chromium)** เพราะเว็บอ่านมังงะเป็น JS/lazy-load + กัน hotlink — requests/bs4 ใช้ไม่ได้
- flow: เปิดหน้าแต่ละ URL → scroll ลงทีละช่วงจน scrollHeight นิ่ง (trigger lazy-load) → เก็บ `<img>` ที่กว้าง ≥ `--min-width` ตาม **DOM order** (มังงะคงลำดับหน้า) dedupe → ดาวน์โหลดผ่าน `context.request` (แชร์ cookie + ใส่ referer) **bypass hotlink** → เซฟ `eval/scraped/<page-slug>/001.jpg, 002.jpg...`
- ตั้ง ext จาก content-type (jpg/png/webp/gif/avif), **ข้าม SVG/non-raster** (ไม่ใช่หน้ามังงะ), retry บน HTTP 429 + `--delay` กัน rate-limit
- CLI: รับ URL หลายอัน หรือ `--urls-file`; options `--out/--min-width/--limit/--scroll-rounds/--delay/--headful/--timeout`
- ติดตั้ง playwright 1.60.0 + chromium ใน MangaTVL_ENV; ใส่เป็น **optional** ใน requirement.txt (ไม่ใช่ dep ของ API runtime)
- verify: `py_compile` + รันจริงบน Wikipedia → โหลด/scroll/เก็บ/ดาวน์โหลดได้, ไฟล์เป็น raster เปิดด้วย PIL ได้ (JPEG/PNG, ext ถูก), SVG ถูกข้าม, 429 retry ทำงาน
- `eval/scraped/` ใส่ใน .gitignore (ไม่ commit รูป scraped)

**ไฟล์ที่แก้ไข:**
- `scrape_images.py` (ไฟล์ใหม่) — Playwright scraper
- `requirement.txt` — comment optional playwright
- `.gitignore` — ignore `eval/scraped/`
- `README.md` — หัวข้อ "Scraping an evaluation set"

---

## [2026-06-09 19:39] แก้ streaming ใน tempomunkey ไม่ทำงาน (รูปเปลี่ยนตอนจบพร้อมกัน)

**ประเภท:** แก้ bug

**รายละเอียด:**
- อาการ: ใน demo รูปไม่ทยอยเปลี่ยน รอจน complete ทั้งหมดแล้วค่อย replace พร้อมกัน
- **พิสูจน์แล้วว่า server stream ถูกต้อง**: รัน `/translate/stream/` จริง (stub yield ทีละ 1 วิ) แล้ว `curl -N` เห็นบรรทัด NDJSON ทยอยมาห่างกัน 1 วิจริง → ปัญหาอยู่ฝั่ง client
- สาเหตุ: Tampermonkey หลาย build **buffer `onprogress`/`responseText` จนจบ** (โดยเฉพาะ request ข้าม origin https→http localhost) → onload ค่อยยิงทีเดียว = เปลี่ยนรูปพร้อมกัน
- แก้ฝั่ง client: ใช้ `responseType: "stream"` อ่าน **ReadableStream แบบ incremental** ใน `onloadstart` (วิธี streaming ที่ถูกต้องของ Tampermonkey ≥5) — reader.read() loop → decode → parse NDJSON ทีละบรรทัด → swap รูปทันที; ถ้า build ไม่รองรับ (ไม่มี getReader) **fallback ไป onprogress (delta จาก accumulated text) แล้ว onload** = ไม่ regress; รวม parser เป็น buffer เดียว (`pushText`/`applyLine`) กันบรรทัดถูกตัดข้าม chunk + flush tail ที่ไม่มี \n; กัน double-apply/double-finish ด้วย `usedStream`/`finished`
- แก้ฝั่ง server เสริม: ใส่ header `X-Accel-Buffering: no` + `Cache-Control: no-cache` ใน StreamingResponse กัน proxy/CDN buffer (เผื่อ deploy หลัง nginx)
- verify: `node --check` + `py_compile` ผ่าน; server streaming ยืนยันด้วย curl -N (บรรทัดห่าง ~1 วิ)

**ไฟล์ที่แก้ไข:**
- `tempomunkey.user.js` — streaming ผ่าน responseType:"stream" + fallback onprogress/onload, unified NDJSON buffer
- `main.py` — header กัน buffering ใน `/translate/stream/`

---

## [2026-06-09 19:16] แก้ Dockerfile.gpu build fail (PEP 668 externally-managed)

**ประเภท:** แก้ bug

**รายละเอียด:**
- GPU build ล้มที่ `pip install -r requirement.txt` ด้วย error PEP 668 "externally-managed-environment" — base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` มาร์ก Python เป็น externally managed เลยบล็อก pip ลง system env
- แก้: ตั้ง `ENV PIP_BREAK_SYSTEM_PACKAGES=1` ใน Dockerfile.gpu (ครอบทั้ง `pip install --upgrade pip` และ `pip install -r requirement.txt`) — เหมาะกับ container เพราะ env เป็นของ image นั้นโดยเฉพาะ
- CPU `Dockerfile` (`python:3.12-slim`) ไม่กระทบ (ไม่ได้มาร์ก externally-managed) จึงไม่แตะ

**ไฟล์ที่แก้ไข:**
- `Dockerfile.gpu` — เพิ่ม `PIP_BREAK_SYSTEM_PACKAGES=1` ใน ENV

---

## [2026-06-09 19:08] เพิ่ม upscale ภาพตอน return (LANCZOS + Real-ESRGAN, เลือกต่อ request)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- ปัญหา: ต้นฉบับ low-res/เบลอ ถึงเพิ่ม font ก็ยังอ่านยาก → เพิ่มขั้น upscale ภาพ final ตอน return
- ไฟล์ใหม่ `rendering/upscale.py` — `upscale_image(image, factor, method, device)`:
  - `lanczos` (default): PIL resize คุณภาพสูง เร็ว ไม่ลง dep — ภาพ (และข้อความที่เราเรนเดอร์) ใหญ่ขึ้น แต่อาร์ตเวิร์กไม่ deblur
  - `realesrgan`: AI super-resolution (โมเดล anime 6B, x4) deblur อาร์ตเวิร์กจริง — **lazy import** realesrgan/basicsr + โมเดลโหลดครั้งแรก, ใช้ tiling กัน OOM, รันบน device เดียวกับ pipeline; ถ้า import/รันไม่ได้ (basicsr ชนกับ torchvision ใหม่) **fallback เป็น LANCZOS อัตโนมัติ** + log
  - factor <=1 = no-op, clamp ที่ 4.0, factor มั่ว = คืนภาพเดิม
- upscale ที่ขั้น final (หลังเรนเดอร์) → ใช้เหมือนกันทั้ง 2 method ไม่ต้องแก้พิกัด detection; ข้อความเราที่ supersample มาแล้วยังคมตอนขยาย
- ปรับ **ต่อ request**: `_run_on_image` รับ `upscale`/`upscaler` → ไหลผ่าน run/run_batch/stream → `main.py` field `upscale`(float)/`upscaler`(str) ใน TranslateRequest (ทั้งสอง endpoint) → `tempomunkey` ช่อง "upscale image" (1-4) + dropdown fast(lanczos)/AI(realesrgan), ส่งเฉพาะเมื่อ >1
- `realesrgan`/`basicsr` ไม่อยู่ใน requirement.txt หลัก (basicsr พังกับ torchvision ใหม่) — ใส่เป็น optional comment; default lanczos ทำงานชัวร์
- verify: `py_compile` + `node --check` + `pytest` 74 passed (เพิ่ม `tests/test_upscale.py` 6 เคส รวม fallback) + import main ไม่ดึง realesrgan ตอน start (lazy ผ่าน) + lanczos 2x ขยายถูก + realesrgan fallback→lanczos จริง

**ไฟล์ที่แก้ไข:**
- `manga_translator/rendering/upscale.py` (ไฟล์ใหม่) — upscale_image (lanczos + realesrgan lazy/fallback)
- `pipeline.py` — เรียก upscale_image ตอน return + thread `upscale`/`upscaler`
- `main.py` — field `upscale`/`upscaler` (ทั้งสอง endpoint)
- `tempomunkey.user.js` — ช่อง upscale factor + method picker
- `requirement.txt` — comment optional realesrgan/basicsr
- `tests/test_upscale.py` (ไฟล์ใหม่) — 6 เคส
- `README.md` — doc upscale/upscaler + userscript

---

## [2026-06-09 18:58] ทำให้ข้อความที่แปลใหญ่ขึ้น + คมขึ้น (text_scale + supersample)

**ประเภท:** เพิ่ม feature / แก้ไข

**รายละเอียด:**
- ปัญหา: renderer auto-fit ฟอนต์ลง LIR (สี่เหลี่ยมในเส้น polygon) ของบับเบิล ซึ่งอนุรักษ์มาก (บับเบิลทรงรี LIR เล็กกว่ากรอบเยอะ) → ไทยที่ข้อความยาวถูกบีบเล็ก อ่านยาก
- **ใหญ่ขึ้น (text_scale):** เพิ่ม `_inflate_box` ขยายพื้นที่ฟิตจาก LIR ออกไปหา bbox ตามสัดส่วน `text_scale` (รอบจุดกึ่งกลาง, cap ที่ det.bbox inset + ขอบภาพ → ไม่ล้นเกินกรอบบับเบิล/ภาพ) แล้ว binary-search หาขนาดฟอนต์ในกล่องที่ใหญ่ขึ้น = ตัวใหญ่ขึ้นโดยยังไม่ overflow; default `1.2`, clamp `[0.5, 2.0]`
- **คมขึ้น (supersample):** วาดข้อความบน overlay RGBA ที่ scale `supersample`× (default 2) แล้ว downscale LANCZOS composite ทับ → ขอบ glyph คมขึ้นที่ขนาดเดิม โดย**ไม่แตะอาร์ตเวิร์ก** (fit search ยังวัดใน original space, วาดจริง ×ss); เพิ่ม `_load_font` lru cache อยู่แล้วช่วยลดต้นทุน
- ปรับได้ **ต่อ request**: `render(..., text_scale=None)` → ไหลผ่าน pipeline (run/run_batch/stream/_run_on_image) → `main.py` field `text_scale` ใน TranslateRequest (ทั้ง `/translate/` + `/translate/stream/`) → `tempomunkey` ช่อง "text size" (number 0.5-2.0) persist `mtvl_text_scale` ส่งเป็น float
- backward-compat: ไม่ส่ง text_scale ใช้ default ของ renderer (1.2); supersample เป็น config ตอนสร้าง renderer (ไม่ใช่ per-request)
- verify: `py_compile` + `node --check` + `pytest` 68 passed + **render smoke test**: บับเบิลทรงรี ไทย text_scale 1.0/1.2/1.6 → ink เพิ่ม 4473/5603/8035 px (ใหญ่ขึ้นจริง), output size คงเดิม (supersample composite กลับถูกต้อง)

**ไฟล์ที่แก้ไข:**
- `manga_translator/rendering/renderer.py` — `text_scale`/`supersample` params, `_inflate_box`, overlay supersample, `_clamp_text_scale`, `_render_single` วาด ×ss
- `pipeline.py` — thread `text_scale`
- `main.py` — field `text_scale` (ทั้งสอง endpoint)
- `tempomunkey.user.js` — ช่อง text size + persist + ส่ง float
- `README.md` — doc text_scale + supersample + userscript text size

---

## [2026-06-09 18:46] เพิ่มฟอนต์มังงะ/คอมิกหลายแบบ + เลือกฟอนต์ได้ต่อ request (และใน tempomunkey)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- โหลดฟอนต์สไตล์มังงะ/คอมิก 11 แบบ (OFL/Apache จาก Google Fonts) ลง `assets/fonts/` ครอบ 4 สคริปต์ — ตรวจ glyph coverage จริงด้วย fontTools cmap:
  - ไทย+Latin: Itim, Mali, Sriracha, Charm (+ THSarabunNew เดิม)
  - Latin คอมิก: Bangers, Comic Neue, Permanent Marker, Patrick Hand
  - ญี่ปุ่น: Yusei Magic / จีน: Zhi Mang Xing / เกาหลี: Nanum Pen
- ทำ registry `assets/fonts.json` (key/file/label/scripts + `defaults` ต่อภาษา) — เพิ่มฟอนต์ใหม่แค่วาง .ttf + เพิ่ม entry
- ไฟล์ใหม่ `rendering/fonts.py`: `list_fonts()` (catalogue ให้ API/client) + `font_path(name, lang, fallback)` resolve ตามลำดับ key→default ของภาษา→latin default→fallback (key มั่วไม่ crash, fallback อัตโนมัติ)
- `utils/lang.py`: เพิ่ม `canonical_name()` (alias→ชื่อภาษา canonical) ใช้ map ภาษา→ฟอนต์ default
- `renderer.py`: `render(..., font=None)` resolve font path ครั้งเดียวต่อ call แล้ว thread เข้า `_render_single/wrap_extraction/_fits_in_box/_truncate_to_fit` (เลิกใช้ `self.font_path` ตรงๆ); เพิ่ม `_load_font` cached (lru) ลดการอ่านไฟล์ซ้ำตอน binary search ขนาดฟอนต์ — ถ้า `font` ไม่ระบุใช้ default ตาม to_lang → CJK/เกาหลีได้ฟอนต์ที่มี glyph ไม่เป็นกล่อง tofu
- `pipeline.py`: thread `font` ผ่าน run/run_batch/run_batch_stream/_run_on_image → render
- `main.py`: เพิ่ม field `font` ใน TranslateRequest (ส่งเข้าทั้ง `/translate/` และ `/translate/stream/`) + endpoint ใหม่ `GET /fonts/` คืน catalogue
- `tempomunkey.user.js`: เพิ่ม dropdown ฟอนต์ใน Settings, fetch `GET /fonts/` ตอนโหลดมา populate (sync กับ fonts.json อัตโนมัติ), persist `mtvl_font`, ส่ง `font` ใน body เมื่อเลือก
- verify: `py_compile` + `node --check` + import main (routes มี `/fonts/`, 12 ฟอนต์) + `pytest` 68 passed (เพิ่ม `tests/test_fonts.py` 7 เคส) + **render smoke test จริง**: ไทย(THSarabun)/อังกฤษ(Bangers)/ญี่ปุ่น(YuseiMagic auto) วาด glyph ออกครบทุกภาษา

**ไฟล์ที่แก้ไข:**
- `manga_translator/assets/fonts/*.ttf` (11 ไฟล์ใหม่) — ฟอนต์มังงะ/คอมิก
- `manga_translator/assets/fonts.json` (ไฟล์ใหม่) — registry + defaults ต่อภาษา
- `manga_translator/rendering/fonts.py` (ไฟล์ใหม่) — list_fonts / font_path resolver
- `manga_translator/rendering/renderer.py` — `font` param + thread font_path + `_load_font` cache
- `manga_translator/utils/lang.py` — เพิ่ม `canonical_name()`
- `pipeline.py` — thread `font`
- `main.py` — field `font` + `GET /fonts/`
- `tempomunkey.user.js` — font picker + fetch /fonts/
- `tests/test_fonts.py` (ไฟล์ใหม่) — 7 เคส
- `README.md` — ตารางฟอนต์ + `/fonts/` + field font + userscript

---

## [2026-06-09 18:29] ขยาย tokenization ของ renderer ให้รองรับหลายภาษา (ตัดบรรทัดตาม to_lang)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เดิม renderer ตัดคำด้วย pythainlp (`engine="newmm"`) ตายตัว + ใช้ trick แทนช่องว่างด้วย "-" → ใช้ได้เฉพาะไทย ภาษาอื่น (จีน/ญี่ปุ่นไม่มีช่องว่าง, อังกฤษ/เกาหลีมีช่องว่าง) ตัดบรรทัดเพี้ยน
- ทำ tokenization แบบ **ขับด้วยข้อมูลจาก languages.json** (field `wrap` ใหม่) เลือก 3 โหมดตาม target language:
  - `pythainlp` (ไทย) — ตัดคำ ไม่ตัดกลางคำ
  - `char` (ญี่ปุ่น/จีน) — ตัดได้ทุกตัวอักษร (CJK ขึ้นบรรทัดแบบนี้อยู่แล้ว ไม่ต้องลง morphological analyzer หนัก)
  - `space` (อังกฤษ/เกาหลี/อื่นๆ default) — ตัดที่ช่องว่าง คำอยู่ครบ
- ไฟล์ใหม่ `rendering/tokenize.py`: `tokenize(text, lang)` + `is_space_delimited(lang)` (import pythainlp แบบ lazy เฉพาะตอนแปลเป็นไทย)
- `utils/lang.py`: เพิ่ม `wrap_mode(name, default="space")` อ่าน field `wrap` (รองรับ alias/case-insensitive ผ่าน `_lookup` เดิม)
- `renderer.py`: เลิก import pythainlp ตรงๆ; `tokenizer` param เป็น optional override `(text, lang)->list` (default None = ใช้ default ตามภาษา); เพิ่ม `lang` default + `render(image, inputs, lang=None)`; แทน trick ช่องว่าง/ขีด ด้วย `_prepare_text`/`_finalize_text` (no-space lang ป้องกันช่องว่างเดิมด้วย sentinel U+F8FF แล้วคืนภายหลัง, space lang คงช่องว่างเป็น word separator)
- `pipeline.py`: ส่ง `lang=to_lang or self.to_lang` เข้า `renderer.render` (ตัดบรรทัดตามภาษาปลายทางต่อ request)
- verify: `py_compile` + `pytest` 61 passed (เพิ่ม `tests/test_tokenize.py` 5 เคส) + ทดสอบ `_prepare_text`/`_finalize_text` round-trip อังกฤษ/ไทย/ญี่ปุ่น/จีน ครบ (คืนช่องว่างเดิมถูกต้องทุกภาษา)

**ไฟล์ที่แก้ไข:**
- `manga_translator/rendering/tokenize.py` (ไฟล์ใหม่) — tokenize/is_space_delimited ตาม wrap mode
- `manga_translator/rendering/renderer.py` — language-aware prepare/finalize + lang param
- `manga_translator/utils/lang.py` — เพิ่ม `wrap_mode()`
- `manga_translator/assets/languages.json` — เพิ่ม field `wrap` (thai/japanese/chinese/chinese_traditional) + อัปเดต `_comment`
- `pipeline.py` — ส่ง to_lang เข้า render
- `tests/test_tokenize.py` (ไฟล์ใหม่) — 5 เคส
- `README.md` — doc รendering line-breaking ตามภาษา

---

## [2026-06-09 18:19] เพิ่ม manga-ocr engine สำหรับภาษาญี่ปุ่น (เลือก OCR engine ต่อภาษา)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- EasyOCR (default) ออกแบบมาสำหรับข้อความแนวนอน แต่มังงะญี่ปุ่นเขียนแนวตั้ง (tategaki) → อ่านเพี้ยน/ลำดับผิด ใช้แปลจริงไม่ได้
- เพิ่ม `MangaOCREngine` (kha-white/manga-ocr) ที่ train เฉพาะญี่ปุ่นในมังงะรวมแนวตั้ง — รับ crop ของบับเบิลเดียวคืน string ตรงๆ เข้า pattern `get_ocr` (crop bbox → `clean_poly` → OCR) เหมือน engine อื่น; ไม่มี box ต่อบรรทัด → ใช้ bubble bbox เป็น box เดียว
- เลือก engine ต่อภาษาจาก field `engine` ใน `languages.json` (japanese → `mangaocr`, อื่นๆ default → `easyocr`); เกาหลี/จีนเป็นแนวนอน EasyOCR พอ
- `pipeline._get_ocr` เลือก class ตาม `ocr_engine_name(lang)` + เปลี่ยน cache key เป็น `(engine, code)` กัน collision; cache engine ต่อภาษาเหมือนเดิม
- **lazy/optional**: `import manga_ocr` อยู่ใน `__init__` ของ engine → โหลดโมเดล (~400MB) เฉพาะตอนมี request ญี่ปุ่นครั้งแรกเท่านั้น (import main ตอน start ไม่ดึง manga_ocr — verify แล้ว)
- `tempomunkey.user.js` เพิ่ม field source language ใน Settings (persist `mtvl_from_lang`, ส่ง `from_lang` ใน body เมื่อกรอก) — จำเป็นเพื่อ trigger ญี่ปุ่น (เดิมส่งแค่ to_lang → server default english)
- verify: `py_compile` + `node --check` + import main (lazy ผ่าน, en→EasyOCREngine) + `pytest` 56 passed (เพิ่ม 3 เคส `ocr_engine_name`) + ยืนยัน `manga-ocr==0.1.14` เป็นเวอร์ชันล่าสุด

**ไฟล์ที่แก้ไข:**
- `manga_translator/ocr/mangaocr_engine.py` (ไฟล์ใหม่) — `MangaOCREngine` lazy import manga_ocr
- `manga_translator/assets/languages.json` — เพิ่ม `engine: "mangaocr"` ให้ japanese + อัปเดต `_comment`
- `manga_translator/utils/lang.py` — refactor `_lookup` + เพิ่ม `ocr_engine_name()`
- `pipeline.py` — `_get_ocr` เลือก engine ตามภาษา, cache key `(engine, code)`
- `tempomunkey.user.js` — Settings เพิ่ม source language + ส่ง from_lang
- `requirement.txt` — เพิ่ม `manga-ocr==0.1.14` (โมเดลโหลด lazy)
- `tests/test_lang.py` — 3 เคส `ocr_engine_name`
- `README.md` — doc OCR engine ต่อภาษา + userscript source language

---

## [2026-06-09 17:58] เพิ่ม streaming response — ทยอยส่งผลลัพธ์ทีละรูป

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เดิม `/translate/` ใช้ `asyncio.gather` รอทั้ง batch เสร็จก่อนแล้วค่อย return ทีเดียว → รูปที่เสร็จก่อนถูกกักรอจนรูปสุดท้ายเสร็จ
- เพิ่ม `Pipeline.run_batch_stream()` — async generator ใช้ `asyncio.as_completed` yield `(index, image)` **ตามลำดับที่เสร็จ** (ไม่ใช่ลำดับ request); รูปที่ fail yield `(index, None)`; ห่อ `try/except` ต่อรูปกัน 1 รูปพังแล้วล้ม stream ทั้งก้อน
- เพิ่ม endpoint `POST /translate/stream/` (body เดิม) ตอบเป็น **NDJSON** (`application/x-ndjson`) ผ่าน `StreamingResponse` — 1 บรรทัด/รูป `{"index", "image": base64|null}` ส่งทันทีที่รูปนั้นเสร็จ; คง `/translate/` เดิมไว้สำหรับ caller แบบ one-shot
- `tempomunkey.user.js` เปลี่ยนไปเรียก `/translate/stream/` แล้วอ่าน NDJSON ใน `onprogress` (parse เฉพาะบรรทัดที่ครบ `\n`, จำ `consumed` offset) → swap รูปในหน้าทันทีทีละรูป + อัปเดต status `done/fail` แบบ progressive; ตอน Interrupt เคลียร์เฉพาะ badge ที่ยัง pending (`…`) คงรูปที่แปลเสร็จแล้วไว้
- verify: `py_compile` pipeline.py+main.py + `node --check` userscript + import main (routes มี `/translate/stream/`, `run_batch_stream` เป็น async-gen) + `pytest` 53 passed

**ไฟล์ที่แก้ไข:**
- `pipeline.py` — เพิ่ม `run_batch_stream()` (as_completed + index + per-image try/except)
- `main.py` — endpoint `/translate/stream/` (StreamingResponse + NDJSON); import StreamingResponse, json
- `tempomunkey.user.js` — consume NDJSON stream ใน onprogress, swap ทีละรูป, abort คงรูปที่เสร็จแล้ว
- `README.md` — เอกสาร endpoint streaming + อัปเดตตาราง userscript

---

## [2026-06-09 17:52] Wire per-language OCR cache เข้า pipeline (OCR ตาม from_lang)

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- ต่อยอด language mapping [17:43] — ให้ OCR เลือกภาษาตาม `from_lang` ต่อ request **โดยไม่โหลด EasyOCR ใหม่ทุกครั้ง** (ตอบคำถามผู้ใช้เรื่องโหลดซ้ำ)
- เดิม `pipeline.py` สร้าง `EasyOCREngine(language="en")` ตัวเดียวตอน init → เปลี่ยนภาษา OCR ต่อ request ไม่ได้
- แก้: เปลี่ยนเป็น **registry ต่อภาษา (lazy + cache)** — `_get_ocr(from_lang)`: ถ้า caller ส่ง `ocr_engine` มา ใช้ตัวนั้นเสมอ (`_ocr_override`); ไม่งั้น `code = ocr_lang_code(from_lang, "easyocr")` แล้ว build `EasyOCREngine(language=code)` cache ใน `self._ocr_engines` → **ภาษาเดิม reuse engine เดิม โหลด model แค่ครั้งแรกต่อภาษา**
- prebuild OCR ของ default `from_lang` ตอน init + คง `self.ocr_engine` (backward-compat); `_run_on_image` ใช้ `self._get_ocr(from_lang).get_ocr(...)`
- ครบ loop: client ส่ง `from_lang` → OCR อ่านภาษานั้น → translate ภาษานั้น
- verify: `py_compile` + import main (prebuild `en`, reuse engine เดิมเมื่อภาษาเดิม=True) + `pytest` 53 passed

**ไฟล์ที่แก้ไข:**
- `pipeline.py` — OCR registry ต่อภาษา (`_ocr_override`/`_ocr_engines`/`_get_ocr`) + ใช้ใน `_run_on_image`; import `ocr_lang_code`

---

## [2026-06-09 17:43] เพิ่ม language mapping (assets) สำหรับ map ภาษา → OCR code

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เพิ่ม `manga_translator/assets/languages.json` — map ชื่อภาษา (lowercase แบบเดียวกับ `from_lang`/`to_lang`) → OCR language code โดยมีทั้ง **easyocr และ paddleocr** (โค้ดต่างกัน เช่น japanese → easyocr `ja` / paddleocr `japan`, korean → `ko`/`korean`, chinese → `ch_sim`/`ch`) + field `aliases` ให้ ISO code/สะกดอื่น (`jp`, `zh-cn`, `eng`, …) resolve เข้า entry เดียวกัน
- ครอบคลุม english/thai/japanese/korean/chinese/chinese_traditional/vietnamese/french/german/spanish/italian/portuguese/russian/indonesian
- เพิ่ม helper `manga_translator/utils/lang.py`: `load_language_map()` (lru_cache, กรอง key `_comment`) + `ocr_lang_code(name, engine="easyocr", default="en")` — resolve canonical/alias แบบ case-insensitive + trim, fallback default เมื่อไม่รู้จัก
- จุดประสงค์: แปลง `from_lang` ต่อ request → OCR code (เตรียม wire เข้า `EasyOCREngine`/`PaddleOCREngine` ที่ตอนนี้ hardcode `language="en"`) — **ยังไม่ wire** asset+loader พร้อมใช้
- เพิ่ม `tests/test_lang.py` 6 เทสต์ (canonical, paddle codes ต่าง, alias, case-insensitive, unknown→default/None, ไม่มี `_comment`); verify JSON valid + py_compile + pytest 6 passed

**ไฟล์ที่แก้ไข:**
- `manga_translator/assets/languages.json` — (ไฟล์ใหม่) language → OCR code map
- `manga_translator/utils/lang.py` — (ไฟล์ใหม่) loader + `ocr_lang_code`
- `tests/test_lang.py` — (ไฟล์ใหม่) เทสต์ helper

---

## [2026-06-09 17:38] ย้าย model/ภาษา/provider/api_key เป็น per-request

**ประเภท:** refactor / เพิ่ม feature

**รายละเอียด:**
- เดิม translator ผูก model/ภาษา/provider+key ที่ `__init__` (สร้าง `system_prompt` ครั้งเดียว) + Pipeline มี translator ตัวเดียว → เปลี่ยนต่อ request ไม่ได้; ปรับให้ client (API) เลือก provider/key/model/ภาษา ต่อ request ได้
- **`translator.py`:** เลิก build `self.system_prompt` ตายตัว → เก็บ `user_prompt/guidelines/from_lang/to_lang/model` เป็น default + `_system_prompt(from_lang,to_lang)` ที่ cache; `translate()` รับ keyword `model/from_lang/to_lang` ต่อ call (override) แล้วส่ง `model`+`system_prompt` เข้า `_translate`; abstract signature → `_translate(inputs, image=None, *, model=None, system_prompt=None)`
- **`gemini.py` + `OpenAICompatibleTranslator`:** `_translate` ใช้ model/system_prompt จาก args (openrouter/groq subclass pass ไม่ต้องแก้)
- **`pipeline.py`:** registry `_translators` cache key `(provider, api_key)` + `_resolve_key` (fallback env `OPENROUTER_API_KEY`/`GOOGLE_API_KEY`/`GROQ_API_KEY`) + `_get_translator` (lazy build client จาก key ที่ส่งมา); `run`/`run_batch`/`_run_on_image` รับ `provider/api_key/model/from_lang/to_lang` ส่งต่อถึง translate; ยังสร้าง default translator ตอน init ไว้ validate
- **`main.py`:** `TranslateRequest` เพิ่ม optional `provider/api_key/model/from_lang/to_lang` → ส่งเข้า run_batch (env เป็น fallback)
- **`tempomunkey.user.js`:** เพิ่ม `@grant GM_setValue/GM_getValue` + Settings panel (provider/model/target language/api_key password) persist ผ่าน GM storage + ใส่ลง POST body เฉพาะ field ที่กรอก
- **tests:** อัปเดต FakeTranslator (default attrs + `_translate` รับ kwargs, เก็บ last_kwargs) + 2 เทสต์ใหม่ (model override ถูกส่ง, to_lang override สร้าง system_prompt ต่างกัน + cache 2 keys)
- **README:** fields ใหม่ในตัวอย่าง `/translate/` + security note (api_key ใน body → ใช้ HTTPS production) + Settings row ในตาราง userscript
- verify: `py_compile` + `node --check` + `pytest` 47 passed (45+2) + `import main` สร้าง registry ได้

**ไฟล์ที่แก้ไข:**
- `manga_translator/translators/translator.py` — per-call model/ภาษา + `_system_prompt` cache + `_translate` signature
- `manga_translator/translators/gemini.py` — `_translate` ใช้ model/system_prompt args
- `pipeline.py` — `(provider,key)` registry + `_resolve_key` + per-call params
- `main.py` — TranslateRequest + ส่ง params เข้า run_batch
- `tempomunkey.user.js` — Settings panel + ส่งใน body
- `tests/test_translator_mapping.py` — FakeTranslator + 2 เทสต์ใหม่
- `README.md` — fields + security note

---

## [2026-06-09 17:02] เพิ่มปุ่ม Interrupt ใน tempomunkey.user.js

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- เพิ่มปุ่ม **Interrupt** ใน floating bar ของ userscript เพื่อยกเลิกคำขอแปลที่กำลังทำอยู่ (เช่น เลือกผิด หรือรอนานเกิน)
- เก็บ handle ของ `GM_xmlhttpRequest` ที่กำลังทำงาน (`activeReq`) + รายการรูปที่กำลังแปล (`activeImgs`); ปุ่ม Interrupt เรียก `activeReq.abort()`
- เพิ่ม `setBusy(busy)` สลับ enable/disable ระหว่างปุ่ม Translate ↔ Interrupt (Interrupt disabled ตอน idle)
- เพิ่ม `onabort` callback: เคลียร์ badge "…" ของรูปที่ค้าง + status "interrupted"; ทุก path (onload/onerror/ontimeout/onabort) reset `setBusy(false)` + `activeReq=null` ครบ
- verify: `node --check` ผ่าน

**ไฟล์ที่แก้ไข:**
- `tempomunkey.user.js` — เพิ่มปุ่ม Interrupt + abort logic + busy-state toggle

---

## [2026-06-09 16:58] เพิ่ม tempomunkey.user.js — userscript เลือกแปลรูป + process_image

**ประเภท:** เพิ่ม feature

**รายละเอียด:**
- สร้าง Tampermonkey userscript ฝั่ง client ให้ผู้ใช้เลือกแปลรูปมังงะบนหน้า reader (ทีละรูป / ทั้งหมด) + toggle process_image (visual context) ตามที่ออกแบบ — คู่กับ `POST /translate/` (รับ `images` list + `process_image`, คืน base64 เรียง index)
- ฟีเจอร์: (1) `scanImages` หา candidate รูป (`naturalWidth >= 400` ตัด icon/banner) วาง **checkbox overlay** มุมรูป + เก็บ original src ไว้ revert; (2) **floating bar** — Select all/Clear, checkbox Visual context, Translate selected, Revert all, Rescan (lazy-load), status; (3) ส่งผ่าน **`GM_xmlhttpRequest`** (ข้าม CORS/mixed-content https→http localhost); (4) swap `img.src` เป็น base64 in-place + badge ✓/✕ ต่อรูป (null = คงรูปเดิม); (5) Revert all คืน src เดิม; reposition overlay ตอน scroll/resize + periodic rescan
- config ด้านบนไฟล์: `API_BASE` (default localhost:8000), `MIN_WIDTH`, `REQUEST_TIMEOUT`; ผู้ใช้ต้องแก้ `@match` เป็นโดเมน tempomunkey + รัน server + ตั้ง provider key
- หมายเหตุ hotlink: ถ้า CDN กัน server fetch → รูปนั้นได้ null (อนาคต: userscript upload bytes แทน URL)
- verify: `node --check` syntax ผ่าน; ไม่แตะ server code (endpoint พร้อมแล้ว)

**ไฟล์ที่แก้ไข:**
- `tempomunkey.user.js` — (ไฟล์ใหม่) Tampermonkey userscript
- `README.md` — เพิ่มหัวข้อ Browser userscript (tempomunkey)

---

## [2026-06-09 15:41] แก้ Docker build fail — opencv libxcb.so.1 (system libs)

**ประเภท:** แก้ bug

**รายละเอียด:**
- `docker compose up --build` พังที่ขั้น pre-download EasyOCR: `ImportError: libxcb.so.1: cannot open shared object file` ตอน `import cv2`
- สาเหตุ: **ultralytics hard-depend `opencv-python` (non-headless) เสมอ** → ต่อให้ pin `opencv-python-headless` pip ก็ลง `opencv-python` ตามมาด้วย และ cv2 ที่โหลดเป็น non-headless ต้องการ X/GL system libs ที่ `python:3.12-slim` ไม่มี
- แก้: (1) `requirement.txt` เปลี่ยน `opencv-python-headless` → `opencv-python==4.13.0.92` ให้ตรงกับ ultralytics (เลี่ยงติดตั้ง opencv 2 ตัวซ้อน); (2) `Dockerfile` + `Dockerfile.gpu` เพิ่ม apt system libs ที่ opencv ต้องการ: `libglib2.0-0 libgl1 libxcb1 libsm6 libxext6 libxrender1` (เดิมมีแค่ libglib2.0-0)
- หมายเหตุ: requirement.txt เปลี่ยน → docker rebuild จะ reinstall pip deps (torch ~540s) อีกรอบ; verify จริงต้องรัน `docker compose up --build` ใหม่ (Docker daemon ไม่ได้รันใน dev นี้)

**ไฟล์ที่แก้ไข:**
- `requirement.txt` — opencv-python-headless → opencv-python
- `Dockerfile`, `Dockerfile.gpu` — เพิ่ม opencv system libs (libgl1/libxcb1/libsm6/libxext6/libxrender1)

---

## [2026-06-09 14:52] /translate/ รับ list รูปจาก client (ใช้กับ tempomunkey)

**ประเภท:** refactor / เพิ่ม feature

**รายละเอียด:**
- เดิม `POST /translate/` รับ `url` หน้าเว็บเดียวแล้ว server scrape `<img>` ทั้งหมด (requests+BeautifulSoup) → เอาทุกรูป (ads/logo/thumbnail ปน) + reader site ที่ JS-render มัก scrape ไม่ได้รูป
- เปลี่ยนเป็นรับ **`images: list[str]`** (image URLs ที่ client/extension เก็บจาก DOM ที่ render แล้ว ตามไอเดีย click+class/sibling ที่หารือกัน) — ย้ายหน้าที่ "หารูปมังงะ" ไปฝั่ง client ที่เห็น DOM จริง
- endpoint ส่ง list เข้า `pipe.run_batch` ตรงๆ (`load_image` ดึง http URL ได้อยู่แล้ว); คืน base64 list ที่ **index-aligned กับ request** (`None` สำหรับรูปที่แปลไม่สำเร็จ) → client map ผลกลับไป swap รูปต้นทาง in-place ได้
- ลบ import ที่ไม่ใช้แล้ว `BeautifulSoup` + `requests` จาก main.py
- decision: tempomunkey ใส่รูปเป็น static `<img src>` URL เข้าถึงได้ → server ดึงเองได้
- หมายเหตุ: extension content-script (เก็บ URL จาก DOM) เป็น deliverable ฝั่ง client แยก; ถ้า CDN กัน hotlink ให้ client ส่ง bytes แทน URL (future); ควรเพิ่ม SSRF allowlist ก่อน production
- verify: `py_compile` + `import main` (สร้าง app+pipeline+EasyOCR จริง, route `/translate/` พร้อม, ไม่มี scrape refs) + `pytest` 45 passed

**ไฟล์ที่แก้ไข:**
- `main.py` — `TranslateRequest` → `images: list[str]`; endpoint ใช้ run_batch + คืน aligned base64; ลบ bs4/requests
- `README.md` — อัปเดตตัวอย่าง `/translate/` เป็น image-list + index-aligned response + hotlink note

---

## [2026-06-09 14:37] Update requirement.txt + เพิ่ม Docker สำหรับ deploy API

**ประเภท:** เพิ่ม feature / refactor

**รายละเอียด:**
- **requirement.txt เขียนใหม่** — pin version ตาม env จริง อิงจาก imports ที่ code ใช้ตอน startup; ตัดที่ไม่จำเป็น: `paddlepaddle`+`paddleocr` (default ใช้ EasyOCR ไม่ import paddle ตอน startup — ใส่เป็น comment optional), `tiktoken` (วัด token ครั้งเดียว), `pytest` (อยู่ `requirements-dev.txt`); เปลี่ยน `opencv-python` → `opencv-python-headless` (server ไม่ต้อง GUI/libGL)
- **ตัด matplotlib ออกจาก runtime** — lazy import: ย้าย `import matplotlib.pyplot` เข้าในฟังก์ชัน debug viz (`show_image_with_boxes`/`show_image_with_polygons`/`show_images` ใน `utils/common.py`, `show_masks` ใน `inpainting/inpainter.py`) เพราะ API path ไม่เรียก → image ไม่ต้องมี matplotlib (verify แล้วไม่มี `plt` ที่ module level)
- **main.py อ่าน config จาก env** — `DEVICE`/`PROVIDER`/`MODEL` (default `cpu`/`openrouter`/`google/gemini-2.5-flash`) แทน hardcode `device="cuda"` → image เดียวรันได้ทั้ง CPU/GPU โดยไม่แก้ code
- **Docker:** `Dockerfile` (CPU, `python:3.12-slim` + `libglib2.0-0` + torch CPU + pre-download EasyOCR en models ให้ self-contained); `Dockerfile.gpu` (base `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, `DEVICE=cuda`, ต้อง `--gpus all`); model `best_diplom.pt` (6.3MB) + font ถูก bake เข้า image ผ่าน COPY
- **ไฟล์ประกอบ:** `.dockerignore` (exclude .git/notebooks/tests/.env/MangaTVL_ENV แต่**เก็บ assets**), `docker-compose.yml` (service + env_file + port 8000 + GPU block comment), `.env.example` เพิ่ม DEVICE/PROVIDER/MODEL, README เพิ่มหัวข้อ Run with Docker
- verify: `py_compile` + `from pipeline import Pipeline` ผ่าน + `pytest` 45 passed; Docker daemon ไม่ได้รันใน dev → build จริงให้ผู้ใช้รัน (`docker compose up --build`)

**ไฟล์ที่แก้ไข:**
- `requirement.txt` — เขียนใหม่ pinned + lean
- `manga_translator/utils/common.py`, `manga_translator/inpainting/inpainter.py` — lazy import matplotlib
- `main.py` — DEVICE/PROVIDER/MODEL จาก env
- `Dockerfile`, `Dockerfile.gpu`, `.dockerignore`, `docker-compose.yml` — (ไฟล์ใหม่)
- `.env.example` — เพิ่ม DEVICE/PROVIDER/MODEL
- `README.md` — หัวข้อ Run with Docker

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
