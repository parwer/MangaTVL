"""Batch eval over a folder of raw manga pages, with full raw-data logging so
every metric can be audited back to the exact stage I/O that produced it.

Usage (from D:\\WorkSpace\\MangaTVL):

    ..\\MangaTVL_ENV\\python.exe -m eval.run_eval --dir eval/scraped --limit 10

Key flags:
    --dir DIR            folder of images (recursively globs *.png/*.jpg). default: eval/scraped
    --limit N            only the first N images (0 = all). default: 0
    --provider P         openrouter | gemini | groq. default: openrouter
    --model M            model id passed to the translator. default: google/gemini-3-flash-preview
    --process-image      send the page to the VLM as visual context (stage 3)
    --fake-translate     skip the LLM entirely; copy OCR text through (no API cost)
    --judge              2nd LLM call scoring adequacy/fluency of each translation
    --judge-provider P / --judge-model M   independent judge (default: same as translator)
    --save-images        also dump original/inpainted/rendered/text-mask/overlay per page
    --device cpu|cuda    default: cpu
    --out DIR            results dir. default: eval/results/<run_id>

Output layout (eval/results/<run_id>/):
    config.json     params + JSON schema of the data contracts + git commit + image list
    manifest.json   per-image: status, bubble count, timings
    corpus.json     micro-averaged aggregate metrics
    summary.md      Thai summary table
    bubbles.jsonl   one row per bubble (the flat audit table)
    images/<id>.json  full per-image record: raw stage I/O + per-bubble metrics
    images/<id>/*.png (only with --save-images)
"""

import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env up front: the pipeline also calls load_dotenv() on import, but that
# happens *after* we check for a provider key below — and load_dotenv won't
# override an env var once set. Loading here first ensures real keys win over
# the construction-time "eval-dummy" placeholder.
load_dotenv()

from eval.harness import run_capture
from eval import metrics as M


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AGG_IGNORE = {"bubble_index"}   # provenance fields that must not be averaged


def find_images(folder: Path, limit: int = 0) -> list[str]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if limit:
        files = files[:limit]
    return [str(p) for p in files]


def aggregate(pers: list[dict]) -> dict:
    """Corpus-level (micro-averaged) summary over pooled per-bubble records:
    booleans become rates, numbers become means. Provenance keys are skipped."""
    if not pers:
        return {"bubbles": 0}
    out = {"bubbles": len(pers)}
    keys = {k for p in pers for k in p} - AGG_IGNORE
    for k in sorted(keys):
        vals = [p[k] for p in pers if p.get(k) is not None]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            out[f"{k}_rate"] = round(sum(vals) / len(vals), 3)
        elif all(isinstance(v, (int, float)) for v in vals):
            out[f"mean_{k}"] = round(sum(vals) / len(vals), 3)
    return out


def _index_by_bubble(per: list[dict]) -> dict:
    return {p["bubble_index"]: p for p in per if "bubble_index" in p}


def bubble_rows(cap, run_id, tr, ip, sp, ft, jd) -> list[dict]:
    """Flatten every bubble of one page into one audit row: provenance ids +
    raw text + every stage's metric, merged by bubble_index."""
    t_i, i_i, s_i, f_i = (_index_by_bubble(x["per"]) for x in (tr, ip, sp, ft))
    j_i = _index_by_bubble(jd["per"]) if jd else {}
    rows = []
    for i, det in enumerate(cap.detection):
        ocr = cap.ocr[i] if i < len(cap.ocr) else None
        trans = cap.translation[i] if i < len(cap.translation) else None
        row = {
            "run_id": run_id,
            "image_id": cap.image_id,
            "image_path": cap.image_path,
            "bubble_index": i,
            "form_type": det.form_type,
            "bbox": [int(v) for v in det.bbox],
            "has_polygon": bool(det.segmentation),
            "ocr_text": ocr.text if ocr else None,
            "source_text": trans.ocr_result.text if trans else None,
            "translated_text": trans.translated_text if trans else None,
        }
        for src, drop in ((t_i, {"source", "translated"}), (i_i, set()), (s_i, set()), (f_i, set()), (j_i, set())):
            rec = src.get(i)
            if rec:
                row.update({k: v for k, v in rec.items() if k != "bubble_index" and k not in drop})
        rows.append(row)
    return rows


def schemas_snapshot() -> dict:
    from manga_translator.schemas.interface import DetectionResult, OCRResult, TranslationResult
    return {c.__name__: c.model_json_schema() for c in (DetectionResult, OCRResult, TranslationResult)}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _ensure_provider_key(provider: str, fake: bool) -> bool:
    env_key = {"openrouter": "OPENROUTER_API_KEY",
               "gemini": "GOOGLE_API_KEY",
               "groq": "GROQ_API_KEY"}.get(provider, "")
    has_real = bool(os.getenv(env_key))
    if not has_real:
        os.environ[env_key] = "eval-dummy"  # lets the client construct; calls will fail -> fallback
        if not fake:
            print(f"[warn] {env_key} not set — stage-3 calls will fail and fall back to the "
                  f"original text (fallback_rate will be ~1.0). Use --fake-translate to silence this.")
    return has_real


def _slug(path: str) -> str:
    base = Path(path).name.lower()
    return "".join(c if c.isalnum() else "-" for c in base).strip("-")[:40] or "dataset"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/scraped")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--process-image", action="store_true")
    ap.add_argument("--fake-translate", action="store_true")
    ap.add_argument("--judge", action="store_true",
                    help="run a 2nd LLM call to score adequacy/fluency of each translation")
    ap.add_argument("--judge-provider", default=None, help="default: same as --provider")
    ap.add_argument("--judge-model", default=None, help="default: same as --model (self-bias — see README)")
    ap.add_argument("--judge-no-images", action="store_true",
                    help="don't attach the original+rendered page to the judge (use for text-only judge models)")
    ap.add_argument("--save-images", action="store_true",
                    help="dump original/inpainted/rendered/text-mask/overlay per page")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--to-lang", default="thai")
    ap.add_argument("--from-lang", default="english")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    images = find_images(Path(args.dir), args.limit)
    if not images:
        print(f"No images found under {args.dir}")
        return
    print(f"Found {len(images)} image(s) under {args.dir}")

    _ensure_provider_key(args.provider, args.fake_translate)

    # Build the pipeline (this loads the YOLO + OCR models — can take a moment).
    try:
        from pipeline import Pipeline
        pipeline = Pipeline(provider=args.provider, model=args.model,
                            from_lang=args.from_lang, to_lang=args.to_lang,
                            device=args.device)
    except FileNotFoundError as e:
        print(f"[error] could not build pipeline: {e}\n"
              f"        The YOLO weights (assets/models/best_diplom.pt) must be present.")
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + _slug(args.dir)
    out_dir = Path(args.out) if args.out else Path("eval/results") / run_id
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # config.json — written up front so a crashed run still records what it ran.
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "git_commit": git_commit(),
        "n_images": len(images),
        "images": images,
        "schemas": schemas_snapshot(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional LLM judge (2nd call): reuse the pipeline's translator client for
    # the chosen provider, but with a separate judge rubric prompt + model.
    judge_cfg = None
    if args.judge and not args.fake_translate:
        from eval.judge import run_judge
        jprov = args.judge_provider or args.provider
        jmodel = args.judge_model or args.model
        jclient = pipeline._get_translator(jprov, pipeline._resolve_key(jprov, None)).client
        same = (jprov == args.provider) and (jmodel == args.model)
        judge_cfg = {"run": run_judge, "provider": jprov, "client": jclient, "model": jmodel, "same": same}
        if same:
            print(f"[judge] using SAME model as translator ({jmodel}) — scores are optimistic "
                  f"(self-bias). Pass --judge-model for an independent judge.")

    if args.save_images:
        from eval import artifacts

    manifest = []
    all_rows = []
    pooled = {"translation": [], "inpainting": [], "spill": [], "fit": [], "judge": []}
    timings = {}

    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path}")
        cap = await run_capture(
            pipeline, path,
            dataset_root=args.dir,
            process_image=args.process_image,
            fake_translate=args.fake_translate,
            from_lang=args.from_lang, to_lang=args.to_lang,
        )
        if cap.error:
            print(f"    error: {cap.error}")
            manifest.append({"image_id": cap.image_id, "image_path": path, "status": "error",
                             "error": cap.error, "n_bubbles": 0})
            continue
        if cap.skipped:
            print("    no bubbles detected — skipped")
            manifest.append({"image_id": cap.image_id, "image_path": path, "status": "skipped",
                             "n_bubbles": 0, "timings": cap.timings})
            continue

        tr = M.translation_metrics(cap)
        ip = M.inpainting_metrics(cap)
        sp = M.rendering_spill_metrics(cap)
        ft = M.rendering_fit_metrics(cap, pipeline.renderer, args.to_lang)

        jd = None
        if judge_cfg is not None:
            # Attach the original + rendered page (resized to keep tokens down) so
            # the judge scores adequacy/fluency against the real visual context.
            orig_img = rend_img = None
            if not args.judge_no_images:
                orig_img = pipeline.resize_image(cap.original, max_size=pipeline.resize_max)
                if cap.rendered is not None:
                    rend_img = pipeline.resize_image(cap.rendered, max_size=pipeline.resize_max)
            jd = await judge_cfg["run"](
                judge_cfg["provider"], judge_cfg["client"], judge_cfg["model"],
                cap.translation, from_lang=args.from_lang, to_lang=args.to_lang,
                same_model_as_translator=judge_cfg["same"],
                original_image=orig_img, rendered_image=rend_img,
            )
            cap.judge_raw = jd.get("raw")
            pooled["judge"] += jd["per"]

        pooled["translation"] += tr["per"]
        pooled["inpainting"] += ip["per"]
        pooled["spill"] += sp["per"]
        pooled["fit"] += ft["per"]
        for k, v in cap.timings.items():
            timings.setdefault(k, []).append(v)

        rows = bubble_rows(cap, run_id, tr, ip, sp, ft, jd)
        all_rows += rows

        # Full per-image record: raw stage I/O (to_record) + computed metrics.
        image_rec = cap.to_record()
        image_rec["metrics_summary"] = {
            "translation": tr["summary"], "inpainting": ip["summary"],
            "rendering_spill": sp["summary"], "rendering_fit": ft["summary"],
            **({"translation_judge": jd["summary"]} if jd else {}),
        }
        image_rec["bubble_metrics"] = rows
        (images_dir / f"{cap.image_id}.json").write_text(
            json.dumps(image_rec, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.save_images:
            artifacts.save_page(cap, images_dir / cap.image_id, M._text_mask(cap))

        manifest.append({"image_id": cap.image_id, "image_path": path, "status": "ok",
                         "n_bubbles": len(cap.detection),
                         "timings": {k: round(v, 3) for k, v in cap.timings.items()}})

    corpus = {
        "images_evaluated": sum(1 for m in manifest if m["status"] == "ok"),
        "images_skipped": sum(1 for m in manifest if m["status"] == "skipped"),
        "images_errored": sum(1 for m in manifest if m["status"] == "error"),
        "translation": aggregate(pooled["translation"]),
        "inpainting": aggregate(pooled["inpainting"]),
        "rendering_spill": aggregate(pooled["spill"]),
        "rendering_fit": aggregate(pooled["fit"]),
        "mean_timings_sec": {k: round(sum(v) / len(v), 3) for k, v in timings.items()},
    }
    if judge_cfg is not None:
        jc = aggregate(pooled["judge"])
        jc["self_judged"] = judge_cfg["same"]
        corpus["translation_judge"] = jc

    # Write the run artefacts.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "bubbles.jsonl").open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "summary.md").write_text(render_summary_md(args, run_id, corpus), encoding="utf-8")

    print(f"\nDone. Run: {out_dir}")
    print(f"  bubbles.jsonl: {len(all_rows)} rows  |  images/: {corpus['images_evaluated']} records")
    print(json.dumps(corpus, ensure_ascii=False, indent=2))


def render_summary_md(args, run_id, corpus: dict) -> str:
    c = corpus
    def g(d, k):
        return d.get(k, "—")
    lines = [
        f"# MangaTVL — สรุปผลการวัด (reference-free)",
        "",
        f"- run: `{run_id}`",
        f"- โฟลเดอร์: `{args.dir}`  |  provider: `{args.provider}`  |  model: `{args.model}`  |  fake-translate: `{args.fake_translate}`",
        f"- ภาพที่วัดได้: **{c['images_evaluated']}**  |  ข้าม(ไม่เจอ bubble): {c['images_skipped']}  |  error: {c['images_errored']}",
        f"- raw log: `images/<id>.json` ต่อรูป · `bubbles.jsonl` ต่อ bubble (audit ย้อนกลับได้)",
        "",
        "## สเตจ 3 — Translation",
        "",
        "| metric | ค่า | ความหมาย |",
        "|--------|-----|----------|",
        f"| fallback_rate | {g(c['translation'],'fallback_rate')} | ตกกลับใช้ข้อความเดิม (API fail/ไม่มี key) — ต่ำดี |",
        f"| untranslated_rate | {g(c['translation'],'untranslated_rate')} | output ไม่ใช่ภาษาเป้าหมาย — ต่ำดี |",
        f"| mean_target_script_ratio | {g(c['translation'],'mean_target_script_ratio')} | สัดส่วนอักษรไทยใน output — สูงดี |",
        f"| mean_len_ratio | {g(c['translation'],'mean_len_ratio')} | ความยาวแปล/ต้นฉบับ |",
        "",
    ]
    if "translation_judge" in c:
        j = c["translation_judge"]
        bias = " ⚠️ judge ใช้ model เดียวกับ translator → คะแนนเอนสูง (self-bias)" if j.get("self_judged") else ""
        lines += [
            f"### สเตจ 3 — LLM-as-judge (เชิงความหมาย){bias}",
            "",
            "| metric | ค่า | ความหมาย |",
            "|--------|-----|----------|",
            f"| mean_adequacy | {g(j,'mean_adequacy')} | ความหมายตรงต้นฉบับ (1–5) — สูงดี |",
            f"| mean_fluency | {g(j,'mean_fluency')} | ภาษาไทยเป็นธรรมชาติ (1–5) — สูงดี |",
            f"| low_adequacy_rate | {g(j,'low_adequacy_rate')} | สัดส่วน bubble ที่ adequacy ≤ 2 — ต่ำดี |",
            "",
        ]
    lines += [
        "## สเตจ 4 — Inpainting (วัดในขอบเขต polygon)",
        "",
        "| metric | ค่า | ความหมาย |",
        "|--------|-----|----------|",
        f"| mean_interior_std_after | {g(c['inpainting'],'mean_interior_std_after')} | ความแปรปรวนในฟองหลังลบ — ต่ำ=เรียบ/สะอาด |",
        f"| mean_std_reduction | {g(c['inpainting'],'mean_std_reduction')} | สัดส่วนความแปรปรวนที่ลบได้ — สูงดี |",
        f"| mean_ink_residual_after | {g(c['inpainting'],'mean_ink_residual_after')} | หมึกดำที่ยังค้าง — ต่ำดี |",
        f"| mean_ink_removed | {g(c['inpainting'],'mean_ink_removed')} | สัดส่วนหมึกที่ลบออกได้ — สูงดี |",
        "",
        "## สเตจ 5 — Rendering",
        "",
        "| metric | ค่า | ความหมาย |",
        "|--------|-----|----------|",
        f"| spills_rate | {g(c['rendering_spill'],'spills_rate')} | สัดส่วน bubble ที่ text ล้นนอกฟอง >5% — ต่ำดี |",
        f"| mean_spill_ratio | {g(c['rendering_spill'],'mean_spill_ratio')} | สัดส่วน text ที่ล้นนอกฟองเฉลี่ย — ต่ำดี |",
        f"| mean_fill_ratio | {g(c['rendering_spill'],'mean_fill_ratio')} | สัดส่วน pixel ตัวอักษรเทียบพื้นที่ฟอง (หมึกบาง=ค่าต่ำ) |",
        f"| truncated_rate | {g(c['rendering_fit'],'truncated_rate')} | ถูกตัดด้วย … เพราะไม่พอที่ — ต่ำดี |",
        f"| mean_font_size | {g(c['rendering_fit'],'mean_font_size')} | ขนาดฟอนต์เฉลี่ยที่ fit ได้ |",
        f"| mean_lines | {g(c['rendering_fit'],'mean_lines')} | จำนวนบรรทัดเฉลี่ยต่อฟอง |",
        "",
        "## เวลา (วินาที/หน้า, เฉลี่ย)",
        "",
        "| stage | sec |",
        "|-------|-----|",
    ]
    for k, v in c["mean_timings_sec"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
