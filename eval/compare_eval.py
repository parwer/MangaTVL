"""Compare text-only vs image-process for the manga pipeline, over a folder.

`process_image` only changes the TRANSLATION stage, so per page we translate
twice (text-only / image-process) and judge BOTH with the page image attached
(self-bias constant -> cancels in the delta).

Stages 4/5 are included too:
  * Inpainting depends only on detection/OCR boxes, NOT the translated text, so
    it is **identical for both variants** -> measured once per page.
  * Rendering depends on the translated text -> measured per variant and compared.

``--reuse <prev_run_dir>`` replays a previous compare run from its raw/<id>.json
cache: translations + judge scores are reloaded (NO translate/judge API calls),
and only the local CV (inpaint/render) + metrics are recomputed. Use it to add
stages 4/5 to an existing run without paying for the LLM again.

Usage (from D:\\WorkSpace\\MangaTVL):
    ..\\MangaTVL_ENV\\python.exe -m eval.compare_eval --dir eval/scraped
    ..\\MangaTVL_ENV\\python.exe -m eval.compare_eval --reuse eval/results/<ts>-compare
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from manga_translator.schemas.interface import TranslationResult
from eval.harness import make_image_id, Capture
from eval.judge import run_judge
from eval.metrics import (
    translation_metrics, inpainting_metrics,
    rendering_spill_metrics, rendering_fit_metrics, _text_mask,
)
from eval.run_eval import find_images, aggregate, _ensure_provider_key, schemas_snapshot, git_commit


VARIANTS = ("text_only", "image")


def _usage_nums(usage) -> tuple[float, float]:
    """(total_tokens, cost) from a capture usage dict, best-effort across providers."""
    if not isinstance(usage, dict):
        return 0.0, 0.0
    tokens = usage.get("total_tokens") or usage.get("total_token_count") or 0
    cost = usage.get("cost") or 0.0
    try:
        return float(tokens), float(cost)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _by_idx(per: list[dict]) -> dict:
    return {p["bubble_index"]: p for p in per if "bubble_index" in p}


def _rebuild_translation(ocr, parsed: dict) -> list[TranslationResult]:
    """Reconstruct TranslationResults from a cached text_no->text map, exactly as
    translator.translate maps them (fallback to the OCR text when missing)."""
    parsed = parsed or {}
    out = []
    for i, o in enumerate(ocr):
        t = parsed.get(str(i), parsed.get(i))
        out.append(TranslationResult(ocr_result=o, translated_text=t if t is not None else o.text))
    return out


def _rebuild_judge_per(parsed: dict) -> list[dict]:
    """Rebuild judge per-records from a cached text_no->[adequacy,fluency] map."""
    per = []
    for no, sc in (parsed or {}).items():
        try:
            a, f = int(sc[0]), int(sc[1])
        except (TypeError, ValueError, IndexError):
            continue
        per.append({"bubble_index": int(no), "adequacy": a, "fluency": f, "low_adequacy": a <= 2})
    return per


def _cap(image_id, original, detection, ocr, translation, *, inpainted=None, rendered=None) -> Capture:
    c = Capture(image_path="", image_id=image_id)
    c.original, c.detection, c.ocr, c.translation = original, detection, ocr, translation
    c.inpainted, c.rendered = inpainted, rendered
    return c


async def _translate_variant(translator, ocr, *, image, model, from_lang, to_lang):
    cap: dict = {}
    trans = await translator.translate(
        ocr_result=ocr, image=image, model=model,
        from_lang=from_lang, to_lang=to_lang, capture=cap,
    )
    return trans, cap


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/scraped")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reuse", default=None,
                    help="replay a previous <run_dir>: reuse cached translations+judge, recompute stages 4/5 only (no API)")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="google/gemini-3-flash-preview")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-no-images", action="store_true",
                    help="don't attach original+rendered pages to the judge (for text-only judge models)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from-lang", default="english")
    ap.add_argument("--to-lang", default="thai")
    ap.add_argument("--no-images", action="store_true",
                    help="skip saving intermediate images (saved by default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reuse_dir = Path(args.reuse) if args.reuse else None
    if reuse_dir:
        cfg = json.loads((reuse_dir / "config.json").read_text(encoding="utf-8"))
        # inherit the original run's params so detection/OCR replay identically
        for k in ("dir", "provider", "model", "from_lang", "to_lang", "device",
                  "judge_provider", "judge_model"):
            if k in cfg.get("args", {}):
                setattr(args, k, cfg["args"][k])
        print(f"[reuse] replaying {reuse_dir} — no translate/judge API calls")

    images = find_images(Path(args.dir), args.limit)
    if not images:
        print(f"No images found under {args.dir}")
        return
    print(f"Found {len(images)} image(s) under {args.dir}")

    if not reuse_dir:
        _ensure_provider_key(args.provider, fake=False)
        if args.judge_provider:
            _ensure_provider_key(args.judge_provider, fake=False)

    try:
        from pipeline import Pipeline
        pipeline = Pipeline(provider=args.provider, model=args.model,
                            from_lang=args.from_lang, to_lang=args.to_lang, device=args.device)
    except FileNotFoundError as e:
        print(f"[error] could not build pipeline: {e}")
        return

    jprov = args.judge_provider or args.provider
    jmodel = args.judge_model or args.model
    self_judged = (jprov == args.provider) and (jmodel == args.model)
    translator = jclient = None
    if not reuse_dir:
        translator = pipeline._get_translator(args.provider, pipeline._resolve_key(args.provider, None))
        jclient = pipeline._get_translator(jprov, pipeline._resolve_key(jprov, None)).client
        if self_judged:
            print(f"[judge] same model as translator ({jmodel}); judge sees original+rendered for BOTH "
                  f"variants so self-bias is constant and cancels in the delta.")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else Path("eval/results") / f"{ts}-compare"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({
        "kind": "compare-process-image",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "reused_from": str(reuse_dir) if reuse_dir else None,
        "judge": {"provider": jprov, "model": jmodel,
                  "images": "none" if args.judge_no_images else "original+rendered",
                  "self_judged": self_judged},
        "git_commit": git_commit(),
        "n_images": len(images),
        "images": images,
        "schemas": schemas_snapshot(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest, pairs = [], []
    pooled = {v: {"tr": [], "judge": [], "spill": [], "fit": []} for v in VARIANTS}
    pooled_inpaint = []
    cost = {v: {"translate": 0.0, "judge": 0.0, "tokens": 0.0} for v in VARIANTS}

    for n, path in enumerate(images, 1):
        image_id = make_image_id(path, args.dir)
        print(f"[{n}/{len(images)}] {path}")
        try:
            from manga_translator.utils.common import load_image
            original = load_image(path)
            detection = pipeline.det_model.detect(original)
            if not detection:
                print("    no bubbles — skipped")
                manifest.append({"image_id": image_id, "image_path": path, "status": "skipped", "n_bubbles": 0})
                continue
            ocr = pipeline._get_ocr(args.from_lang).get_ocr(original, detection)
            context = pipeline.resize_image(original, max_size=pipeline.resize_max)

            results = {}
            if reuse_dir:
                cached = json.loads((reuse_dir / "raw" / f"{image_id}.json").read_text(encoding="utf-8"))
            for v in VARIANTS:
                if reuse_dir:
                    capk = "translate_text_only" if v == "text_only" else "translate_image"
                    judk = "judge_text_only" if v == "text_only" else "judge_image"
                    cap = cached.get(capk) or {}
                    trans = _rebuild_translation(ocr, cap.get("parsed"))
                    jper = _rebuild_judge_per((cached.get(judk) or {}).get("parsed"))
                    jd = {"per": jper, "raw": cached.get(judk)}
                else:
                    img_in = None if v == "text_only" else context
                    trans, cap = await _translate_variant(
                        translator, ocr, image=img_in, model=args.model,
                        from_lang=args.from_lang, to_lang=args.to_lang)
                    jd = None  # judged after render (needs the rendered page image)

                tr_m = translation_metrics(_cap(image_id, original, detection, ocr, trans))
                results[v] = {"trans": trans, "cap": cap, "judge": jd, "tr": tr_m}
                pooled[v]["tr"] += tr_m["per"]
                tok, c = _usage_nums(cap.get("usage"))
                cost[v]["translate"] += c
                cost[v]["tokens"] += tok

            # stage 4 — inpaint ONCE (invariant to the translated text)
            inpainted = pipeline.inpainter.inpaint(original, results["text_only"]["trans"])
            inp_m = inpainting_metrics(_cap(image_id, original, detection, ocr,
                                            results["text_only"]["trans"], inpainted=inpainted))
            pooled_inpaint += inp_m["per"]

            # stage 5 — render + measure per variant
            for v in VARIANTS:
                rendered = pipeline.renderer.render(inpainted, results[v]["trans"], lang=args.to_lang)
                capv = _cap(image_id, original, detection, ocr, results[v]["trans"],
                            inpainted=inpainted, rendered=rendered)
                sp = rendering_spill_metrics(capv)
                ft = rendering_fit_metrics(capv, pipeline.renderer, args.to_lang)
                results[v]["rendered"] = rendered
                results[v]["mask"] = _text_mask(capv)
                results[v]["sp"], results[v]["ft"] = sp, ft
                pooled[v]["spill"] += sp["per"]
                pooled[v]["fit"] += ft["per"]

            # judge per variant — now that the rendered page exists, attach the
            # original + that variant's rendered page (resized) so the judge scores
            # against the real visual context. reuse path already loaded judge from cache.
            for v in VARIANTS:
                if not reuse_dir:
                    orig_img = rend_img = None
                    if not args.judge_no_images:
                        orig_img = context  # already resized to resize_max
                        rend_img = pipeline.resize_image(results[v]["rendered"], max_size=pipeline.resize_max)
                    results[v]["judge"] = await run_judge(
                        jprov, jclient, jmodel, results[v]["trans"],
                        from_lang=args.from_lang, to_lang=args.to_lang,
                        same_model_as_translator=self_judged,
                        original_image=orig_img, rendered_image=rend_img,
                    )
                jd = results[v]["judge"]
                pooled[v]["judge"] += jd["per"]
                _, jc = _usage_nums((jd.get("raw") or {}).get("usage"))
                cost[v]["judge"] += jc

            # save intermediate images (default on) — original / overlay / inpaint (shared) /
            # both renders / both text-masks — so the eval can be checked by eye.
            if not args.no_images:
                from eval import artifacts
                dest = out_dir / "images" / image_id
                dest.mkdir(parents=True, exist_ok=True)
                artifacts._save(original, dest / "original.png")
                artifacts._save(artifacts.polygon_overlay(
                    _cap(image_id, original, detection, ocr, results["text_only"]["trans"])),
                    dest / "overlay.png")
                artifacts._save(inpainted, dest / "inpainted.png")
                for v in VARIANTS:
                    artifacts._save(results[v]["rendered"], dest / f"rendered_{v}.png")
                    m = results[v].get("mask")
                    if m is not None:
                        artifacts._save((m > 0).astype("uint8") * 255, dest / f"text_mask_{v}.png")
        except Exception as e:
            print(f"    error: {type(e).__name__}: {e}")
            manifest.append({"image_id": image_id, "image_path": path, "status": "error", "error": str(e), "n_bubbles": 0})
            continue

        # Pair per bubble across variants (+ shared inpaint score).
        inp_idx = _by_idx(inp_m["per"])
        idx = {v: {grp: _by_idx(results[v][grp]["per"] if grp in ("sp", "ft")
                                else results[v][{"tr": "tr", "judge": "judge"}[grp]]["per"])
                   for grp in ("tr", "judge", "sp", "ft")} for v in VARIANTS}
        for i, det in enumerate(detection):
            src = (ocr[i].text if i < len(ocr) else "") or ""
            if not src.strip():
                continue
            row = {"image_id": image_id, "bubble_index": i, "source": src.strip()}
            ink = inp_idx.get(i, {})
            row["inpaint"] = {"std_reduction": ink.get("std_reduction"),
                              "ink_removed": ink.get("ink_removed"),
                              "interior_std_after": ink.get("interior_std_after")}
            for v in VARIANTS:
                trm, jdg = idx[v]["tr"].get(i, {}), idx[v]["judge"].get(i, {})
                spm, ftm = idx[v]["sp"].get(i, {}), idx[v]["ft"].get(i, {})
                row[v] = {
                    "translated": results[v]["trans"][i].translated_text,
                    "fallback": trm.get("fallback"),
                    "target_script_ratio": trm.get("target_script_ratio"),
                    "adequacy": jdg.get("adequacy"),
                    "fluency": jdg.get("fluency"),
                    "spill_ratio": spm.get("spill_ratio"),
                    "fill_ratio": spm.get("fill_ratio"),
                    "font_size": ftm.get("font_size"),
                    "truncated": ftm.get("truncated"),
                }
            ta, ia = row["text_only"]["adequacy"], row["image"]["adequacy"]
            row["delta_adequacy"] = (ia - ta) if (ta is not None and ia is not None) else None
            pairs.append(row)

        (raw_dir / f"{image_id}.json").write_text(json.dumps({
            "image_id": image_id, "image_path": path,
            "translate_text_only": results["text_only"]["cap"],
            "translate_image": results["image"]["cap"],
            "judge_text_only": results["text_only"]["judge"].get("raw"),
            "judge_image": results["image"]["judge"].get("raw"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"image_id": image_id, "image_path": path, "status": "ok", "n_bubbles": len(detection)})

    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    corpus = {v: {
        "translation": aggregate(pooled[v]["tr"]),
        "judge": aggregate(pooled[v]["judge"]),
        "rendering": {"spill": aggregate(pooled[v]["spill"]), "fit": aggregate(pooled[v]["fit"])},
        "mean_translate_tokens_per_page": round(cost[v]["tokens"] / n_ok, 1) if n_ok else None,
        "total_translate_cost_usd": round(cost[v]["translate"], 6),
        "total_judge_cost_usd": round(cost[v]["judge"], 6),
    } for v in VARIANTS}
    inpainting = aggregate(pooled_inpaint)   # shared (same for both variants)

    def _d(grp_a, grp_b, key, sub=None):
        a = (corpus["text_only"][grp_a][sub] if sub else corpus["text_only"][grp_a]).get(key)
        b = (corpus["image"][grp_b][sub] if sub else corpus["image"][grp_b]).get(key)
        return round(b - a, 3) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    delta = {
        "fallback_rate": _d("translation", "translation", "fallback_rate"),
        "untranslated_rate": _d("translation", "translation", "untranslated_rate"),
        "mean_target_script_ratio": _d("translation", "translation", "mean_target_script_ratio"),
        "mean_adequacy": _d("judge", "judge", "mean_adequacy"),
        "mean_fluency": _d("judge", "judge", "mean_fluency"),
        "low_adequacy_rate": _d("judge", "judge", "low_adequacy_rate"),
        "mean_spill_ratio": _d("rendering", "rendering", "mean_spill_ratio", sub="spill"),
        "mean_fill_ratio": _d("rendering", "rendering", "mean_fill_ratio", sub="spill"),
        "truncated_rate": _d("rendering", "rendering", "truncated_rate", sub="fit"),
        "mean_font_size": _d("rendering", "rendering", "mean_font_size", sub="fit"),
        "mean_lines": _d("rendering", "rendering", "mean_lines", sub="fit"),
    }

    summary = {
        "images_evaluated": n_ok,
        "images_skipped": sum(1 for m in manifest if m["status"] == "skipped"),
        "images_errored": sum(1 for m in manifest if m["status"] == "error"),
        "bubbles": len(pairs),
        "judge_self_bias": self_judged,
        "reused_from": str(reuse_dir) if reuse_dir else None,
        "text_only": corpus["text_only"], "image": corpus["image"],
        "inpainting_shared": inpainting, "delta_image_minus_text": delta,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "compare.md").write_text(render_md(args, summary, delta, inpainting), encoding="utf-8")

    print(f"\nDone. Run: {out_dir}")
    print(f"  pairs.jsonl: {len(pairs)} bubbles  |  ok/skip/err: "
          f"{summary['images_evaluated']}/{summary['images_skipped']}/{summary['images_errored']}")
    print(json.dumps(delta, ensure_ascii=False, indent=2))


def render_md(args, s, delta, inpainting) -> str:
    a, b = s["text_only"], s["image"]
    def row(label, grp, key, sub=None, fmt="{}"):
        da = (a[grp][sub] if sub else a[grp]).get(key)
        db = (b[grp][sub] if sub else b[grp]).get(key)
        d = delta.get(key)
        sa = fmt.format(da) if da is not None else "—"
        sb = fmt.format(db) if db is not None else "—"
        return f"| {label} | {sa} | {sb} | {d if d is not None else '—'} |"
    bias = "⚠️ judge ใช้ model เดียวกับ translator (self-bias คงที่ทั้ง 2 ฝั่ง → Δ ยังเทียบได้)" if s["judge_self_bias"] else ""
    ink = inpainting
    lines = [
        "# เทียบ Translation: text-only vs image-process",
        "",
        f"- dataset: `{args.dir}`  |  translator: `{args.model}`" + (f"  |  reuse: `{s['reused_from']}`" if s.get("reused_from") else ""),
        f"- judge: แนบ original + rendered ของแต่ละ variant · {bias}",
        f"- ภาพที่วัด: **{s['images_evaluated']}**  |  ข้าม: {s['images_skipped']}  |  error: {s['images_errored']}  |  bubbles: {s['bubbles']}",
        "",
        "## สเตจ 3 — Translation quality (Δ = image − text_only)",
        "",
        "| metric | text-only | image-process | Δ |",
        "|--------|-----------|---------------|---|",
        row("mean_adequacy (1–5, สูงดี)", "judge", "mean_adequacy"),
        row("mean_fluency (1–5, สูงดี)", "judge", "mean_fluency"),
        row("low_adequacy_rate (ต่ำดี)", "judge", "low_adequacy_rate"),
        row("fallback_rate (ต่ำดี)", "translation", "fallback_rate"),
        row("untranslated_rate (ต่ำดี)", "translation", "untranslated_rate"),
        row("mean_target_script_ratio (สูงดี)", "translation", "mean_target_script_ratio"),
        "",
        "## สเตจ 4 — Inpainting (เท่ากันทั้ง 2 variant — ไม่ขึ้นกับคำแปล)",
        "",
        "| metric | ค่า |",
        "|--------|-----|",
        f"| mean_interior_std_after (ต่ำดี) | {ink.get('mean_interior_std_after','—')} |",
        f"| mean_std_reduction (สูงดี) | {ink.get('mean_std_reduction','—')} |",
        f"| mean_ink_removed (สูงดี) | {ink.get('mean_ink_removed','—')} |",
        "",
        "## สเตจ 5 — Rendering (Δ = image − text_only)",
        "",
        "| metric | text-only | image-process | Δ |",
        "|--------|-----------|---------------|---|",
        row("mean_spill_ratio (ต่ำดี)", "rendering", "mean_spill_ratio", sub="spill"),
        row("mean_fill_ratio", "rendering", "mean_fill_ratio", sub="spill"),
        row("truncated_rate (ต่ำดี)", "rendering", "truncated_rate", sub="fit"),
        row("mean_font_size", "rendering", "mean_font_size", sub="fit"),
        row("mean_lines", "rendering", "mean_lines", sub="fit"),
        "",
        "## ต้นทุน (ต่อ run ทั้ง dataset)",
        "",
        "| | text-only | image-process |",
        "|--|-----------|---------------|",
        f"| translate tokens/หน้า (เฉลี่ย) | {a['mean_translate_tokens_per_page']} | {b['mean_translate_tokens_per_page']} |",
        f"| translate cost รวม (USD) | {a['total_translate_cost_usd']} | {b['total_translate_cost_usd']} |",
        f"| judge cost รวม (USD) | {a['total_judge_cost_usd']} | {b['total_judge_cost_usd']} |",
        "",
        "> Δ คุณภาพ > 0 = image-process ดีกว่า · render ต่างกันเพราะคำแปลต่างความยาว · inpaint เท่ากันเพราะลบจาก ocr box ไม่ใช่คำแปล",
        "> ดู `pairs.jsonl` เพื่อสาวกลับรายตัว, `raw/<id>.json` เพื่อ raw req/resp",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
