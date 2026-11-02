import json
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from ..schemas.interface import DetectionResult, OCRResult
from .ocr_engine import OCREngine

MODEL_DIR = Path(__file__).resolve().parent.parent / "assets" / "models" / "baberu"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_PAST = [f"past_k{i}" for i in range(6)] + [f"past_v{i}" for i in range(6)]


def pad_square(img: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    """Pad to a square (the model resizes to 224x224; stretching distorts glyphs)."""
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class BaberuOCR:
    """Baberu OCR (Apache-2.0, ~115M params, ONNX on CPU): manga crops -> text, strong on SFX."""

    def __init__(self, model_dir=MODEL_DIR, vision="vision_fp16.onnx", threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        d = Path(model_dir)

        def session(p):
            return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        self.vis = session(d / "onnx" / vision)
        self.pre = session(d / "onnx/decoder_prefill_int8.onnx")
        self.stp = session(d / "onnx/decoder_step_int8.onnx")
        charset = json.loads((d / "tokenizer/vocab.json").read_text(encoding="utf-8"))
        self.id2ch = {i + 4: ch for i, ch in enumerate(charset)}
        self.content = {i + 4 for i, ch in enumerate(charset)
                        if len(ch) == 1 and ch not in "ーｰ〜~" and unicodedata.category(ch)[0] in "LN"}

    def __call__(self, image: Image.Image, square=True, max_new=256, rep=1.2) -> str:
        img = image.convert("RGB")
        if square:
            img = pad_square(img)
        x = (np.asarray(img.resize((224, 224), Image.BICUBIC), np.float32) / 255.0 - _MEAN) / _STD
        vis = self.vis.run(["vision_embeds"], {"pixel_values": x.transpose(2, 0, 1)[None]})[0]
        out = self.pre.run(None, {"vision_embeds": vis, "input_ids": np.array([[1]], np.int64)})
        logits, present = out[0][0, -1].astype(np.float64), out[1:]
        seq, toks, pos = {1}, [], vis.shape[1] + 1
        for _ in range(max_new):
            for t in seq:  # repetition penalty
                s = logits[t]
                logits[t] = s * rep if s < 0 else s / rep
            if toks:  # cap runs of one token (stops "ーーーー..." loops)
                last = toks[-1]
                cap = 12 if last in self.content else 16 if last > 3 else 0
                run = 0
                for t in reversed(toks):
                    if t != last:
                        break
                    run += 1
                if cap and run >= cap:
                    logits[last] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == 2:  # <eos>
                break
            toks.append(nxt)
            seq.add(nxt)
            feed = {"input_ids": np.array([[nxt]], np.int64), "position_ids": np.array([[pos]], np.int64)}
            feed.update(dict(zip(_PAST, present)))
            out = self.stp.run(None, feed)
            logits, present = out[0][0, -1].astype(np.float64), out[1:]
            pos += 1
        return "".join(self.id2ch.get(t, "") for t in toks)


class BaberuOCREngine(OCREngine):
    """Baberu OCR per detection (Japanese, torch-free)."""

    def __init__(self, **kw):
        super().__init__(backend="baberu")
        self.model = BaberuOCR(**kw)

    def get_ocr(self, image: Image.Image, detection_results: list[DetectionResult]) -> list[OCRResult]:
        res = []
        for det in detection_results:
            x1, y1, x2, y2 = det.bbox
            crop = self.clean_poly(image.crop((x1, y1, x2, y2)), det.segmentation, offset=(x1, y1))
            res.append(OCRResult(text=self.model(crop) or "", boxes=[[x1, y1, x2, y2]], detection_result=det))
        return res
