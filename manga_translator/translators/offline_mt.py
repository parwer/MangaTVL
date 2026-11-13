import re
import threading
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "assets" / "models" / "m2m100_418m_ct2"
# app language name -> M2M100 language code (the model has one Chinese)
CODES = {"english": "en", "thai": "th", "japanese": "ja", "korean": "ko", "chinese": "zh", "chinese_traditional": "zh",
         "vietnamese": "vi", "french": "fr", "german": "de", "spanish": "es", "italian": "it", "portuguese": "pt",
         "russian": "ru", "indonesian": "id"}
_lock = threading.Lock()
_model = None


def available() -> bool:
    """True when the offline model ships with this app."""
    return (MODEL_DIR / "model.bin").exists()


def _load():
    """The CTranslate2 model and its tokenizer, loaded on first use (CPU, int8)."""
    global _model
    with _lock:
        if _model is None:
            import ctranslate2
            import sentencepiece
            _model = (ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8"),
                      sentencepiece.SentencePieceProcessor(model_file=str(MODEL_DIR / "sentencepiece.bpe.model")))
    return _model


def _clean(text: str) -> str:
    """One line; manga lettering in capitals becomes sentence case (the model reads it far better)."""
    text = re.sub(r"\s+", " ", text).strip()
    if text.isupper():
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text.lower())
        text = re.sub(r"\bi\b", "I", text)
    return text


def translate(texts: list[str], from_lang: str | None, to_lang: str) -> list[str]:
    """Translate bubble texts offline with M2M100 (free, no account, no rate limit)."""
    src, tgt = CODES.get(from_lang or "english", "en"), CODES.get(to_lang, "th")
    if src == tgt or not texts:
        return list(texts)
    model, sp = _load()
    lines = [_clean(t) for t in texts]
    tokens = [[f"__{src}__"] + sp.encode(t, out_type=str) + ["</s>"] for t in lines]
    out = model.translate_batch(tokens, target_prefix=[[f"__{tgt}__"]] * len(tokens), beam_size=4,
                                repetition_penalty=1.2, no_repeat_ngram_size=3, max_decoding_length=128)
    done = [sp.decode(h.hypotheses[0][1:]) for h in out]
    # Thai: the model writes SARA AM as two code points, fonts want the single one
    return [d.replace("ํา", "ำ") if d.strip() else t for d, t in zip(done, lines)]
