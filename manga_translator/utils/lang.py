import json
import functools
from pathlib import Path

_LANG_PATH = Path(__file__).resolve().parent.parent / "assets" / "languages.json"


@functools.lru_cache(maxsize=1)
def load_language_map() -> dict:
    with open(_LANG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _lookup(name):
    """Language entry for a name or alias, or None."""
    if not name:
        return None
    key = str(name).strip().lower()
    langs = load_language_map()
    entry = langs.get(key)
    if entry is None:
        for info in langs.values():
            if key in (a.lower() for a in info.get("aliases", [])):
                return info
    return entry


def canonical_name(name):
    """Canonical language key for ``name`` (aliases resolved)."""
    if not name:
        return None
    key = str(name).strip().lower()
    langs = load_language_map()
    if key in langs:
        return key
    for k, info in langs.items():
        if key in (a.lower() for a in info.get("aliases", [])):
            return k
    return key


def ocr_lang_code(name, engine: str = "easyocr", default: str = "en") -> str:
    """Return the OCR code for ``name`` on ``engine`` ("easyocr" | "paddleocr")."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get(engine) or default


def ocr_engine_name(name, default: str = "easyocr") -> str:
    """OCR backend for source language ``name``."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get("engine", default)


def wrap_mode(name, default: str = "space") -> str:
    """Line-break mode for target language ``name``: "pythainlp", "char" or "space"."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get("wrap", default)
