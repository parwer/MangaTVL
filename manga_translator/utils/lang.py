"""Resolve a human language name (e.g. "english", "japanese", or an ISO code
like "ja") to the language code an OCR engine expects. Backed by
``manga_translator/assets/languages.json`` so codes live in one editable place."""
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
    """Resolve a language name/alias (case-insensitive, trimmed) to its entry
    dict, or None if unknown."""
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
    """Return the canonical language key for ``name`` (matching an alias,
    case-insensitively), or the lowered input when unknown / None when empty."""
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
    """Return the OCR code for ``name`` on ``engine`` ("easyocr" | "paddleocr").
    Matches the canonical key or any alias, case-insensitively. Falls back to
    ``default`` when the language or its code for that engine is unknown."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get(engine) or default


def ocr_engine_name(name, default: str = "easyocr") -> str:
    """Return which OCR backend to use for source language ``name`` (e.g.
    "easyocr" | "mangaocr"). Matches the canonical key or any alias. Falls back
    to ``default`` when the language is unknown or specifies no engine."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get("engine", default)


def wrap_mode(name, default: str = "space") -> str:
    """Return how the renderer should break lines for TARGET language ``name``:
    "pythainlp" (Thai word segmentation), "char" (per-character, spaceless CJK),
    or "space" (break on spaces). Falls back to ``default`` ("space") for
    unknown languages or ones with no ``wrap`` set."""
    entry = _lookup(name)
    if entry is None:
        return default
    return entry.get("wrap", default)
