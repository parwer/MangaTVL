"""Font registry for the renderer.

Fonts live in ``assets/fonts/`` and are described in ``assets/fonts.json`` so a
request can pick one by key (``font``) and the renderer/userscript stay in sync.
Each font records which scripts it actually has glyphs for, so a font is only
offered for target languages it can render (a Latin-only comic font can't draw
Thai/CJK). When a request doesn't name a font, a per-language default is used.
"""
import json
import functools
from pathlib import Path

from ..utils.lang import canonical_name

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_FONTS_JSON = _ASSETS / "fonts.json"
_FONTS_DIR = _ASSETS / "fonts"


@functools.lru_cache(maxsize=1)
def _registry() -> dict:
    with open(_FONTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _by_key(key):
    if not key:
        return None
    for font in _registry().get("fonts", []):
        if font["key"] == key:
            return font
    return None


def list_fonts() -> list[dict]:
    """Public catalogue for the API / userscript: ``[{key, label, scripts}]``."""
    return [
        {"key": f["key"], "label": f["label"], "scripts": f.get("scripts", [])}
        for f in _registry().get("fonts", [])
    ]


def font_path(name, lang=None, fallback=None) -> str:
    """Resolve a font selection to a TTF path.

    Priority: explicit ``name`` (font key) -> the ``defaults`` entry for target
    ``lang`` (canonicalised) -> ``defaults['latin']`` -> ``fallback`` -> the
    first registered font. Unknown keys fall through to the language default so a
    bad ``font`` never crashes rendering.
    """
    entry = _by_key(name)
    if entry is None and lang is not None:
        defaults = _registry().get("defaults", {})
        key = defaults.get(canonical_name(lang)) or defaults.get("latin")
        entry = _by_key(key)
    if entry is None:
        if fallback:
            return str(fallback)
        fonts = _registry().get("fonts", [])
        if fonts:
            return str(_FONTS_DIR / fonts[0]["file"])
        raise FileNotFoundError("No fonts registered in fonts.json")
    return str(_FONTS_DIR / entry["file"])
