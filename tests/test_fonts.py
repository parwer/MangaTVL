"""Tests for rendering.fonts — font registry & per-language resolution."""
import os
from manga_translator.rendering.fonts import list_fonts, font_path, _registry


def test_list_fonts_shape():
    fonts = list_fonts()
    assert len(fonts) >= 5
    for f in fonts:
        assert set(f) == {"key", "label", "scripts"}
        assert isinstance(f["scripts"], list)


def test_every_registered_font_file_exists():
    # the resolver must point at a real file for every key (catches typos / missing downloads)
    for f in _registry()["fonts"]:
        path = font_path(f["key"])
        assert os.path.isfile(path), f"{f['key']} -> {path} missing"


def test_explicit_key_wins():
    p = font_path("bangers")
    assert p.endswith("Bangers-Regular.ttf")


def test_language_default_when_no_font():
    assert font_path(None, lang="japanese").endswith("YuseiMagic-Regular.ttf")
    assert font_path(None, lang="chinese").endswith("ZhiMangXing-Regular.ttf")
    assert font_path(None, lang="korean").endswith("NanumPenScript-Regular.ttf")
    assert font_path(None, lang="thai").endswith("THSarabunNew.ttf")


def test_language_default_resolves_aliases():
    # "jp" is an alias of japanese in languages.json
    assert font_path(None, lang="jp").endswith("YuseiMagic-Regular.ttf")


def test_unknown_lang_falls_back_to_latin_default():
    assert font_path(None, lang="english").endswith("ComicNeue-Regular.ttf")
    assert font_path(None, lang="klingon").endswith("ComicNeue-Regular.ttf")


def test_unknown_font_key_falls_through_to_language_default():
    # a bad font key must not crash; it falls through to the language default
    assert font_path("no_such_font", lang="thai").endswith("THSarabunNew.ttf")
