import os
from manga_translator.rendering.fonts import list_fonts, font_path, bold_font_path, _registry


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
    assert font_path(None, lang="thai").endswith("Mali-Medium.ttf")


def test_language_default_resolves_aliases():
    # "jp" is an alias of japanese in languages.json
    assert font_path(None, lang="jp").endswith("YuseiMagic-Regular.ttf")


def test_unknown_lang_falls_back_to_latin_default():
    assert font_path(None, lang="english").endswith("ComicNeue-Regular.ttf")
    assert font_path(None, lang="klingon").endswith("ComicNeue-Regular.ttf")


def test_unknown_font_key_falls_through_to_language_default():
    # a bad font key must not crash; it falls through to the language default
    assert font_path("no_such_font", lang="thai").endswith("Mali-Medium.ttf")


def test_bold_face_for_emphasis():
    assert bold_font_path(None, lang="thai").endswith("Mali-Bold.ttf")
    assert os.path.isfile(bold_font_path("mali"))
    assert bold_font_path("bangers") is None          # no bold registered
    assert bold_font_path(None, lang="english") is None


SAMPLES = {"english": "Where?", "japanese": "どこ行くの", "korean": "어디 가", "chinese": "你去哪里",
           "chinese_traditional": "你要去哪裡", "thai": "ไปไหน", "vietnamese": "Bạn đi đâu", "indonesian": "Mau ke mana",
           "french": "Où vas-tu", "german": "Wohin gehst", "spanish": "¿Adónde vas?", "italian": "Dove vai",
           "portuguese": "Aonde você vai", "russian": "Куда ты идёшь"}


def test_every_target_language_default_font_has_its_letters():
    from fontTools.ttLib import TTFont
    from manga_translator.utils.lang import load_language_map
    assert set(SAMPLES) == {k for k in load_language_map() if not k.startswith("_")}
    for lang, text in SAMPLES.items():
        cmap = TTFont(font_path(None, lang=lang)).getBestCmap()
        assert [c for c in text if c != " " and ord(c) not in cmap] == [], lang
