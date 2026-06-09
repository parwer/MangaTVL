"""Tests for utils/lang — language name/alias -> OCR engine code & backend."""
from manga_translator.utils.lang import ocr_lang_code, ocr_engine_name, load_language_map


def test_canonical_easyocr():
    assert ocr_lang_code("english") == "en"
    assert ocr_lang_code("japanese") == "ja"
    assert ocr_lang_code("korean") == "ko"


def test_paddleocr_codes_differ():
    assert ocr_lang_code("japanese", engine="paddleocr") == "japan"
    assert ocr_lang_code("korean", engine="paddleocr") == "korean"
    assert ocr_lang_code("chinese", engine="paddleocr") == "ch"


def test_alias_resolves():
    assert ocr_lang_code("jp") == "ja"
    assert ocr_lang_code("zh-cn") == "ch_sim"


def test_case_insensitive():
    assert ocr_lang_code("ENGLISH") == "en"
    assert ocr_lang_code("  Japanese  ") == "ja"


def test_unknown_returns_default():
    assert ocr_lang_code("klingon") == "en"
    assert ocr_lang_code("klingon", default="xx") == "xx"
    assert ocr_lang_code(None) == "en"


def test_map_has_no_comment_key():
    assert "_comment" not in load_language_map()


def test_ocr_engine_name_japanese_uses_mangaocr():
    assert ocr_engine_name("japanese") == "mangaocr"
    assert ocr_engine_name("jp") == "mangaocr"      # alias resolves too
    assert ocr_engine_name("JAPANESE") == "mangaocr"  # case-insensitive


def test_ocr_engine_name_defaults_to_easyocr():
    assert ocr_engine_name("english") == "easyocr"
    assert ocr_engine_name("korean") == "easyocr"   # manhwa is horizontal -> EasyOCR
    assert ocr_engine_name("chinese") == "easyocr"


def test_ocr_engine_name_unknown_returns_default():
    assert ocr_engine_name("klingon") == "easyocr"
    assert ocr_engine_name(None) == "easyocr"
    assert ocr_engine_name("klingon", default="xx") == "xx"
