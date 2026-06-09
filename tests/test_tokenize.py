"""Tests for rendering.tokenize — language-aware line-break tokenization."""
from manga_translator.rendering.tokenize import tokenize, is_space_delimited


def test_space_delimited_flags():
    assert is_space_delimited("english") is True
    assert is_space_delimited("korean") is True       # manhwa uses spaces
    assert is_space_delimited("french") is True
    assert is_space_delimited("thai") is False
    assert is_space_delimited("japanese") is False
    assert is_space_delimited("chinese") is False


def test_unknown_language_defaults_to_space():
    assert is_space_delimited("klingon") is True
    assert tokenize("hello world", "klingon") == ["hello", "world"]


def test_space_language_splits_on_spaces():
    assert tokenize("hello world foo", "english") == ["hello", "world", "foo"]
    # words stay intact (no mid-word breaks)
    assert tokenize("annyeonghaseyo chingu", "korean") == ["annyeonghaseyo", "chingu"]


def test_cjk_splits_per_character():
    assert tokenize("こんにちは", "japanese") == list("こんにちは")
    assert tokenize("你好世界", "chinese") == ["你", "好", "世", "界"]


def test_thai_uses_word_segmentation():
    # pythainlp should segment into more than one word (not a single blob, not per-char)
    text = "สวัสดีชาวโลก"  # "สวัสดีชาวโลก"
    tokens = tokenize(text, "thai")
    assert len(tokens) >= 2
    assert "".join(tokens) == text          # lossless: rejoining reproduces the input
    assert all(len(t) >= 1 for t in tokens)  # real words, not empty
