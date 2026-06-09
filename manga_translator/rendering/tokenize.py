"""Language-aware tokenization for the renderer's line breaking.

The renderer wraps a translated string to fit a speech bubble by inserting
candidate break points (spaces) and letting ``textwrap`` break on them. *Where*
those break points may go depends on the target language's script:

- Thai has no spaces between words, so a word segmenter (pythainlp) decides the
  break points; lines must not split mid-word.
- Japanese / Chinese are also spaceless but break per character (any character
  boundary is a valid break), so character-level tokenization is enough and
  avoids a heavy morphological analyzer.
- Latin scripts, Korean, etc. are space-delimited — the existing spaces already
  mark the break points (whole words stay intact).

The mode for each language lives in ``assets/languages.json`` (``wrap`` field)
so new languages are added by editing data, not code.
"""
from ..utils.lang import wrap_mode


def is_space_delimited(lang) -> bool:
    """True when ``lang`` separates words with spaces (Latin, Korean, ...), so
    its spaces are real word separators to keep. False for Thai/CJK, whose
    break points are inserted only for wrapping and stripped afterwards."""
    return wrap_mode(lang) == "space"


def tokenize(text, lang):
    """Split ``text`` into break-unit tokens for target language ``lang``.

    Returns a list of tokens; a line break is allowed between adjacent tokens
    but not within one. The caller joins them with spaces for ``textwrap``.
    """
    mode = wrap_mode(lang)
    if mode == "pythainlp":
        from pythainlp import word_tokenize  # lazy: only needed for Thai output
        return word_tokenize(text, engine="newmm")
    if mode == "char":
        return list(text)
    return text.split(" ")
