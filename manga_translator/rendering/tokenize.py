from ..utils.lang import wrap_mode


def is_space_delimited(lang) -> bool:
    """True when ``lang`` separates words with spaces."""
    return wrap_mode(lang) == "space"


def tokenize(text, lang):
    """Split ``text`` into break-unit tokens for target language ``lang``."""
    mode = wrap_mode(lang)
    if mode == "pythainlp":
        from pythainlp import word_tokenize  # lazy: only needed for Thai output
        return word_tokenize(text, engine="newmm")
    if mode == "char":
        return list(text)
    return text.split(" ")
