import json
import re
import threading
import time

import requests

from .offline_mt import _clean

URL = "https://www.bing.com/translator"
# app language name -> Bing language code
CODES = {"english": "en", "thai": "th", "japanese": "ja", "korean": "ko", "chinese": "zh-Hans",
         "chinese_traditional": "zh-Hant", "vietnamese": "vi", "french": "fr", "german": "de", "spanish": "es",
         "italian": "it", "portuguese": "pt", "russian": "ru", "indonesian": "id"}
MAX_CHARS = 900  # the page's translator takes about 1000 characters a request
_lock = threading.Lock()
_state: dict = {}


def _setup():
    """Open the Bing Translator page and keep the token it gives every visitor."""
    s = requests.Session()
    s.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/141.0.0.0 Safari/537.36")
    r = s.get(URL, timeout=15)
    r.raise_for_status()
    key, token, ttl = json.loads(re.search(r"params_AbusePreventionHelper\s*=\s*(\[[^\]]+\])", r.text).group(1))[:3]
    _state.update(session=s, ig=re.search(r'IG:"([^"]+)"', r.text).group(1),
                  iid=re.search(r'data-iid="([^"]+)"', r.text).group(1), key=key, token=token, n=0,
                  until=time.time() + min(int(ttl) / 1000, 1800) - 60)


def _request(lines: list[str], src: str, tgt: str) -> list[str]:
    """One request: the lines joined by line breaks, which Bing keeps."""
    if not _state or time.time() > _state["until"]:
        _setup()
    _state["n"] += 1
    r = _state["session"].post("https://www.bing.com/ttranslatev3",
                               params={"isVertical": "1", "IG": _state["ig"], "IID": f'{_state["iid"]}.{_state["n"]}'},
                               data={"fromLang": src, "to": tgt, "text": "\n".join(lines), "token": _state["token"],
                                     "key": _state["key"]}, timeout=20)
    r.raise_for_status()
    body = r.json()
    if isinstance(body, dict):  # {"statusCode": 400 / 429, ...} or a captcha request
        raise RuntimeError(f"Bing refused the request: {str(body)[:120]}")
    out = [x.strip() for x in body[0]["translations"][0]["text"].split("\n")]
    if len(out) != len(lines):
        raise RuntimeError("Bing returned a different number of lines")
    return out


def translate(texts: list[str], from_lang: str | None, to_lang: str) -> list[str]:
    """Translate bubble texts with Bing Translator (free, no account); raises when Bing can't be used."""
    src, tgt = CODES.get(from_lang or "english", "en"), CODES.get(to_lang, "th")
    if src == tgt or not texts:
        return list(texts)
    lines = [_clean(t)[:MAX_CHARS] or "." for t in texts]
    batches, size = [[]], 0
    for line in lines:  # whole bubbles per request, up to the character limit
        if batches[-1] and size + len(line) + 1 > MAX_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(line)
        size += len(line) + 1
    with _lock:
        try:
            return [t for b in batches for t in _request(b, src, tgt)]
        except Exception:
            _state.clear()  # a stale token: one more try with a new one
            return [t for b in batches for t in _request(b, src, tgt)]
