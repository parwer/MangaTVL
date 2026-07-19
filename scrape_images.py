"""Scrape page images into an evaluation set (for testing the translator).

Loads each URL in a real (headless) browser via Playwright, scrolls to the
bottom to trigger lazy-loading, collects ``<img>`` elements at least
``--min-width`` px wide (in DOM order, so manga pages keep their reading order),
and downloads them through the browser's own session (cookies + referer) so
hotlink-protected CDNs still serve them. Saves to:

    <out>/<page-slug>/001.jpg, 002.jpg, ...

Usage (run from the MangaTVL/ directory):
    ..\\MangaTVL_ENV\\python.exe scrape_images.py URL [URL ...] [options]
    ..\\MangaTVL_ENV\\python.exe scrape_images.py --urls-file urls.txt

Options:
    --out DIR          output root (default: eval/scraped)
    --min-width PX     skip images narrower than this (default: 400; cuts icons/banners)
    --limit N          keep at most N images per page (default: 0 = all)
    --scroll-rounds N  max lazy-load scroll steps (default: 60)
    --headful          show the browser window (useful to debug / pass a captcha)
    --timeout MS       per-navigation timeout (default: 60000)

Setup (once):
    ..\\MangaTVL_ENV\\python.exe -m pip install playwright
    ..\\MangaTVL_ENV\\python.exe -m playwright install chromium
"""
import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlsplit

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit(
        "Playwright is not installed. Run:\n"
        "  python -m pip install playwright\n"
        "  python -m playwright install chromium"
    )

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_CTYPE_EXT = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
    "image/webp": ".webp", "image/gif": ".gif", "image/avif": ".avif",
}


def slugify(text, fallback="page"):
    text = re.sub(r"[^\w\-]+", "-", (text or "").strip().lower()).strip("-")
    return text[:60] or fallback


def page_slug(page, url):
    try:
        title = page.title() or ""
    except Exception:
        title = ""
    netloc = urlparse(url).netloc
    return slugify(f"{netloc}-{title}" if title else netloc)


_FORCE_LAZY_JS = """() => {
  // Promote common lazy-load patterns to a real, eagerly-loaded src so every
  // image actually downloads (not just the few near the initial viewport).
  let n = 0;
  for (const img of document.querySelectorAll('img')) {
    const ds = img.getAttribute('data-src') || img.getAttribute('data-lazy-src')
            || img.getAttribute('data-original') || img.getAttribute('data-lazy')
            || img.getAttribute('data-url');
    if (ds && !ds.startsWith('data:') && img.src !== ds) { img.src = ds; n++; }
    try { img.loading = 'eager'; } catch (e) {}
  }
  return n;
}"""


def load_all_images(page, max_rounds=60, pause_ms=400):
    """Scroll the whole page and force lazy images to load, so every page image
    is fetched. Many manga readers keep off-screen pages as a data-URI
    placeholder with the real URL in data-src and only swap it when scrolled into
    view — and the placeholder reserves layout height, so a height-stable scroll
    stops early. We instead force every data-src to load and scroll to the bottom,
    stopping when the count of loaded images stops growing."""
    last_count, stable = -1, 0
    for _ in range(max_rounds):
        page.evaluate(_FORCE_LAZY_JS)
        page.evaluate("window.scrollBy(0, Math.round(window.innerHeight * 0.9))")
        page.wait_for_timeout(pause_ms)
        count = page.evaluate("[...document.querySelectorAll('img')].filter(i => i.naturalWidth > 1).length")
        at_bottom = page.evaluate(
            "(window.innerHeight + window.scrollY) >= (document.documentElement.scrollHeight - 50)"
        )
        if count == last_count and at_bottom:
            stable += 1
            if stable >= 2:
                break
        else:
            stable, last_count = 0, count
    page.evaluate(_FORCE_LAZY_JS)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    page.wait_for_timeout(800)


def collect_images(page, min_width, selector):
    """Image URLs in DOM (reading) order, deduped. Prefers a real http(s) URL
    (lazy images hide it in data-*); keeps images wide enough to be real pages,
    plus any whose size couldn't be measured but have a real remote URL."""
    return page.evaluate(
        """([minW, sel]) => {
          const seen = new Set(), out = [];
          for (const img of document.querySelectorAll(sel)) {
            const cands = [img.currentSrc, img.src, img.getAttribute('data-src'),
                           img.getAttribute('data-lazy-src'), img.getAttribute('data-original')];
            let raw = '';
            for (const c of cands) { if (c && !c.startsWith('data:')) { raw = c; break; } }
            if (!raw) continue;
            let abs; try { abs = new URL(raw, location.href).href; } catch { continue; }
            if (seen.has(abs)) continue;
            const w = img.naturalWidth || img.width || 0;
            if (w && w < minW) continue;   // measured and too small -> skip; unknown (0) -> keep
            seen.add(abs);
            out.push({ url: abs, w });
          }
          return out;
        }""",
        [min_width, selector],
    )


def ext_for(content_type, url):
    """Return a raster image extension, or None to skip (e.g. SVG / non-image)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _CTYPE_EXT:
        return _CTYPE_EXT[ct]
    if ct.startswith("image/svg") or ct in ("text/html", "application/xml"):
        return None  # vector / not a page image — skip
    suffix = Path(urlsplit(url).path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"):
        return suffix
    if suffix == ".svg":
        return None
    return ".jpg" if ct.startswith("image/") or not ct else None


def download(context, url, referer, dest_noext, retries=2):
    """Fetch via the browser context (shares cookies) with a referer to defeat
    hotlink checks; write bytes. Retries on HTTP 429 (rate limit) with backoff.
    Returns the saved Path, or None (skipped/failed)."""
    resp = None
    for attempt in range(retries + 1):
        try:
            resp = context.request.get(url, headers={"referer": referer}, timeout=30000)
        except Exception as e:
            print(f"      ! {url} -> {e}")
            return None
        if resp.status == 429 and attempt < retries:
            wait = 2 ** attempt
            print(f"      . 429 rate-limited, retrying in {wait}s")
            time.sleep(wait)
            continue
        break
    if not resp.ok:
        print(f"      ! {url} -> HTTP {resp.status}")
        return None
    ext = ext_for(resp.headers.get("content-type", ""), url)
    if ext is None:
        print(f"      - skip (non-raster) {url}")
        return None
    dest = dest_noext.with_suffix(ext)
    dest.write_bytes(resp.body())
    return dest


def scrape_url(context, url, out_root, min_width, limit, scroll_rounds, timeout, delay, selector):
    page = context.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    load_all_images(page, max_rounds=scroll_rounds)
    images = collect_images(page, min_width, selector)
    slug = page_slug(page, url)
    referer = page.url

    folder = out_root / slug
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n    -> {len(images)} image(s) >= {min_width}px, saving to {folder}/")

    saved = 0
    for i, item in enumerate(images, 1):
        if limit and saved >= limit:
            break
        dest = download(context, item["url"], referer, folder / f"{i:03d}")
        if dest:
            saved += 1
            print(f"      {dest.name}  ({item['w']}px)")
        if delay:
            time.sleep(delay)
    page.close()
    print(f"    done: {saved}/{len(images)} downloaded")
    return saved


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Scrape page images for an evaluation set.")
    ap.add_argument("urls", nargs="*", help="page URL(s) to scrape")
    ap.add_argument("--urls-file", help="text file with one URL per line (# comments allowed)")
    ap.add_argument("--out", default="eval/scraped", help="output root dir (default: eval/scraped)")
    ap.add_argument("--min-width", type=int, default=400, help="min image width px (default: 400)")
    ap.add_argument("--selector", default="img", help="CSS selector for images (e.g. 'img.comic-image' for precision; default: img)")
    ap.add_argument("--limit", type=int, default=0, help="max images per page (0 = all)")
    ap.add_argument("--scroll-rounds", type=int, default=60, help="max lazy-load scroll steps")
    ap.add_argument("--headful", action="store_true", help="show the browser window")
    ap.add_argument("--timeout", type=int, default=60000, help="navigation timeout ms")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between downloads (politeness; default 0.25)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    urls = list(args.urls)
    if args.urls_file:
        for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        sys.exit("No URLs given. Pass URL(s) or --urls-file.")

    out_root = Path(args.out)
    total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(user_agent=USER_AGENT)
        try:
            for url in urls:
                try:
                    total += scrape_url(context, url, out_root, args.min_width,
                                        args.limit, args.scroll_rounds, args.timeout,
                                        args.delay, args.selector)
                except Exception as e:
                    print(f"  ! failed {url}: {e}")
        finally:
            browser.close()

    print(f"\nTotal downloaded: {total} image(s) under {out_root}/")


if __name__ == "__main__":
    main()
