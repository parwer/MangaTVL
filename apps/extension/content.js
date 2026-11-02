// MangaTVL — content script: one floating button, translate pages in reading
// order, carry a running "story context" from page to page so names/tone stay
// consistent across a chapter, swap images in place, revert on demand.
(() => {
  if (window.__mtvl) return; window.__mtvl = true;
  const MIN_W = 380;
  const originals = new Map(); // img -> original src
  let storyContext = null;      // returned by backend, fed to next page
  let busy = false;

  const fab = document.createElement("div");
  fab.className = "mtvl-fab";
  fab.innerHTML = `<button class="mtvl-go" title="Translate pages">แปล</button><button class="mtvl-undo" title="Revert">↺</button><span class="mtvl-msg"></span>`;
  document.documentElement.appendChild(fab);
  const msg = (t) => (fab.querySelector(".mtvl-msg").textContent = t || "");

  const pages = () => [...document.images]
    .filter((im) => im.naturalWidth >= MIN_W && im.getBoundingClientRect().width >= MIN_W * 0.5)
    .sort((a, b) => (a.getBoundingClientRect().top + scrollY) - (b.getBoundingClientRect().top + scrollY));

  function badge(img, text, color) {
    let b = img.__mtvlBadge;
    if (!b) {
      b = document.createElement("div"); b.className = "mtvl-badge";
      document.documentElement.appendChild(b); img.__mtvlBadge = b;
    }
    const r = img.getBoundingClientRect();
    b.style.left = `${r.left + scrollX + 8}px`; b.style.top = `${r.top + scrollY + 8}px`;
    b.style.background = color; b.textContent = text;
    if (!text) b.remove(), (img.__mtvlBadge = null);
  }

  // Same-origin / CORS-enabled images can be read straight from the DOM (fastest,
  // works with blob: URLs readers use); otherwise the service worker fetches it.
  function tryCanvas(img) {
    try {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      return c.toDataURL("image/jpeg", 0.95);
    } catch (_) { return null; } // tainted canvas
  }

  // Chrome built-in on-device Translator (no account, no key, offline).
  // Used only when the user picks "builtin" or nothing else is available.
  async function builtinTranslate(texts, from, to) {
    if (!("Translator" in self)) throw new Error("Chrome built-in Translator API not available");
    const iso = { thai: "th", english: "en", japanese: "ja", korean: "ko", chinese: "zh" };
    const t = await Translator.create({ sourceLanguage: iso[from] || "en", targetLanguage: iso[to] || "th" });
    return Promise.all(texts.map((x) => t.translate(x)));
  }

  async function translateOne(img) {
    if (!originals.has(img)) originals.set(img, img.currentSrc || img.src);
    badge(img, "กำลังแปล…", "#c70");
    const src = originals.get(img);
    let res = await chrome.runtime.sendMessage({
      type: "translate", src, dataUrl: src.startsWith("blob:") || src.startsWith("data:") ? tryCanvas(img) : null,
      pageUrl: location.href, prevContext: storyContext,
    });
    if (res && res.needs_client_translation) { // builtin flow: backend did detect/OCR/inpaint, we translate
      const out = await builtinTranslate(res.bubbles.map((b) => b.text), res.from_lang, res.to_lang);
      res = await chrome.runtime.sendMessage({ type: "render", payload: { job: res.job, translations: out } });
    }
    if (!res || res.error) {
      badge(img, res && res.error === "no-backend" ? "ยังไม่ได้เปิดแอป MangaTVL" : `ผิดพลาด: ${res && res.error}`, "#b22");
      if (res && res.error === "no-backend") showInstallHint();
      return false;
    }
    if (res.context) storyContext = res.context;
    img.removeAttribute("srcset");
    img.src = res.image;
    badge(img, "✓", "#2a2"); setTimeout(() => badge(img, ""), 1500);
    return true;
  }

  function showInstallHint() {
    msg("ติดตั้ง/เปิดแอป MangaTVL Companion ก่อน (ครั้งเดียว)");
  }

  async function translateAll(only) {
    if (busy) return; busy = true;
    const list = only ? [only] : pages().filter((im) => !originals.has(im));
    msg(`${list.length} หน้า`);
    let i = 0;
    for (const im of list) { // sequential on purpose: page N's context feeds page N+1
      msg(`${++i}/${list.length}`);
      const ok = await translateOne(im);
      if (!ok) break;
    }
    busy = false; if (i === list.length) msg("เสร็จ");
  }

  function revertAll() {
    for (const [img, src] of originals) img.src = src;
    originals.clear(); storyContext = null; msg("");
  }

  fab.querySelector(".mtvl-go").onclick = () => translateAll();
  fab.querySelector(".mtvl-undo").onclick = revertAll;
  chrome.runtime.onMessage.addListener((m) => {
    if (m.type === "translate-all") translateAll();
    if (m.type === "translate-one") {
      const im = [...document.images].find((x) => (x.currentSrc || x.src) === m.src);
      if (im) translateAll(im);
    }
  });
})();
