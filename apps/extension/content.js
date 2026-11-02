// MangaTVL — content script: one floating button. Once on, every manga page on the tab is
// translated as it loads (lazy readers too) until stopped. Chapter mode carries the story
// context page to page; page mode translates pages on their own, several at a time.
(() => {
  if (window.__mtvl) return; window.__mtvl = true;
  const MIN_W = 380;
  const LOOKAHEAD = 3;   // pages analysed ahead of the one being translated
  const PARALLEL = 4;    // page mode: LLM calls at once
  const DOWNLOAD_URL = "https://github.com/parwer/MangaTVL/releases/latest";
  const originals = new Map(); // img -> original src
  const queued = new Set();
  let storyContext = null;      // chapter mode: returned by the backend, fed to the next page
  let on = false, mode = "chapter", done = 0, failed = 0;

  const fab = document.createElement("div");
  fab.className = "mtvl-fab";
  fab.innerHTML = `<button class="mtvl-go" title="Translate pages">แปล</button><button class="mtvl-undo" title="Revert">↺</button><span class="mtvl-msg"></span>`;
  fab.hidden = true; // shown only on sites switched on in the popup
  document.documentElement.appendChild(fab);
  const showFab = (sites) => { fab.hidden = !(sites || {})[location.hostname] && !on; };
  chrome.storage.local.get("sites").then((v) => showFab(v.sites));
  chrome.storage.onChanged.addListener((ch) => {
    if (ch.sites) showFab(ch.sites.newValue);
    if (ch.contextMode) mode = ch.contextMode.newValue || "chapter"; // pages queued from now on
  });
  const go = fab.querySelector(".mtvl-go");
  const msg = (t) => (fab.querySelector(".mtvl-msg").textContent = t || "");
  const send = (m) => chrome.runtime.sendMessage(m).catch((e) => ({ error: String(e) }));

  const isPage = (im) => im.complete && im.naturalWidth >= MIN_W && im.getBoundingClientRect().width >= MIN_W * 0.5;
  const top = (im) => im.getBoundingClientRect().top + scrollY;

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
  async function builtinTranslate(texts, from, to) {
    if (!("Translator" in self)) throw new Error("Chrome built-in Translator API not available");
    const iso = { thai: "th", english: "en", japanese: "ja", korean: "ko", chinese: "zh" };
    const t = await Translator.create({ sourceLanguage: iso[from] || "en", targetLanguage: iso[to] || "th" });
    return Promise.all(texts.map((x) => t.translate(x)));
  }

  function slots(n) {
    let free = n; const wait = [];
    return {
      take: () => (free > 0 ? (free--, Promise.resolve()) : new Promise((r) => wait.push(r))),
      give: () => { const r = wait.shift(); r ? r() : free++; },
    };
  }
  const ahead = slots(LOOKAHEAD), llm = slots(PARALLEL);
  let turnTail = Promise.resolve(); // chapter mode: LLM turns in reading order

  function progress() {
    const left = queued.size;
    msg(on ? `แปลแล้ว ${done} หน้า${left ? ` · เหลือ ${left}` : " · รอหน้าใหม่"}${failed ? ` · ผิดพลาด ${failed}` : ""}`
           : done ? `แปลแล้ว ${done} หน้า` : "");
  }

  function enqueue(im, force) {
    if (queued.has(im)) return;
    queued.add(im);
    if (!originals.has(im)) originals.set(im, im.currentSrc || im.src);
    badge(im, "รอคิว", "#555");
    let turn = null;
    if (mode === "chapter") { // reserve this page's LLM turn now, in reading order
      const prev = turnTail; let release; turnTail = new Promise((r) => (release = r));
      turn = { prev, release };
    }
    run(im, turn, force).finally(() => { queued.delete(im); progress(); });
    progress();
  }

  function fail(im, error) {
    failed++;
    badge(im, error === "no-backend" ? "ยังไม่ได้เปิดแอป MangaTVL" : error === "old-backend"
      ? "อัปเดตแอป MangaTVL Companion" : `ผิดพลาด: ${error}`, "#b22");
    if (error === "no-backend" || error === "old-backend") {
      stop();
      const a = document.createElement("a"); // the app is a one-time install: link to it from the button
      a.href = DOWNLOAD_URL; a.target = "_blank"; a.className = "mtvl-get";
      a.textContent = error === "no-backend" ? "เปิดแอป MangaTVL หรือดาวน์โหลด" : "ดาวน์โหลดแอปเวอร์ชันใหม่";
      fab.querySelector(".mtvl-msg").replaceChildren(a);
    }
  }

  async function run(im, turn, force) {
    let turnHeld = !!turn, aheadHeld = false, job = null;
    const releaseTurn = () => { if (turnHeld) { turnHeld = false; turn.release(); } };
    const releaseAhead = () => { if (aheadHeld) { aheadHeld = false; ahead.give(); } };
    try {
      await ahead.take(); aheadHeld = true;
      try {
        if (!on && !force) { originals.delete(im); badge(im, ""); return; } // stopped while waiting
        badge(im, "กำลังอ่านภาพ…", "#c70");
        const src = originals.get(im);
        const a = await send({ type: "analyze", src, pageUrl: location.href,
          dataUrl: src.startsWith("blob:") || src.startsWith("data:") ? tryCanvas(im) : null });
        if (!a || a.error) return fail(im, a && a.error);
        if (!a.regions) { badge(im, ""); originals.delete(im); return; } // nothing to translate
        job = a.job;
        if (turn) await turn.prev;
        else { await llm.take(); releaseAhead(); } // page mode: in the LLM, so analyse another page meanwhile
        let t;
        try {
          badge(im, "กำลังแปล…", "#c70");
          t = await send({ type: "job", job, context: turn ? storyContext : null });
          if (t && t.needs_client_translation) {
            const out = await builtinTranslate(a.bubbles.map((b) => b.text), a.from_lang, a.to_lang);
            t = await send({ type: "render", payload: { job, translations: out } });
            if (t && t.image) return swap(im, t.image);
          }
          if (!t || t.error) return fail(im, t && t.error);
          if (turn && t.context) storyContext = t.context;
        } finally {
          if (turn) releaseTurn(); else llm.give();
        }
      } finally {
        releaseAhead();
      }
      badge(im, "กำลังวางข้อความ…", "#c70");
      const r = await send({ type: "result", job });
      if (!r || r.error) return fail(im, r && r.error);
      if (r.image) swap(im, r.image); else badge(im, "");
    } catch (e) {
      fail(im, String(e && e.message || e));
    } finally {
      releaseTurn();
    }
  }

  function swap(im, image) {
    done++;
    im.__mtvlOut = image;
    im.removeAttribute("srcset");
    im.src = image;
    badge(im, "✓", "#2a2"); setTimeout(() => badge(im, ""), 1500);
  }

  function scan() {
    if (!on) return;
    [...document.images].filter((im) => !queued.has(im) && !originals.has(im) && isPage(im))
      .sort((a, b) => top(a) - top(b)).forEach((im) => enqueue(im));
  }

  // Lazy readers load pages as you scroll; single-page readers swap the src of one <img>.
  const observer = new MutationObserver((muts) => {
    for (const m of muts) {
      const im = m.target;
      if (m.type === "attributes" && im.tagName === "IMG" && originals.has(im) && !queued.has(im)
          && im.src !== im.__mtvlOut && im.src !== originals.get(im)) {
        originals.delete(im); // the reader put a new page in this <img>
      }
    }
    scan();
  });
  const onLoad = (e) => { if (e.target.tagName === "IMG") scan(); };

  async function start() {
    mode = (await send({ type: "settings" })).contextMode || "chapter";
    on = true; failed = 0; go.textContent = "หยุด"; fab.hidden = false;
    observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    document.addEventListener("load", onLoad, true);
    scan(); progress();
  }

  function stop() {
    on = false; go.textContent = "แปล";
    chrome.storage.local.get("sites").then((v) => showFab(v.sites));
    observer.disconnect();
    document.removeEventListener("load", onLoad, true);
    progress();
  }

  function revertAll() {
    stop();
    for (const [img, src] of originals) if (!queued.has(img)) { img.src = src; badge(img, ""); }
    for (const img of [...originals.keys()]) if (!queued.has(img)) originals.delete(img);
    storyContext = null; done = 0; failed = 0; msg("");
  }

  go.onclick = () => (on ? stop() : start());
  fab.querySelector(".mtvl-undo").onclick = revertAll;
  chrome.runtime.onMessage.addListener(async (m) => {
    if (m.type === "translate-all" && !on) start();
    if (m.type === "translate-one") {
      const im = [...document.images].find((x) => (x.currentSrc || x.src) === m.src);
      if (im && !queued.has(im) && im.src !== im.__mtvlOut) { // not already translated
        if (!on) mode = (await send({ type: "settings" })).contextMode || "chapter";
        enqueue(im, true);
      }
    }
  });
})();
