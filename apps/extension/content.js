// MangaTVL — content script: one floating panel. Once on, every manga page on the tab is
// translated as it loads (lazy readers too) until stopped. Chapter mode carries the story
// context page to page; page mode translates pages on their own, several at a time.
(() => {
  if (window.__mtvl) return; window.__mtvl = true;
  const MIN_W = 380;
  const LOOKAHEAD = 3;   // pages analysed ahead of the one being translated
  const PARALLEL = 4;    // page mode: LLM calls at once
  const DOWNLOAD_URL = "https://github.com/parwer/MangaTVL/releases/latest";
  const STEPS = 3;       // read the text, translate, write the translation in
  const GUESS_MS = { gpu: 15000, cpu: 30000 }; // time per page before the first one finishes
  const originals = new Map(); // img -> original src
  const queued = new Set();
  const stage = new Map();     // img -> step 0..3 of a page in flight
  let storyContext = null;      // chapter mode: returned by the backend, fed to the next page
  let on = false, checking = false, mode = "chapter", done = 0, failed = 0, cpuOnly = false;
  let runDone = 0, lastTick = 0; const intervals = []; // time per page, for the time left
  let notice = null; // {title, text, actions, tone}: why translating stopped or can't start
  let sites = {};

  const ICON_LOGO = `<svg viewBox="0 0 64 64" width="18" height="18" aria-hidden="true"><rect x="4" y="6" width="56" height="40" rx="14" fill="currentColor"/><path d="M14 40 10 60 30 45Z" fill="currentColor"/><rect x="14" y="17" width="36" height="6" rx="3" fill="var(--brand)"/><rect x="14" y="29" width="26" height="6" rx="3" fill="var(--brand)"/></svg>`;
  const ICON_UNDO = `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg>`;
  const CSS = `
    :host { all: initial; position: absolute; top: 0; left: 0; width: 0; height: 0; z-index: 2147483646;
      --brand: #e2334a; --brand-dark: #c4223a; --ok: #22a355; --warn: #d98a00; --bad: #d93636; --info: #3b82f6;
      --bg: rgba(22, 24, 29, .96); --fg: #f4f5f7; --muted: #b6bac3; --line: rgba(255,255,255,.12); }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
    [hidden] { display: none !important; }
    button { font: inherit; cursor: pointer; border: 0; }
    .fab { position: fixed; right: 20px; bottom: 20px; display: flex; flex-direction: column; align-items: flex-end;
      gap: 10px; color: var(--fg); font-size: 14px; line-height: 1.4; }
    .card { width: 300px; background: var(--bg); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px;
      box-shadow: 0 12px 32px rgba(0,0,0,.35); backdrop-filter: blur(8px); }
    .head { display: flex; align-items: center; gap: 8px; }
    .title { font-size: 15px; font-weight: 650; flex: 1; }
    .close { background: none; color: var(--muted); font-size: 14px; padding: 2px 4px; border-radius: 6px; align-self: flex-start; }
    .close:hover { color: var(--fg); background: rgba(255,255,255,.1); }
    .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; background: var(--info); }
    .card[data-tone=ok] .dot { background: var(--ok); }
    .card[data-tone=bad] .dot { background: var(--bad); }
    .card[data-tone=busy] .dot { background: var(--warn); animation: pulse 1.2s ease-in-out infinite; }
    .text { color: var(--muted); margin-top: 6px; font-size: 13px; }
    .text:empty { display: none; }
    .bar { height: 6px; border-radius: 99px; background: var(--line); margin-top: 10px; overflow: hidden; }
    .bar i { display: block; height: 100%; background: var(--brand); border-radius: 99px; transition: width .4s ease; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .actions:empty { display: none; }
    .act { padding: 7px 12px; border-radius: 8px; background: rgba(255,255,255,.1); color: var(--fg);
      font-size: 13px; font-weight: 600; text-decoration: none; }
    .act:hover { background: rgba(255,255,255,.18); }
    .act.primary { background: var(--brand); color: #fff; }
    .act.primary:hover { background: var(--brand-dark); }
    .buttons { display: flex; gap: 8px; align-items: center; }
    .go { display: flex; align-items: center; gap: 8px; height: 48px; padding: 0 22px 0 18px; border-radius: 999px;
      background: var(--brand); color: #fff; font-size: 16px; font-weight: 700; box-shadow: 0 6px 20px rgba(226,51,74,.45); }
    .go:hover { background: var(--brand-dark); }
    .go.stop { background: #2b2f38; box-shadow: 0 6px 20px rgba(0,0,0,.35); }
    .go.stop svg { display: none; }
    .undo { width: 44px; height: 44px; border-radius: 50%; background: var(--bg); color: var(--fg);
      display: grid; place-items: center; box-shadow: 0 6px 20px rgba(0,0,0,.3); border: 1px solid var(--line); }
    .undo:hover { background: #2b2f38; }
    button:focus-visible, a:focus-visible { outline: 3px solid #8ab4ff; outline-offset: 2px; }

    .label { position: absolute; width: max-content; max-width: 360px; padding: .5em .75em; border-radius: .6em; color: #fff;
      background: rgba(22,24,29,.88); box-shadow: 0 4px 14px rgba(0,0,0,.3); font-weight: 600; line-height: 1.3;
      pointer-events: none; transition: top .15s ease; }
    .label.wide { max-width: none; border-radius: 0; padding: .55em .7em; }
    .label .row { display: flex; align-items: center; gap: .5em; }
    .label .n { font-weight: 500; opacity: .75; font-size: .85em; margin-left: auto; padding-left: .5em; white-space: nowrap; }
    .label.ok { background: var(--ok); }
    .label.err { background: var(--bad); pointer-events: auto; }
    .label .retry { margin-left: auto; padding: .25em .7em; border-radius: .4em; background: #fff; color: var(--bad);
      font-weight: 700; font-size: .9em; }
    .mini { height: .3em; border-radius: 99px; background: rgba(255,255,255,.2); margin-top: .45em; overflow: hidden; }
    .mini i { display: block; height: 100%; background: var(--brand); transition: width .4s ease; }
    .spin { width: 1em; height: 1em; border-radius: 50%; border: .18em solid rgba(255,255,255,.3);
      border-top-color: #fff; animation: spin .8s linear infinite; flex: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse { 50% { opacity: .35; } }
    @media (prefers-reduced-motion: reduce) { .spin, .card .dot { animation: none; } .bar i, .mini i, .label { transition: none; } }
    @media (max-width: 480px) { .fab { right: 12px; bottom: 12px; } .card { width: calc(100vw - 24px); } }`;

  // ------------------------------------------------------------ UI (own shadow root: site CSS can't reach it)
  const host = document.createElement("mtvl-root");
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>${CSS}</style>
    <div class="labels"></div>
    <div class="fab" hidden>
      <div class="card" hidden>
        <div class="head"><span class="dot"></span><b class="title"></b><button class="close" title="Close" aria-label="Close">✕</button></div>
        <div class="bar" hidden><i></i></div>
        <div class="text"></div>
        <div class="actions"></div>
      </div>
      <div class="buttons">
        <button class="undo" title="Show the original pages" aria-label="Show the original pages">${ICON_UNDO}</button>
        <button class="go">${ICON_LOGO}<span>Translate</span></button>
      </div>
    </div>`;
  document.documentElement.appendChild(host);
  const $ = (sel) => root.querySelector(sel);
  const fab = $(".fab"), card = $(".card"), go = $(".go"), undo = $(".undo"), labels = $(".labels");

  chrome.storage.local.get("sites").then((v) => { sites = v.sites || {}; render(); });
  chrome.storage.onChanged.addListener((ch) => {
    if (ch.sites) { sites = ch.sites.newValue || {}; render(); }
    if (ch.contextMode) mode = ch.contextMode.newValue || "chapter"; // pages queued from now on
  });
  const send = (m) => chrome.runtime.sendMessage(m).catch((e) => ({ error: String(e) }));

  const isPage = (im) => im.complete && im.naturalWidth >= MIN_W && im.getBoundingClientRect().width >= MIN_W * 0.5;
  const top = (im) => im.getBoundingClientRect().top + scrollY;

  // ------------------------------------------------------------ errors in plain words, with the fix
  const openSettings = { label: "Open MangaTVL", fn: () => send({ type: "open-settings" }) };
  const download = (label) => ({ label, href: DOWNLOAD_URL, primary: true });
  const retryStart = { label: "Try again", fn: () => start() };
  const BLOCKING = {
    "no-backend": { title: "MangaTVL Companion isn't running",
      text: "Start MangaTVL Companion from the Start menu, then try again. Not installed yet? Download it once, it's free.",
      actions: [retryStart, download("Download the app")] },
    "old-backend": { title: "MangaTVL Companion needs an update",
      text: "This version of the app is too old for the extension. Install the latest version.",
      actions: [download("Download the update")] },
    "missing-openrouter": { title: "OpenRouter isn't connected",
      text: "You picked OpenRouter as your AI but haven't signed in yet.", actions: [openSettings] },
    "missing-gemini": { title: "Gemini API key is missing",
      text: "You picked Gemini as your AI but haven't added an API key yet.", actions: [openSettings] },
    "load-error": { title: "MangaTVL Companion couldn't start",
      text: "It couldn't load its models. Quit it from the tray icon and start it again. If that doesn't help, reinstall it.",
      actions: [retryStart, download("Reinstall")] },
    "builtin-missing": { title: "No translator available",
      text: "Couldn't reach Bing Translator. Check your internet connection, or connect an AI in MangaTVL (OpenRouter, Gemini or Local AI).",
      actions: [openSettings] },
    "builtin-pair": { title: "Language pair not supported",
      text: "Chrome's built-in translator can't translate between these two languages. Connect an AI in MangaTVL instead.",
      actions: [openSettings] },
    "builtin-click": { title: "One more click",
      text: "Chrome needs to download its translator first. Click Translate again.", actions: [] },
  };
  // one page failed: a short reason on its label, the full error in the tooltip
  function reason(e) {
    const s = String(e || "").toLowerCase();
    if (/job expired|404/.test(s)) return "it took too long";
    if (/image fetch/.test(s)) return "the site blocked the image";
    if (/401|unauthori|invalid api key|api key not valid|user not found/.test(s)) return "the AI rejected your key";
    if (/402|credit|insufficient/.test(s)) return "out of AI credits";
    if (/429|quota|rate limit|resource_exhausted/.test(s)) return "the AI is busy, wait a minute";
    if (/connection refused|connecterror|11434|1234/.test(s) && /local/.test(s)) return "Local AI isn't running";
    if (/timeout|timed out/.test(s)) return "the AI didn't answer";
    if (/extension context invalidated|receiving end does not exist|could not establish/.test(s)) return "reload this page";
    if (/failed to fetch|networkerror/.test(s)) return "lost the connection to the app";
    return "something went wrong";
  }

  // ------------------------------------------------------------ labels on the pages
  const STEP_TEXT = ["Waiting in line", "Reading the text…", "Translating…", "Writing it in…"];
  function badge(img, kind, text, step, err) {
    let b = img.__mtvlBadge;
    if (!kind) { if (b) b.remove(); img.__mtvlBadge = null; return; }
    if (!b) { b = document.createElement("div"); labels.appendChild(b); img.__mtvlBadge = b; b.__img = img; }
    b.className = `label ${kind}`;
    b.title = err ? `Error: ${err}` : "";
    const pct = step == null ? 0 : Math.round((step / STEPS) * 100);
    b.innerHTML = `<div class="row">${kind === "busy" ? '<span class="spin"></span>' : kind === "ok" ? "✓" : kind === "err" ? "⚠" : ""}
      <span class="t"></span>${step != null && step > 0 ? `<span class="n">Step ${step} of ${STEPS}</span>` : ""}
      ${kind === "err" ? '<button class="retry">Retry</button>' : ""}</div>
      ${kind === "busy" ? `<div class="mini"><i style="width:${pct}%"></i></div>` : ""}`;
    b.querySelector(".t").textContent = text;
    const retry = b.querySelector(".retry");
    if (retry) retry.onclick = () => { failed = Math.max(0, failed - 1); badge(img, ""); enqueue(img, true); };
    place(b);
  }
  // big enough to read on small images; stays in view on tall (webtoon) pages
  function place(b) {
    const im = b.__img;
    if (!im.isConnected) { b.remove(); im.__mtvlBadge = null; return; }
    const r = im.getBoundingClientRect();
    if (r.width === 0) { b.style.display = "none"; return; }
    b.style.display = "";
    const small = r.width < 320;
    b.classList.toggle("wide", small);
    b.style.fontSize = `${small ? 13 : Math.round(Math.min(18, Math.max(14, r.width / 45)))}px`;
    b.style.width = small ? `${Math.max(0, r.width)}px` : "";
    b.style.maxWidth = small ? "none" : `${Math.max(160, Math.min(360, r.width - 2 * 10))}px`;
    const inset = small ? 0 : 10, h = b.offsetHeight || 32;
    const y = Math.min(Math.max(r.top + inset, 10), Math.max(r.top + inset, r.bottom - h - inset));
    b.style.left = `${r.left + scrollX + inset}px`;
    b.style.top = `${y + scrollY}px`;
  }
  let placing = false;
  const placeAll = () => {
    if (placing) return; placing = true;
    requestAnimationFrame(() => { placing = false; for (const b of labels.children) place(b); });
  };
  addEventListener("scroll", placeAll, { passive: true, capture: true });
  addEventListener("resize", placeAll, { passive: true });
  setInterval(() => { if (labels.children.length) placeAll(); }, 700); // readers that move pages around

  function step(im, n, text) {
    stage.set(im, n);
    badge(im, n === 0 ? "wait" : "busy", text || STEP_TEXT[n], n);
    render();
  }

  // ------------------------------------------------------------ the floating panel
  function fmtTime(ms) {
    const s = Math.round(ms / 1000);
    if (s < 5) return "almost done";
    if (s < 60) return `about ${Math.max(5, Math.round(s / 5) * 5)} sec left`;
    const m = Math.round(s / 60);
    return `about ${m} min left`;
  }
  function perPage() {
    const last = intervals.slice(-5);
    return last.length ? last.reduce((a, b) => a + b, 0) / last.length : (cpuOnly ? GUESS_MS.cpu : GUESS_MS.gpu);
  }
  function render() {
    const left = queued.size;
    fab.hidden = !(sites[location.hostname] || on || checking || notice || left || done);
    go.querySelector("span").textContent = on || checking ? "Stop" : done ? "Translate more" : "Translate";
    go.classList.toggle("stop", on || checking);
    undo.hidden = !done && !originals.size;
    const set = (tone, title, text, actions, bar) => {
      card.hidden = false; card.dataset.tone = tone;
      $(".close").hidden = !notice || checking;
      $(".title").textContent = title; $(".text").textContent = text || "";
      $(".bar").hidden = bar == null;
      if (bar != null) $(".bar i").style.width = `${Math.round(bar * 100)}%`;
      const box = $(".actions"); box.replaceChildren();
      for (const a of actions || []) {
        const el = document.createElement(a.href ? "a" : "button");
        el.textContent = a.label; el.className = a.primary ? "act primary" : "act";
        if (a.href) { el.href = a.href; el.target = "_blank"; el.rel = "noopener"; } else el.onclick = a.fn;
        box.appendChild(el);
      }
    };
    if (checking) return set("busy", "Getting ready…", "Checking MangaTVL Companion and your AI.");
    if (notice) return set(notice.tone || "bad", notice.title, notice.text, notice.actions);
    if (on) {
      const total = runDone + left;
      let partial = 0;
      for (const im of queued) partial += (stage.get(im) || 0) / (STEPS + 1);
      const frac = total ? (runDone + partial) / total : 1;
      const failText = failed ? ` ${failed} page${failed > 1 ? "s" : ""} failed: press Retry on the red label.` : "";
      if (!left) return runDone
        ? set("ok", `${runDone} page${runDone > 1 ? "s" : ""} translated`,
          "New pages are translated as they load. Scroll down to load more." + failText)
        : set("busy", "Looking for manga pages…", "Pages are translated as soon as they load. Scroll to the chapter's pages." + failText);
      const remaining = Math.max(0, perPage() * left - (Date.now() - lastTick));
      return set("busy", `Translating · page ${Math.min(runDone + 1, total)} of ${total}`,
        `${fmtTime(remaining)}${cpuOnly ? " (no graphics card, so it's slower)" : ""}.${failText}`, null, frac);
    }
    if (done) return set("ok", `${done} page${done > 1 ? "s" : ""} translated`,
      failed ? `${failed} page${failed > 1 ? "s" : ""} failed: press Retry on the red label.` : "Press ↺ to see the original pages.");
    card.hidden = true;
  }
  setInterval(() => { if (on && queued.size) render(); }, 1000); // the time left counts down

  // ------------------------------------------------------------ reading the page images
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

  // Chrome built-in on-device Translator (no account, no key, offline). One per language pair:
  // create() needs a user click while its model still has to be downloaded, so start() warms it up.
  const ISO = { thai: "th", english: "en", japanese: "ja", korean: "ko", chinese: "zh", chinese_traditional: "zh-Hant",
               vietnamese: "vi", indonesian: "id", french: "fr", german: "de", spanish: "es", italian: "it",
               portuguese: "pt", russian: "ru" };
  const translators = {};
  function getTranslator(from, to) {
    const opts = { sourceLanguage: ISO[from] || "en", targetLanguage: ISO[to] || "th" };
    const key = `${opts.sourceLanguage}>${opts.targetLanguage}`;
    if (!translators[key]) {
      translators[key] = (async () => {
        if (!("Translator" in self)) throw new Error("builtin-missing");
        if ((await Translator.availability(opts)) === "unavailable") throw new Error("builtin-pair");
        try { return await Translator.create(opts); }
        catch (e) { throw new Error(e && e.name === "NotAllowedError" ? "builtin-click" : String(e && e.message || e)); }
      })();
      translators[key].catch(() => delete translators[key]); // try again next time
    }
    return translators[key];
  }
  async function builtinTranslate(texts, from, to) {
    const t = await getTranslator(from, to);
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

  function enqueue(im, force) {
    if (queued.has(im)) return;
    if (!queued.size) lastTick = Date.now(); // the clock for the time left starts with work
    queued.add(im);
    if (!originals.has(im)) originals.set(im, im.currentSrc || im.src);
    if (notice && !BLOCKING[notice.code]) notice = null;
    step(im, 0);
    let turn = null;
    if (mode === "chapter") { // reserve this page's LLM turn now, in reading order
      const prev = turnTail; let release; turnTail = new Promise((r) => (release = r));
      turn = { prev, release };
    }
    run(im, turn, force).finally(() => { queued.delete(im); stage.delete(im); render(); });
  }

  function fail(im, error) {
    if (BLOCKING[error]) { // same for every page: one message on the panel, not a red label per page
      badge(im, "");
      notice = { code: error, ...BLOCKING[error] };
      if (on) stop(); else render();
      return;
    }
    failed++;
    badge(im, "err", `Couldn't translate: ${reason(error)}`, null, error);
    render();
  }

  async function run(im, turn, force) {
    let turnHeld = !!turn, aheadHeld = false, job = null, fetched = false;
    const releaseTurn = () => { if (turnHeld) { turnHeld = false; turn.release(); } };
    const releaseAhead = () => { if (aheadHeld) { aheadHeld = false; ahead.give(); } };
    try {
      await ahead.take(); aheadHeld = true;
      try {
        if (!on && !force) { originals.delete(im); badge(im, ""); return; } // stopped while waiting
        step(im, 1);
        const src = originals.get(im);
        const a = await send({ type: "analyze", src, pageUrl: location.href,
          dataUrl: src.startsWith("blob:") || src.startsWith("data:") ? tryCanvas(im) : null });
        if (!a || a.error) return fail(im, a && a.error);
        if (!a.regions) { badge(im, ""); originals.delete(im); return; } // nothing to translate
        job = a.job;
        if (turn) { step(im, 1, "Waiting for the page before…"); await turn.prev; }
        else { await llm.take(); releaseAhead(); } // page mode: in the LLM, so analyse another page meanwhile
        if (!on && !force) { originals.delete(im); badge(im, ""); if (!turn) llm.give(); return; } // stopped while waiting
        let t;
        try {
          step(im, 2);
          t = await send({ type: "job", job, context: turn ? storyContext : null });
          if (t && t.needs_client_translation) {
            const out = await builtinTranslate(a.bubbles.map((b) => b.text), a.from_lang, a.to_lang);
            fetched = true; // /v1/render answers with the page and frees the job
            step(im, 3);
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
      step(im, 3);
      fetched = true;
      const r = await send({ type: "result", job });
      if (!r || r.error) return fail(im, r && r.error);
      if (r.image) swap(im, r.image); else badge(im, "");
    } catch (e) {
      fail(im, String(e && e.message || e));
    } finally {
      releaseTurn();
      // failed after analysis: fetch the result anyway so the app frees the page (else it holds it 30 min)
      if (job && !fetched) send({ type: "result", job });
    }
  }

  function swap(im, image) {
    done++; runDone++;
    const now = Date.now();
    intervals.push(now - lastTick); lastTick = now;
    im.__mtvlOut = image;
    im.removeAttribute("srcset");
    im.src = image;
    badge(im, "ok", "Translated");
    setTimeout(() => { if (im.__mtvlBadge && im.__mtvlBadge.classList.contains("ok")) badge(im, ""); }, 1800);
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

  // ------------------------------------------------------------ start / stop
  // Is everything in place? Says what to fix instead of failing page by page.
  async function ready() {
    const st = await send({ type: "status" });
    const b = st && st.backend;
    if (!b) return "no-backend";
    if (b.kind !== "companion" || !b.info.jobs) return "old-backend";
    if (b.info.load_error) return "load-error";
    if (b.info.ready === false) return "loading";
    if (st.missing) return `missing-${st.missing}`;
    cpuOnly = !!b.info.cpu_only;
    return "ok";
  }

  let waitTimer = null;
  async function start() {
    clearTimeout(waitTimer);
    const s = await send({ type: "settings" });
    mode = s.contextMode || "chapter";
    notice = null; checking = true; render();
    const r = await ready();
    if (!checking) return; // stopped while checking
    checking = false;
    if (r === "loading") { // the app is starting: wait for it, then go on by itself
      notice = { code: "loading", tone: "busy", title: "MangaTVL Companion is starting",
        text: "It's loading its models, this takes a few seconds. Translating starts by itself.",
        actions: [{ label: "Cancel", fn: () => { clearTimeout(waitTimer); notice = null; render(); } }] };
      render();
      waitTimer = setTimeout(() => { if (notice && notice.code === "loading") start(); }, 3000);
      return;
    }
    if (r !== "ok") { notice = { code: r, ...BLOCKING[r] }; render(); return; }
    on = true; failed = 0; runDone = 0; intervals.length = 0; lastTick = Date.now();
    observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
    document.addEventListener("load", onLoad, true);
    scan(); render();
  }

  function stop() {
    on = false; checking = false;
    observer.disconnect();
    document.removeEventListener("load", onLoad, true);
    render();
  }

  function revertAll() {
    stop();
    notice = null;
    for (const [img, src] of originals) if (!queued.has(img)) { img.src = src; badge(img, ""); }
    for (const img of [...originals.keys()]) if (!queued.has(img)) originals.delete(img);
    storyContext = null; done = 0; failed = 0; runDone = 0;
    render();
  }

  go.onclick = () => (on || checking ? stop() : start());
  undo.onclick = revertAll;
  $(".close").onclick = () => { clearTimeout(waitTimer); notice = null; render(); };
  chrome.runtime.onMessage.addListener((m, _sender, respond) => {
    if (m.type === "state") { respond({ on: on || checking, done, left: queued.size, failed }); return; }
    if (m.type === "stop") { stop(); respond({ ok: true }); return; }
    if (m.type === "translate-all" && !on && !checking) start();
    if (m.type === "translate-one") {
      const im = [...document.images].find((x) => (x.currentSrc || x.src) === m.src);
      if (im && !queued.has(im) && im.src !== im.__mtvlOut) { // not already translated
        (async () => {
          if (!on) mode = (await send({ type: "settings" })).contextMode || "chapter";
          const r = await ready(); // one image: still say what to fix first
          if (r === "ok") enqueue(im, true);
          else { notice = r === "loading"
            ? { code: r, tone: "busy", title: "MangaTVL Companion is starting", text: "Wait a few seconds, then try again.", actions: [] }
            : { code: r, ...BLOCKING[r] }; render(); }
        })();
      }
    }
  });
})();
