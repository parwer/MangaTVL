// ==UserScript==
// @name         MangaTVL — tempomunkey translator
// @namespace    mangatvl
// @version      0.1
// @description  Pick manga page images (or all) on a reader site and translate them via the MangaTVL API.
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @connect      localhost
// @connect      *
// @run-at       document-idle
// ==/UserScript==
//
// SETUP:
//   1. Run the MangaTVL API locally (docker compose up  OR  uvicorn main:app --port 8000)
//      and set a provider key in .env (otherwise text comes back untranslated).
//   2. Edit API_BASE below if the server is elsewhere.
//   3. (Recommended) narrow @match above to the tempomunkey domain instead of *://*/*.
//
// USAGE: open a chapter -> tick the checkbox on the pages you want (or "Select all")
//        -> optionally enable "Visual context" -> "Translate selected". "Revert all" undoes it.

(function () {
  "use strict";

  // ----- config -----
  const API_BASE = "http://localhost:8000";
  const MIN_WIDTH = 400;          // only offer images at least this wide (skip icons/banners)
  const REQUEST_TIMEOUT = 180000; // ms

  // ----- persisted settings (provider / model / target language / API key) -----
  const gmGet = (k, d) => (typeof GM_getValue === "function" ? GM_getValue(k, d) : d);
  const gmSet = (k, v) => { if (typeof GM_setValue === "function") GM_setValue(k, v); };
  const settings = {
    provider: gmGet("mtvl_provider", ""),   // "" = use server default
    model: gmGet("mtvl_model", ""),
    from_lang: gmGet("mtvl_from_lang", ""), // source language (e.g. "japanese" -> manga-ocr)
    to_lang: gmGet("mtvl_to_lang", ""),
    font: gmGet("mtvl_font", ""),           // font key from GET /fonts/ ("" = per-language default)
    text_scale: gmGet("mtvl_text_scale", ""), // >1 = bigger text ("" = server default)
    upscale: gmGet("mtvl_upscale", ""),     // output upscale factor ("" / 1 = off)
    upscaler: gmGet("mtvl_upscaler", ""),   // "" = lanczos (fast) | "realesrgan" (AI)
    api_key: gmGet("mtvl_api_key", ""),
  };

  // ----- styles -----
  const css = `
    .mtvl-cb { position:absolute; z-index:2147483600; width:22px; height:22px; cursor:pointer;
               accent-color:#e23; background:#fff; border-radius:4px; box-shadow:0 0 0 2px #0006; }
    .mtvl-badge { position:absolute; z-index:2147483600; font:600 12px/1.4 sans-serif; color:#fff;
                  padding:2px 6px; border-radius:4px; pointer-events:none; }
    .mtvl-bar { position:fixed; right:16px; bottom:16px; z-index:2147483601;
                background:#1b1b1f; color:#eee; font:13px/1.4 sans-serif; padding:12px;
                border-radius:10px; box-shadow:0 6px 24px #0008; width:230px; }
    .mtvl-bar h4 { margin:0 0 8px; font-size:13px; color:#9cf; }
    .mtvl-bar button { width:100%; margin:3px 0; padding:7px; border:0; border-radius:6px;
                       background:#333; color:#eee; cursor:pointer; font-size:13px; }
    .mtvl-bar button.primary { background:#e23; color:#fff; font-weight:700; }
    .mtvl-bar label { display:flex; align-items:center; gap:6px; margin:6px 0; }
    .mtvl-row { display:flex; gap:6px; }
    .mtvl-row button { flex:1; }
    .mtvl-status { margin-top:8px; font-size:12px; color:#bbb; min-height:16px; }
    .mtvl-bar details { margin:6px 0; }
    .mtvl-bar summary { cursor:pointer; color:#9cf; font-size:12px; }
    .mtvl-bar input[type=text], .mtvl-bar input[type=password], .mtvl-bar select {
      width:100%; margin:3px 0; padding:5px; border:0; border-radius:5px;
      background:#2a2a30; color:#eee; font-size:12px; box-sizing:border-box; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  // ----- state -----
  const overlays = new Map(); // img -> {cb, badge}

  function absolutize(src) {
    try { return new URL(src, location.href).href; } catch { return src; }
  }
  function imgUrl(img) {
    return absolutize(img.currentSrc || img.src || img.dataset.src || "");
  }

  function placeOverlay(img) {
    if (overlays.has(img)) return;
    const r = img.getBoundingClientRect();
    if (r.width < MIN_WIDTH) return;

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "mtvl-cb";
    const badge = document.createElement("span");
    badge.className = "mtvl-badge";
    badge.style.display = "none";

    document.body.appendChild(cb);
    document.body.appendChild(badge);
    overlays.set(img, { cb, badge });
    if (!img.dataset.mtvlOriginal) img.dataset.mtvlOriginal = imgUrl(img);

    reposition(img);
  }

  function reposition(img) {
    const o = overlays.get(img);
    if (!o) return;
    const r = img.getBoundingClientRect();
    const top = r.top + window.scrollY, left = r.left + window.scrollX;
    o.cb.style.top = top + 6 + "px";
    o.cb.style.left = left + 6 + "px";
    o.badge.style.top = top + 6 + "px";
    o.badge.style.left = left + 34 + "px";
  }
  function repositionAll() { overlays.forEach((_, img) => reposition(img)); }

  function setBadge(img, text, color) {
    const o = overlays.get(img);
    if (!o) return;
    o.badge.textContent = text;
    o.badge.style.background = color;
    o.badge.style.display = text ? "block" : "none";
  }

  function scanImages() {
    document.querySelectorAll("img").forEach((img) => {
      if ((img.naturalWidth || img.width) >= MIN_WIDTH) placeOverlay(img);
    });
    repositionAll();
    updateCount();
  }

  function selectedImgs() {
    return [...overlays.entries()].filter(([, o]) => o.cb.checked).map(([img]) => img);
  }

  // ----- floating bar -----
  const bar = document.createElement("div");
  bar.className = "mtvl-bar";
  bar.innerHTML = `
    <h4>MangaTVL</h4>
    <div class="mtvl-row">
      <button id="mtvl-all">Select all</button>
      <button id="mtvl-clear">Clear</button>
    </div>
    <label><input type="checkbox" id="mtvl-proc"> Visual context (slower, better)</label>
    <button id="mtvl-go" class="primary">Translate selected</button>
    <button id="mtvl-stop" disabled>Interrupt</button>
    <button id="mtvl-revert">Revert all</button>
    <button id="mtvl-rescan">Rescan images</button>
    <details>
      <summary>Settings</summary>
      <select id="mtvl-provider" title="provider">
        <option value="">provider: server default</option>
        <option value="openrouter">openrouter</option>
        <option value="gemini">gemini</option>
        <option value="groq">groq</option>
      </select>
      <input type="text" id="mtvl-model" placeholder="model (blank = default)">
      <input type="text" id="mtvl-fromlang" placeholder="source language (e.g. japanese)">
      <input type="text" id="mtvl-tolang" placeholder="target language (e.g. thai)">
      <select id="mtvl-font" title="font">
        <option value="">font: auto (per language)</option>
      </select>
      <input type="number" id="mtvl-scale" min="0.5" max="2" step="0.1" placeholder="text size (e.g. 1.2)">
      <input type="number" id="mtvl-upscale" min="1" max="4" step="0.5" placeholder="upscale image (e.g. 2)">
      <select id="mtvl-upscaler" title="upscaler">
        <option value="">upscale: fast (lanczos)</option>
        <option value="realesrgan">upscale: AI (real-esrgan, slow)</option>
      </select>
      <input type="password" id="mtvl-key" placeholder="API key (blank = server env)">
    </details>
    <div class="mtvl-status" id="mtvl-status"></div>
  `;
  document.body.appendChild(bar);

  // populate + persist settings
  const provEl = bar.querySelector("#mtvl-provider");
  const modelEl = bar.querySelector("#mtvl-model");
  const fromlangEl = bar.querySelector("#mtvl-fromlang");
  const tolangEl = bar.querySelector("#mtvl-tolang");
  const fontEl = bar.querySelector("#mtvl-font");
  const scaleEl = bar.querySelector("#mtvl-scale");
  const upscaleEl = bar.querySelector("#mtvl-upscale");
  const upscalerEl = bar.querySelector("#mtvl-upscaler");
  const keyEl = bar.querySelector("#mtvl-key");
  provEl.value = settings.provider; modelEl.value = settings.model;
  fromlangEl.value = settings.from_lang; tolangEl.value = settings.to_lang; keyEl.value = settings.api_key;
  scaleEl.value = settings.text_scale;
  upscaleEl.value = settings.upscale; upscalerEl.value = settings.upscaler;
  provEl.onchange = () => gmSet("mtvl_provider", (settings.provider = provEl.value));
  modelEl.onchange = () => gmSet("mtvl_model", (settings.model = modelEl.value.trim()));
  fromlangEl.onchange = () => gmSet("mtvl_from_lang", (settings.from_lang = fromlangEl.value.trim()));
  tolangEl.onchange = () => gmSet("mtvl_to_lang", (settings.to_lang = tolangEl.value.trim()));
  fontEl.onchange = () => gmSet("mtvl_font", (settings.font = fontEl.value));
  scaleEl.onchange = () => gmSet("mtvl_text_scale", (settings.text_scale = scaleEl.value.trim()));
  upscaleEl.onchange = () => gmSet("mtvl_upscale", (settings.upscale = upscaleEl.value.trim()));
  upscalerEl.onchange = () => gmSet("mtvl_upscaler", (settings.upscaler = upscalerEl.value));
  keyEl.onchange = () => gmSet("mtvl_api_key", (settings.api_key = keyEl.value.trim()));

  // Populate the font picker from the server so it stays in sync with fonts.json.
  GM_xmlhttpRequest({
    method: "GET",
    url: API_BASE + "/fonts/",
    timeout: 15000,
    onload: (res) => {
      let fonts;
      try { fonts = (JSON.parse(res.responseText) || {}).fonts || []; } catch { return; }
      for (const f of fonts) {
        const opt = document.createElement("option");
        opt.value = f.key;
        opt.textContent = f.scripts && f.scripts.length ? `${f.label} [${f.scripts.join("/")}]` : f.label;
        fontEl.appendChild(opt);
      }
      fontEl.value = settings.font;  // restore saved choice once options exist
    },
  });

  const statusEl = bar.querySelector("#mtvl-status");
  function setStatus(msg, color) { statusEl.textContent = msg; statusEl.style.color = color || "#bbb"; }
  function updateCount() {
    const total = overlays.size, sel = selectedImgs().length;
    if (!statusEl.textContent.startsWith("translat")) setStatus(`${sel}/${total} selected`);
  }

  bar.querySelector("#mtvl-all").onclick = () => { overlays.forEach((o) => (o.cb.checked = true)); updateCount(); };
  bar.querySelector("#mtvl-clear").onclick = () => { overlays.forEach((o) => (o.cb.checked = false)); updateCount(); };
  bar.querySelector("#mtvl-rescan").onclick = scanImages;
  bar.querySelector("#mtvl-revert").onclick = () => {
    overlays.forEach((o, img) => {
      if (img.dataset.mtvlOriginal) img.src = img.dataset.mtvlOriginal;
      setBadge(img, "", "");
    });
    setStatus("reverted");
  };

  const goBtn = bar.querySelector("#mtvl-go");
  const stopBtn = bar.querySelector("#mtvl-stop");
  let activeReq = null;          // in-flight GM_xmlhttpRequest handle (for interrupt)
  let activeImgs = null;         // images being translated (to clear their badges on abort)

  function setBusy(busy) {
    goBtn.disabled = busy;
    stopBtn.disabled = !busy;
  }

  goBtn.onclick = () => {
    const imgs = selectedImgs();
    if (!imgs.length) { setStatus("nothing selected", "#fa6"); return; }
    const process_image = bar.querySelector("#mtvl-proc").checked;
    const urls = imgs.map(imgUrl);
    imgs.forEach((img) => setBadge(img, "…", "#06c"));
    setStatus(`translating ${imgs.length} image(s)…`, "#9cf");
    activeImgs = imgs;
    setBusy(true);

    // include only the settings the user filled in (server falls back to its defaults/env)
    const payload = { images: urls, process_image };
    if (settings.provider) payload.provider = settings.provider;
    if (settings.model) payload.model = settings.model;
    if (settings.from_lang) payload.from_lang = settings.from_lang;
    if (settings.to_lang) payload.to_lang = settings.to_lang;
    if (settings.font) payload.font = settings.font;
    if (settings.text_scale) payload.text_scale = parseFloat(settings.text_scale);
    if (settings.upscale && parseFloat(settings.upscale) > 1) {
      payload.upscale = parseFloat(settings.upscale);
      if (settings.upscaler) payload.upscaler = settings.upscaler;
    }
    if (settings.api_key) payload.api_key = settings.api_key;

    const total = imgs.length;
    let done = 0, fail = 0;
    let buf = "";        // unparsed tail (lines may arrive split across chunks)
    let prevLen = 0;     // for onprogress, which gives the *accumulated* text
    let usedStream = false, finished = false;

    // The server streams NDJSON: one {index, image} line per page as it finishes.
    // We feed it whatever transport delivers (stream chunks or accumulated text)
    // into a buffer and apply each complete line the moment it arrives.
    function applyLine(line) {
      let msg;
      try { msg = JSON.parse(line); } catch { return; }
      const img = imgs[msg.index];
      if (!img) return;
      if (msg.image) { img.src = msg.image; setBadge(img, "✓", "#2a2"); done++; }
      else { setBadge(img, "✕", "#a33"); fail++; }
      setStatus(`translating… ${done + fail}/${total} (${done} ok, ${fail} failed)`, "#9cf");
    }
    function pushText(delta) {
      if (!delta) return;
      buf += delta;
      let nl;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) applyLine(line);
      }
    }
    function finish() {
      if (finished) return;
      finished = true;
      if (buf.trim()) applyLine(buf.trim());   // flush any tail with no trailing newline
      setStatus(`done: ${done} ok, ${fail} failed`, fail ? "#fa6" : "#6d6");
      setBusy(false); activeReq = null;
    }

    activeReq = GM_xmlhttpRequest({
      method: "POST",
      url: API_BASE + "/translate/stream/",
      headers: { "Content-Type": "application/json" },
      data: JSON.stringify(payload),
      timeout: REQUEST_TIMEOUT,
      // Preferred path: real incremental streaming. Tampermonkey (>=5) hands back a
      // ReadableStream in onloadstart when responseType is "stream"; many builds
      // otherwise buffer onprogress until the whole response is in, which made
      // pages swap only at the end. We fall back to onprogress/onload if needed.
      responseType: "stream",
      onloadstart: (res) => {
        const stream = res && res.response;
        if (!stream || typeof stream.getReader !== "function") return; // unsupported -> fallback below
        usedStream = true;
        const reader = stream.getReader();
        const dec = new TextDecoder();
        (function pump() {
          reader.read().then(({ done: rdone, value }) => {
            if (rdone) { finish(); return; }
            pushText(dec.decode(value, { stream: true }));
            pump();
          }).catch(() => { /* aborted/errored — onabort/onerror handle the UI */ });
        })();
      },
      onprogress: (res) => {
        if (usedStream) return;
        const t = res.responseText;
        if (t && t.length > prevLen) { pushText(t.slice(prevLen)); prevLen = t.length; }
      },
      onload: (res) => {
        if (usedStream) return;            // stream path finishes via reader 'done'
        const t = res.responseText || "";
        if (t.length > prevLen) { pushText(t.slice(prevLen)); prevLen = t.length; }
        finish();
      },
      onerror: () => { setStatus("request failed (server running? CORS/@connect?)", "#f66"); setBusy(false); activeReq = null; },
      ontimeout: () => { setStatus("request timed out", "#f66"); setBusy(false); activeReq = null; },
      onabort: () => {
        // leave already-translated pages (✓/✕) alone; clear only the still-pending ones
        (activeImgs || []).forEach((img) => {
          const o = overlays.get(img);
          if (o && o.badge.textContent === "…") setBadge(img, "", "");
        });
        setStatus(`interrupted (${done} ok, ${fail} failed)`, "#fa6");
        setBusy(false); activeReq = null;
      },
    });
  };

  stopBtn.onclick = () => {
    if (activeReq && activeReq.abort) activeReq.abort();
  };

  // ----- keep overlays aligned -----
  window.addEventListener("scroll", repositionAll, { passive: true });
  window.addEventListener("resize", repositionAll);

  // initial + periodic scan (handles lazy-loaded pages)
  scanImages();
  let scans = 0;
  const iv = setInterval(() => { scanImages(); if (++scans > 20) clearInterval(iv); }, 1500);
})();
