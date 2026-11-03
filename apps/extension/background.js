// MangaTVL — service worker.
// Responsibilities:
//   1. "Sign in with OpenRouter" (OAuth PKCE) -> the user never copies an API key.
//   2. Discover the local companion app (127.0.0.1:8765); the user never types a server URL.
//   3. Fetch page images itself (host_permissions) and ship bytes to the backend,
//      so hot-link-protected reader sites still work (the old userscript sent URLs
//      and the server re-downloaded them without the site's cookies/referrer).

const COMPANION_URLS = ["http://127.0.0.1:8765", "http://localhost:8765"];
const DEFAULTS = {
  llmMode: "auto",          // auto | openrouter | gemini | local | builtin
  openrouterKey: "",
  geminiKey: "",
  model: "",                // blank = backend default for the chosen provider
  localUrl: "",             // Local AI: OpenAI-compatible server; blank = Ollama (http://127.0.0.1:11434/v1)
  toLang: "thai",
  fromLang: "auto",
  // relay off: a hosted backend in place of the app (needs the analyze / job / result flow there too)
  // relayUrl: "",
  // relayToken: "",
  quality: "best",          // best | fast
  llmInput: "image",        // image: the AI sees the page | text: OCR text only (text-only models, fewer tokens)
  contextMode: "chapter",   // chapter: story context page to page | page: each page on its own, in parallel
};

async function getSettings() {
  const s = await chrome.storage.local.get(DEFAULTS);
  return { ...DEFAULTS, ...s };
}

// ---------------------------------------------------------------- OAuth PKCE
function b64url(bytes) {
  let s = btoa(String.fromCharCode(...new Uint8Array(bytes)));
  return s.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function signInOpenRouter() {
  const verifier = b64url(crypto.getRandomValues(new Uint8Array(32)));
  const challenge = b64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  const state = b64url(crypto.getRandomValues(new Uint8Array(12)));
  const authUrl = (cb) => `https://openrouter.ai/auth?callback_url=${encodeURIComponent(cb)}` +
    `&code_challenge=${challenge}&code_challenge_method=S256&state=${state}&key_label=MangaTVL`;
  let code = null;
  try {
    // Path 1: extension-native redirect (https://<ext-id>.chromiumapp.org/openrouter).
    const result = await chrome.identity.launchWebAuthFlow({ url: authUrl(chrome.identity.getRedirectURL("openrouter")), interactive: true });
    code = new URL(result).searchParams.get("code");
  } catch (e) {
    // Path 2: OpenRouter documents localhost callbacks as always allowed -> let the
    // companion app catch the redirect and hand the code back (poll by `state`).
    const b = await findBackend(await getSettings());
    if (!b || b.kind !== "companion") throw new Error("Sign-in redirect was rejected and no companion app is running: " + e.message);
    // Docs name `localhost` explicitly (any port); state goes in the path because
    // OpenRouter isn't documented to echo the `state` query param.
    const port = new URL(b.base).port || "8765";
    await chrome.tabs.create({ url: authUrl(`http://localhost:${port}/oauth/callback/${state}`) });
    for (let i = 0; i < 300 && !code; i++) { // up to 5 min
      await new Promise((r) => setTimeout(r, 1000));
      const r = await fetch(`${b.base}/oauth/code?state=${state}`);
      if (r.ok) code = (await r.json()).code;
    }
  }
  if (!code) throw new Error("OpenRouter did not return a code");
  const r = await fetch("https://openrouter.ai/api/v1/auth/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: "S256" }),
  });
  if (!r.ok) throw new Error(`key exchange failed: HTTP ${r.status}`);
  const { key } = await r.json();
  await chrome.storage.local.set({ openrouterKey: key, llmMode: "openrouter" });
  return { ok: true };
}

// ---------------------------------------------------------------- backend discovery
let cachedBackend = null;
async function findBackend(settings) {
  if (cachedBackend) return cachedBackend;
  for (const base of COMPANION_URLS) {
    try {
      const r = await fetch(`${base}/v1/health`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) {
        const info = await r.json();
        cachedBackend = { kind: "companion", base, info };
        return cachedBackend;
      }
    } catch (_) { /* not running */ }
  }
  // relay off (see DEFAULTS)
  // if (settings.relayUrl) {
  //   cachedBackend = { kind: "relay", base: settings.relayUrl.replace(/\/$/, ""), info: {} };
  //   return cachedBackend;
  // }
  return null;
}

// ---------------------------------------------------------------- LLM credentials
// Pick what to send. "auto" prefers whatever the user already has, in the order
// that gives the best context-aware quality, and falls back to the companion's
// local model (no account at all).
function llmConfig(s, backendInfo) {
  const mode = s.llmMode;
  if ((mode === "openrouter" || mode === "auto") && s.openrouterKey)
    return { provider: "openrouter", api_key: s.openrouterKey, model: s.model || null };
  if ((mode === "gemini" || mode === "auto") && s.geminiKey)
    return { provider: "gemini", api_key: s.geminiKey, model: s.model || null };
  if (mode === "builtin") return { provider: "builtin" }; // Chrome on-device Translator, done in content.js
  const local = { provider: "local", model: s.model || null, base_url: s.localUrl || null };
  if (mode === "local" || (mode === "auto" && s.localUrl)) return local; // the user's own server
  // relay off: free daily quota, no account (apps/relay)
  // if (mode === "auto" && backendInfo && backendInfo.relay) return { provider: "relay" };
  if (backendInfo && backendInfo.local_llm) return local;
  return { provider: "none" };
}

async function blobToDataURL(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  return `data:${blob.type || "image/jpeg"};base64,${btoa(bin)}`;
}

async function pageImage(src, dataUrl, pageUrl) {
  if (dataUrl) return dataUrl;
  const r = await fetch(src, { referrer: pageUrl, credentials: "include" });
  if (!r.ok) throw new Error(`image fetch HTTP ${r.status}`);
  return blobToDataURL(await r.blob());
}

async function call(backend, path, body) {
  const headers = { "Content-Type": "application/json" };
  const r = await fetch(`${backend.base}${path}`, body ? { method: "POST", headers, body: JSON.stringify(body) } : {});
  if (!r.ok) {
    let detail = (await r.text()).slice(0, 300);
    try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
    return { error: r.status === 429 ? detail : `backend HTTP ${r.status}: ${detail}`, quota: r.status === 429 };
  }
  return r.json();
}

// Companion jobs: analyze (GPU) -> job (LLM, returns the context at once) -> result (lettered page).
// Split so the page can analyse page N+1 and letter page N while the LLM works on another page.
async function analyzePage({ src, dataUrl, pageUrl }) {
  const s = await getSettings();
  const backend = await findBackend(s);
  if (!backend) return { error: "no-backend" };
  if (backend.kind !== "companion" || !backend.info.jobs) return { error: "old-backend" };
  return call(backend, "/v1/analyze", { image: await pageImage(src, dataUrl, pageUrl), from_lang: s.fromLang, to_lang: s.toLang,
    quality: s.quality });
}

async function translateJob({ job, context }) {
  const s = await getSettings();
  const backend = await findBackend(s);
  return call(backend, "/v1/job", { job, llm: llmConfig(s, backend.info), context: context || null,
    send_image: s.llmInput !== "text" });
}

async function jobResult({ job }) {
  return call(await findBackend(await getSettings()), `/v1/result/${job}`);
}

async function translateImage({ src, dataUrl, pageUrl, prevContext }) {
  const s = await getSettings();
  const backend = await findBackend(s);
  if (!backend) return { error: "no-backend" };
  const llm = llmConfig(s, backend.info);
  const headers = { "Content-Type": "application/json" };
  // relay off: if (backend.kind === "relay" && s.relayToken) headers.Authorization = `Bearer ${s.relayToken}`;
  const body = {
    image: await pageImage(src, dataUrl, pageUrl), page_url: pageUrl, from_lang: s.fromLang, to_lang: s.toLang,
    quality: s.quality, llm, context: prevContext || null, send_image: s.llmInput !== "text",
  };
  const r = await fetch(`${backend.base}/v1/translate`, { method: "POST", headers, body: JSON.stringify(body) });
  if (!r.ok) {
    let detail = (await r.text()).slice(0, 300);
    try { detail = JSON.parse(detail).detail || detail; } catch (_) {}
    return { error: r.status === 429 ? detail : `backend HTTP ${r.status}: ${detail}`, quota: r.status === 429 };
  }
  return await r.json(); // {image, context, bubbles?, needs_client_translation?}
}

// ---------------------------------------------------------------- wiring
chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  (async () => {
    try {
      if (msg.type === "signin-openrouter") return reply(await signInOpenRouter());
      if (msg.type === "status") {
        cachedBackend = null;
        const s = await getSettings();
        const b = await findBackend(s);
        return reply({ backend: b, llm: llmConfig(s, b && b.info).provider, settings: { ...s, openrouterKey: s.openrouterKey ? "•••" : "", geminiKey: s.geminiKey ? "•••" : "" } });
      }
      if (msg.type === "translate") return reply(await translateImage(msg));
      if (msg.type === "analyze") return reply(await analyzePage(msg));
      if (msg.type === "job") return reply(await translateJob(msg));
      if (msg.type === "result") return reply(await jobResult(msg));
      if (msg.type === "local-models") { // popup: does the Local AI URL answer, and with which models
        const b = await findBackend(await getSettings());
        if (!b) return reply({ ok: false, error: "no-backend" });
        const q = msg.url ? `?url=${encodeURIComponent(msg.url)}` : "";
        return reply(await call(b, `/v1/local-models${q}`));
      }
      if (msg.type === "settings") return reply(await getSettings().then((s) => ({ contextMode: s.contextMode, llmMode: s.llmMode, toLang: s.toLang })));
      if (msg.type === "render") { // second half of the "builtin" flow
        const b = await findBackend(await getSettings());
        const r = await fetch(`${b.base}/v1/render`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(msg.payload) });
        return reply(await r.json());
      }
      reply({ error: "unknown message" });
    } catch (e) {
      reply({ error: String(e && e.message || e) });
    }
  })();
  return true; // async reply
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "mtvl-one", title: "Translate this image (MangaTVL)", contexts: ["image"] });
  chrome.contextMenus.create({ id: "mtvl-all", title: "Translate all manga pages on this page", contexts: ["page"] });
});
chrome.contextMenus.onClicked.addListener((info, tab) => {
  chrome.tabs.sendMessage(tab.id, { type: info.menuItemId === "mtvl-one" ? "translate-one" : "translate-all", src: info.srcUrl });
});
