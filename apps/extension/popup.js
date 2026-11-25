const $ = (id) => document.getElementById(id);
const FIELDS = ["fromLang", "toLang", "contextMode", "llmMode", "localUrl", "quality", "llmInput"]; // relay off: "relayUrl"
const DOWNLOAD_URL = "https://github.com/parwer/MangaTVL/releases/latest";
// the Model field follows the provider in use: each keeps its own model and its own naming
const MODEL_UI = {
  openrouter: { key: "modelOpenrouter", placeholder: "Default", example: "e.g. google/gemini-3-flash-preview", tip: "provider/model, as OpenRouter names it. Blank = the default." },
  gemini: { key: "modelGemini", placeholder: "Default", example: "e.g. gemini-3-flash-preview", tip: "Model name, as Google AI Studio names it. Blank = the default." },
  local: { key: "modelLocal", tip: "One of the models you have installed." },
};
const LLM_HINTS = {
  openrouter: "One account for many AI models. Sign in below.",
  gemini: "Google's AI, with your own API key.",
  local: "Runs on your own PC: free, private and offline. Needs Ollama or LM Studio.",
  builtin: "Free, with no account, in every browser. Uses Bing Translator, so it needs internet. It only sees the text, not the page.",
};
const CONTEXT_HINTS = {
  chapter: "Keeps names and the way characters talk consistent from page to page. Pages are translated in order.",
  page: "Translates several pages at once. Quicker, but each page is translated on its own.",
};
const AI_NAMES = { openrouter: "OpenRouter", gemini: "Google Gemini", local: "Local AI", builtin: "Built-in translator" };

let settings = {}, status = null, first = true, tab = null, host = "", tabState = null, pollTimer = null;
const msg = (m) => chrome.runtime.sendMessage(m);

// ---------------------------------------------------------------- readiness
function check(id, mark, title, text, fixes) {
  const li = $(id);
  li.dataset.mark = mark;
  li.title = mark === "ok" ? text || "" : ""; // nothing to fix: the details stay in the tooltip
  li.querySelector(".mark").className = `mark ${mark}`;
  li.querySelector("b").textContent = title;
  li.querySelector("small").textContent = text || "";
  const box = li.querySelector(".fix");
  if (!box) return;
  box.replaceChildren();
  for (const f of fixes || []) {
    const el = document.createElement(f.href ? "a" : "button");
    el.className = `btn small ${f.primary ? "primary" : "ghost"}`;
    el.textContent = f.label;
    if (f.href) { el.href = f.href; el.target = "_blank"; el.rel = "noopener"; } else el.onclick = f.fn;
    box.appendChild(el);
  }
}

function showTab(name) {
  for (const t of ["home", "settings"]) $(`tab-${t}`).hidden = t !== name;
  for (const b of document.querySelectorAll("nav button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
}
for (const b of document.querySelectorAll("nav button")) b.onclick = () => showTab(b.dataset.tab);
const openSettings = (focus) => { showTab("settings"); $("sec-ai").open = true; if (focus) $(focus).focus(); };
const useBuiltin = { label: "Use the built-in translator", fn: () => setLlmMode("builtin") };
const signInFix = { label: "Sign in with OpenRouter", primary: true, fn: () => signIn() };

async function refresh() {
  status = await msg({ type: "status" });
  const st = status, b = st.backend, info = (b && b.info) || {};
  settings = st.settings;
  const blockers = [];
  let busy = false, warn = false;

  // 1. the Companion app
  if (!b) {
    blockers.push("app");
    check("c-app", "bad", "Companion app isn't running", "Open it from the Start menu, or install it first.",
      [{ label: "Download", href: DOWNLOAD_URL, primary: true }, { label: "Check again", fn: refresh }]);
  } else if (b.kind !== "companion" || !info.jobs) {
    blockers.push("app");
    check("c-app", "bad", "Companion app needs an update", "", [{ label: "Download", href: DOWNLOAD_URL, primary: true }]);
  } else if (info.load_error) {
    blockers.push("app");
    check("c-app", "bad", "Companion app couldn't start", "Quit it from the tray icon and open it again.",
      [{ label: "Check again", fn: refresh }, { label: "Reinstall", href: DOWNLOAD_URL }]);
  } else if (info.ready === false) {
    busy = true;
    check("c-app", "busy", "Companion app is starting…", "");
  } else {
    check("c-app", "ok", info.cpu_only ? "Companion app (no graphics card)" : "Companion app",
      `${info.version ? `Version ${info.version} · ` : ""}${info.cpu_only ? "No graphics card found, so about 30 sec per page" : "Using your graphics card"}`);
    $("version").textContent = `${EXT_VERSION}${info.version ? ` · app ${info.version}` : ""}`;
  }

  // 2. the AI
  if (st.missing === "openrouter") {
    blockers.push("ai");
    check("c-ai", "bad", "OpenRouter isn't connected", "", [signInFix, useBuiltin]);
  } else if (st.missing === "gemini") {
    blockers.push("ai");
    check("c-ai", "bad", "Gemini API key is missing", "",
      [{ label: "Add a key", primary: true, fn: () => openSettings("geminiKey") }, useBuiltin]);
  } else {
    const model = st.llm === "openrouter" ? settings.modelOpenrouter : st.llm === "gemini" ? settings.modelGemini
      : st.llm === "local" ? settings.modelLocal : "";
    const detail = st.llm === "builtin" ? "Free, no account. Uses Bing Translator online."
      : model ? `Model: ${model}` : st.llm === "local" ? "Model: picked automatically" : "Connected";
    check("c-ai", "ok", AI_NAMES[st.llm], detail,
      [{ label: "Change", fn: () => openSettings("llmMode") }]);
    if (st.llm === "local" && b && info.jobs) checkLocalAi(); // is the server up? (answers a moment later)
  }

  // 3. this site
  await siteRow();

  // overall
  const tone = blockers.length ? "bad" : busy ? "busy" : warn ? "warn" : "ok";
  $("status").dataset.tone = tone;
  const titles = { bad: "Not ready", busy: "Starting…", warn: "Ready", ok: "Ready" };
  $("status-title").textContent = titles[tone];
  translateButton(blockers.length > 0 || busy);

  if (first) fillSettings();
  first = false;
  showProviders();
  // the app may be starting or not running yet: look again until it is ready
  clearTimeout(pollTimer);
  if (!b || info.ready === false) pollTimer = setTimeout(refresh, 2500);
}

async function checkLocalAi() {
  const r = await msg({ type: "local-models", url: settings.localUrl || "" });
  if (r && r.ok && r.models.length) return;
  check("c-ai", "bad", "Local AI isn't answering", r && r.ok
    ? "It's running but has no models. Install one, e.g. ollama pull qwen3-vl:4b-instruct"
    : "Start Ollama or LM Studio, or check the server address.",
    [{ label: "Check again", fn: refresh }, { label: "Settings", fn: () => openSettings("localUrl") }]);
  $("status").dataset.tone = "bad";
  $("status-title").textContent = "Not ready";
  translateButton(true);
}

// ---------------------------------------------------------------- this tab
async function siteRow() {
  [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  host = "";
  try { const u = new URL(tab.url); if (/^https?:$/.test(u.protocol)) host = u.hostname; } catch (_) {}
  $("c-site").hidden = !host;
  if (!host) return;
  const { sites = {} } = await chrome.storage.local.get("sites");
  const on = !!sites[host];
  $("site").checked = on;
  $("c-site").querySelector("b").textContent = host;
  $("c-site").title = "Show the Translate button on this site. Right-clicking an image works on any site.";
  tabState = await chrome.tabs.sendMessage(tab.id, { type: "state" }).catch(() => null);
}

$("site").onchange = async () => {
  const { sites = {} } = await chrome.storage.local.get("sites");
  if ($("site").checked) sites[host] = true; else delete sites[host];
  await chrome.storage.local.set({ sites });
  siteRow();
};

function translateButton(blocked) {
  const btn = $("translate"), hint = $("translateHint");
  btn.classList.toggle("stop", !!(tabState && tabState.on));
  if (!host) {
    btn.disabled = true; btn.textContent = "Translate this page";
    hint.textContent = "Open a manga page first.";
    return;
  }
  if (tabState && tabState.on) {
    btn.disabled = false; btn.textContent = "Stop translating";
    hint.textContent = `${tabState.done} page${tabState.done === 1 ? "" : "s"} done${tabState.left ? ` · ${tabState.left} to go` : ""}`;
    return;
  }
  btn.disabled = blocked;
  btn.textContent = tabState && tabState.done ? "Translate more pages" : "Translate this page";
  hint.textContent = "";
}

$("translate").onclick = async () => {
  if (tabState && tabState.on) { await chrome.tabs.sendMessage(tab.id, { type: "stop" }).catch(() => {}); return refresh(); }
  try {
    if (!tabState) { // tab opened before the extension was installed or updated: add the script now
      await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["content.css"] });
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
    }
    await chrome.tabs.sendMessage(tab.id, { type: "translate-all" });
    window.close();
  } catch (e) {
    $("translateHint").textContent = "MangaTVL can't run on this page. Try reloading it.";
  }
};

// ---------------------------------------------------------------- settings
function fillSettings() {
  for (const f of FIELDS) if (settings[f] !== undefined && $(f)) $(f).value = settings[f];
  setSeg(settings.contextMode || "chapter");
  $("llmMode").title = LLM_HINTS[$("llmMode").value];
}

function setSeg(v) {
  for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-checked", String(b.dataset.value === v));
}
for (const b of document.querySelectorAll(".seg button")) b.title = CONTEXT_HINTS[b.dataset.value];
for (const b of document.querySelectorAll(".seg button")) {
  b.onclick = () => { setSeg(b.dataset.value); save({ contextMode: b.dataset.value }); };
}

async function save(v) { // every change saves at once
  await chrome.storage.local.set(v);
  Object.assign(settings, v);
}

function showProviders() {
  const mode = $("llmMode").value;
  for (const p of ["openrouter", "gemini", "local"]) $(`p-${p}`).hidden = mode !== p;
  const orOn = !!settings.openrouterKey, gmOn = !!settings.geminiKey;
  $("or-state").textContent = orOn ? "Connected" : "";
  $("signin").textContent = orOn ? "Reconnect" : "Sign in";
  $("signin").className = orOn ? "ghost small" : "primary small";
  $("or-out").hidden = !orOn;
  $("gm-state").textContent = gmOn ? "Saved" : "";
  $("geminiKey").placeholder = gmOn ? "Paste a new key" : "Paste your key";
  $("gm-out").hidden = !gmOn;
  showModel();
}

async function setLlmMode(v) {
  $("llmMode").value = v;
  await save({ llmMode: v });
  $("llmMode").title = LLM_HINTS[v];
  await refresh(); // ask the service worker which AI this choice resolves to now
}
$("llmMode").onchange = () => setLlmMode($("llmMode").value);

function modelProvider() {
  const mode = $("llmMode").value;
  return mode;
}

let shownModel = null;
async function showModel() {
  const ui = MODEL_UI[modelProvider()];
  $("modelBox").hidden = !ui;
  if (!ui || shownModel === ui.key) return; // polling: don't reset what is being typed
  shownModel = ui.key;
  $("modelHint").textContent = ui.example || ""; // the field is too narrow for a model name as its placeholder
  $("modelText").title = $("modelLocal").title = ui.tip;
  const local = ui.key === "modelLocal";
  $("modelText").hidden = local;
  $("modelLocal").hidden = !local;
  if (!local) {
    $("modelText").placeholder = ui.placeholder;
    $("modelText").value = settings[ui.key] || "";
    return;
  }
  await fillLocalModels();
}

async function fillLocalModels() {
  const sel = $("modelLocal"), saved = settings.modelLocal || "";
  const r = await msg({ type: "local-models", url: $("localUrl").value.trim() });
  const models = r && r.ok ? [...r.models] : [];
  if (saved && !models.includes(saved)) models.unshift(saved);
  sel.innerHTML = "";
  sel.add(new Option("Automatic", ""));
  for (const m of models) sel.add(new Option(m, m));
  sel.value = saved;
  $("modelHint").textContent = !(r && r.ok)
    ? `Can't reach Local AI: ${r && r.error === "no-backend" ? "the Companion app isn't running" : (r && r.error) || "unknown reason"}`
    : !r.models.length ? "No models installed yet (e.g. ollama pull qwen3-vl:4b-instruct)." : "";
}

async function saveModel() {
  const ui = MODEL_UI[modelProvider()];
  if (!ui) return;
  const v = ui.key === "modelLocal" ? $("modelLocal").value : $("modelText").value.trim();
  await save({ [ui.key]: v });
  refresh();
}
$("modelLocal").onchange = saveModel;
$("modelText").onchange = saveModel;

async function signIn() {
  openSettings();
  $("signin").textContent = "Opening…";
  $("or-msg").textContent = "";
  const r = await msg({ type: "signin-openrouter" });
  if (r && r.error) $("or-msg").textContent = `Sign-in didn't finish: ${r.error}`;
  shownModel = null;
  refresh();
}
$("signin").onclick = signIn;
$("or-out").onclick = async () => { await msg({ type: "signout", provider: "openrouter" }); first = true; shownModel = null; refresh(); };
$("gm-out").onclick = async () => { await msg({ type: "signout", provider: "gemini" }); first = true; shownModel = null; refresh(); };
$("geminiKey").onchange = async () => {
  const key = $("geminiKey").value.trim();
  if (!key) return;
  // like signing in to OpenRouter: a new key is what the user wants to use now
  await save({ geminiKey: key, ...(settings.llmMode === "builtin" ? { llmMode: "gemini" } : {}) });
  $("geminiKey").value = "";
  shownModel = null;
  refresh();
};

$("checkLocal").onclick = async () => {
  $("localInfo").textContent = "Testing…";
  const r = await msg({ type: "local-models", url: $("localUrl").value.trim() });
  $("localInfo").textContent = r && r.ok
    ? (r.models.length ? `Connected: ${r.models.length} model${r.models.length > 1 ? "s" : ""}` : "Connected, no models yet")
    : `Can't connect: ${r && r.error === "no-backend" ? "the Companion app isn't running" : (r && r.error) || "unknown reason"}`;
  if (modelProvider() === "local") fillLocalModels();
};
$("localUrl").onchange = async () => {
  await save({ localUrl: $("localUrl").value.trim() });
  if (modelProvider() === "local") fillLocalModels();
  refresh();
};

$("fromLang").onchange = () => save({ fromLang: $("fromLang").value });
$("toLang").onchange = () => save({ toLang: $("toLang").value });
$("swap").onclick = () => {
  const f = $("fromLang").value, t = $("toLang").value;
  $("fromLang").value = t; $("toLang").value = f;
  save({ fromLang: $("fromLang").value, toLang: $("toLang").value });
};
$("quality").onchange = () => save({ quality: $("quality").value });
$("llmInput").onchange = () => save({ llmInput: $("llmInput").value });

const EXT_VERSION = chrome.runtime.getManifest().version;
$("version").textContent = EXT_VERSION;
refresh();
