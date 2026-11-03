const $ = (id) => document.getElementById(id);
const FIELDS = ["toLang", "contextMode", "llmMode", "localUrl", "quality", "llmInput"]; // relay off: "relayUrl"
// the Model field follows the provider in use: each keeps its own model and its own naming
const MODEL_UI = {
  openrouter: { key: "modelOpenrouter", label: "Model ของ OpenRouter", placeholder: "ผู้ให้บริการ/ชื่อโมเดล เช่น google/gemini-3-flash-preview",
                hint: "เว้นว่าง = google/gemini-3-flash-preview" },
  gemini: { key: "modelGemini", label: "Model ของ Gemini", placeholder: "ชื่อโมเดล เช่น gemini-3-flash-preview",
            hint: "เว้นว่าง = ค่าเริ่มต้นของแอป" },
  local: { key: "modelLocal", label: "Model ของ Local AI", hint: "เลือกจากโมเดลที่ติดตั้งไว้ (ไม่ต้องพิมพ์ชื่อ)" },
};
let settings = {}, resolvedLlm = "none";

async function refresh() {
  const st = await chrome.runtime.sendMessage({ type: "status" });
  const b = st.backend;
  $("d-backend").className = "dot " + (b ? "ok" : "bad");
  $("t-backend").textContent = b ? `แอป Companion พร้อม ${b.info.version ? "v" + b.info.version : ""}` : "ยังไม่พบแอป Companion";
  if (b && b.info.cpu_only) $("t-backend").textContent += " — ไม่พบ GPU: ใช้ CPU ประมาณ 30 วินาที/หน้า (คุณภาพเร็ว)";
  $("get-app").hidden = !!b;
  showLlm(st);
  $("signin").textContent = st.settings.openrouterKey ? "เชื่อมต่อ OpenRouter แล้ว ✓ (เชื่อมใหม่)" : "Sign in with OpenRouter (1 คลิก)";
  for (const f of FIELDS) if (st.settings[f] !== undefined) $(f).value = st.settings[f];
  settings = st.settings;
  resolvedLlm = st.llm;
  showModel();
}

const LLM_NAMES = { openrouter: "OpenRouter ✓", gemini: "Gemini ✓", local: "Local AI ✓", builtin: "Chrome built-in",
                    none: "ยังไม่มี AI — กด Sign in หรือเปิด Local AI" };
const MISSING = { openrouter: "เลือก OpenRouter แล้ว แต่ยังไม่ได้ Sign in", gemini: "เลือก Gemini แล้ว แต่ยังไม่ได้ใส่ API key" };

function showLlm(st) {
  $("d-llm").className = "dot " + (st.llm === "none" ? "bad" : "ok");
  $("t-llm").textContent = MISSING[st.missing] || LLM_NAMES[st.llm];
}

// "auto" shows the model of the provider auto picked right now
function modelProvider() {
  const mode = $("llmMode").value;
  return mode === "auto" ? resolvedLlm : mode;
}

async function showModel() {
  const ui = MODEL_UI[modelProvider()];
  $("modelBox").hidden = !ui;
  if (!ui) return;
  $("modelLabel").textContent = ui.label;
  $("modelHint").textContent = ui.hint;
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
  const r = await chrome.runtime.sendMessage({ type: "local-models", url: $("localUrl").value.trim() });
  const models = r && r.ok ? r.models : [];
  if (saved && !models.includes(saved)) models.unshift(saved);
  sel.innerHTML = "";
  sel.add(new Option("อัตโนมัติ (ตัวที่ติดตั้งไว้ เลือกตัวที่รับภาพได้ก่อน)", ""));
  for (const m of models) sel.add(new Option(m, m));
  sel.value = saved;
  if (!(r && r.ok)) $("modelHint").textContent = `ยังต่อ Local AI ไม่ได้: ${(r && r.error) || "ไม่ทราบสาเหตุ"}`;
  else if (!r.models.length) $("modelHint").textContent = "ต่อได้ แต่ยังไม่มีโมเดลติดตั้งไว้ (เช่น ollama pull qwen3-vl:4b-instruct)";
}

async function saveModel() {
  const ui = MODEL_UI[modelProvider()];
  if (!ui) return;
  const v = ui.key === "modelLocal" ? $("modelLocal").value : $("modelText").value.trim();
  settings[ui.key] = v;
  await chrome.storage.local.set({ [ui.key]: v });
}

$("signin").onclick = async () => {
  $("signin").textContent = "กำลังเปิดหน้าล็อกอิน…";
  const r = await chrome.runtime.sendMessage({ type: "signin-openrouter" });
  if (r.error) alert(r.error);
  refresh();
};
$("save").onclick = async () => {
  const v = Object.fromEntries(FIELDS.map((f) => [f, $(f).value]));
  if ($("geminiKey").value) v.geminiKey = $("geminiKey").value;
  await chrome.storage.local.set(v);
  await saveModel();
  refresh();
};
$("checkLocal").onclick = async () => {
  $("localInfo").textContent = "กำลังตรวจ…";
  const r = await chrome.runtime.sendMessage({ type: "local-models", url: $("localUrl").value.trim() });
  $("localInfo").textContent = r && r.ok
    ? (r.models.length ? `เจอ ${r.models.length} โมเดล — เลือกได้ในช่อง Model (ตัวที่รับภาพได้ เช่น qwen3-vl, gemma แม่นกว่า หรือเลือก "ข้อความอย่างเดียว")` : "ต่อได้ แต่ยังไม่มีโมเดลโหลดไว้")
    : `ต่อไม่ได้: ${(r && r.error) || "ไม่ทราบสาเหตุ"}`;
  if (modelProvider() === "local") fillLocalModels();
};
// ask the service worker which AI this choice resolves to now, then show its status and model
$("llmMode").onchange = async () => {
  await chrome.storage.local.set({ llmMode: $("llmMode").value });
  const st = await chrome.runtime.sendMessage({ type: "status" });
  resolvedLlm = st.llm;
  showLlm(st);
  showModel();
};
$("modelLocal").onchange = saveModel;
$("modelText").onchange = saveModel;
$("toLang").onchange = () => chrome.storage.local.set({ toLang: $("toLang").value });
$("contextMode").onchange = () => chrome.storage.local.set({ contextMode: $("contextMode").value });
$("llmInput").onchange = () => chrome.storage.local.set({ llmInput: $("llmInput").value });
refresh();

// the floating button shows only on sites switched on here (the right-click menu works everywhere)
async function siteButton() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let host = "";
  try { host = new URL(tab.url).hostname; } catch (_) {}
  const btn = $("site");
  if (!host) { btn.hidden = true; return; }
  const { sites = {} } = await chrome.storage.local.get("sites");
  const on = !!sites[host];
  btn.textContent = on ? `✓ เปิดใช้บน ${host} (กดเพื่อปิด)` : `เปิดใช้บน ${host}`;
  btn.className = on ? "on" : "ghost";
  btn.onclick = async () => {
    const { sites = {} } = await chrome.storage.local.get("sites");
    if (on) delete sites[host]; else sites[host] = true;
    await chrome.storage.local.set({ sites });
    siteButton();
  };
}
siteButton();
