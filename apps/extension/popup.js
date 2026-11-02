const $ = (id) => document.getElementById(id);
const FIELDS = ["toLang", "contextMode", "llmMode", "model", "localUrl", "quality"]; // relay off: "relayUrl"

async function refresh() {
  const st = await chrome.runtime.sendMessage({ type: "status" });
  const b = st.backend;
  $("d-backend").className = "dot " + (b ? "ok" : "bad");
  $("t-backend").textContent = b ? `แอป Companion พร้อม ${b.info.version ? "v" + b.info.version : ""}` : "ยังไม่พบแอป Companion";
  $("get-app").hidden = !!b;
  const names = { openrouter: "OpenRouter ✓", gemini: "Gemini ✓", local: "Local AI ✓", builtin: "Chrome built-in", none: "ยังไม่มี AI — กด Sign in หรือเปิด Local AI" };
  $("d-llm").className = "dot " + (st.llm === "none" ? "bad" : "ok");
  $("t-llm").textContent = names[st.llm];
  $("signin").textContent = st.settings.openrouterKey ? "เชื่อมต่อ OpenRouter แล้ว ✓ (เชื่อมใหม่)" : "Sign in with OpenRouter (1 คลิก)";
  for (const f of FIELDS) if (st.settings[f] !== undefined) $(f).value = st.settings[f];
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
  refresh();
};
$("checkLocal").onclick = async () => {
  $("localInfo").textContent = "กำลังตรวจ…";
  const r = await chrome.runtime.sendMessage({ type: "local-models", url: $("localUrl").value.trim() });
  $("localInfo").textContent = r && r.ok
    ? (r.models.length ? `เจอ ${r.models.length} โมเดล: ${r.models.slice(0, 5).join(", ")} (ใช้ตัวที่รับภาพได้ เช่น qwen3-vl, gemma)` : "ต่อได้ แต่ยังไม่มีโมเดลโหลดไว้")
    : `ต่อไม่ได้: ${(r && r.error) || "ไม่ทราบสาเหตุ"}`;
};
$("toLang").onchange = () => chrome.storage.local.set({ toLang: $("toLang").value });
$("contextMode").onchange = () => chrome.storage.local.set({ contextMode: $("contextMode").value });
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
