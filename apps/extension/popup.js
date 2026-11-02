const $ = (id) => document.getElementById(id);
const FIELDS = ["toLang", "llmMode", "model", "relayUrl", "quality"];

async function refresh() {
  const st = await chrome.runtime.sendMessage({ type: "status" });
  const b = st.backend;
  $("d-backend").className = "dot " + (b ? "ok" : "bad");
  $("t-backend").textContent = b ? `${b.kind === "companion" ? "แอป Companion" : "Relay"} พร้อม ${b.info.version ? "v" + b.info.version : ""}` : "ยังไม่พบแอป Companion";
  $("get-app").hidden = !!b;
  const names = { openrouter: "OpenRouter ✓", gemini: "Gemini ✓", local: "Local AI ✓", relay: "โควตาฟรีรายวัน ✓ (ไม่ต้องมีบัญชี)", builtin: "Chrome built-in", none: "ยังไม่มี AI — กด Sign in หรือเปิด Local AI" };
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
$("toLang").onchange = () => chrome.storage.local.set({ toLang: $("toLang").value });
refresh();
