import { PacketStore, PagedEngine, Gpu } from "./runtime/olai-runtime.js";
import { decodePacket, bf16ToF32 } from "./runtime/pkt.js";

const $ = (id) => document.getElementById(id);
const log = (msg, cls = "") => {
  const el = document.createElement("div");
  el.className = `logline ${cls}`;
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  $("console").appendChild(el);
  $("console").scrollTop = $("console").scrollHeight;
};

const fmtMB = (b) => (b / 1024 / 1024).toFixed(1) + " MB";
const fmtB = (b) => b.toLocaleString() + " B";

let store = null, engine = null, sampleMan = null, lfm3Man = null, gpu = null;

async function loadManifest(candidates) {
  const errs = [];
  for (const path of candidates) {
    try {
      const buf = await store.get(path, null);
      return JSON.parse(new TextDecoder().decode(buf));
    } catch (e) { errs.push(`${path}: ${e}`); }
  }
  throw new Error(errs.join(" | "));
}

async function init() {
  log("OLAI runtime v0.1 — booting");
  store = new PacketStore({ onEvent: (e) => {
    if (e.type === "mirror-fail") log(`mirror fail: ${e.url} (${e.error})`, "warn");
    if (e.type === "ok") log(`fetched ${fmtMB(e.bytes)} ← ${e.url}`);
  }});

  // WebGPU
  try {
    gpu = await Gpu.request();
    if (gpu) {
      log(`WebGPU: ${gpu.info.vendor || "?"} / ${gpu.info.architecture || "?"} / ${gpu.info.device || "device"}`, "ok");
      $("gpu-status").textContent = `WebGPU ready — ${gpu.info.vendor || "unknown vendor"} ${gpu.info.architecture || ""}`;
    } else {
      $("gpu-status").textContent = "WebGPU unavailable — use Chrome/Edge 113+ (packet inspection still works)";
      log("WebGPU unavailable; packet inspection still works", "warn");
    }
  } catch (e) { log(`WebGPU init failed: ${e}`, "warn"); }

  // LFM3 graph plan
  try {
    lfm3Man = await loadManifest(["graph/lfm3-graph.json", "samples/lfm25-layer0-graph.json"]);
    const t = lfm3Man.tensors;
    const bytes = t.reduce((s, x) => s + x.nbytes, 0);
    const pkts = t.reduce((s, x) => s + x.packets.length, 0);
    $("lfm3-stats").innerHTML = `
      <b>${lfm3Man.model}</b> — <i>${lfm3Man.license}</i><br>
      tensors: <b>${t.length}</b> · packets planned: <b>${pkts}</b> ·
      weight bytes: <b>${fmtMB(bytes)}</b><br>
      fingerprint: <code>${(lfm3Man.fingerprint["model.safetensors"] || "").slice(0, 16)}…</code>
      ${lfm3Man.plan_only ? "· <b>plan</b> (CI fills hashes)" : "· <b>packed</b>"}`;
    log(`LFM3 map: ${t.length} tensors, ${pkts} packets`, "ok");
    renderPlan();
  } catch (e) { log(`LFM3 map load failed: ${e}`, "err"); }

  // sample manifest
  try {
    sampleMan = await loadManifest(["samples/lfm25-layer0-graph.json"]);
    const packed = sampleMan.tensors.filter(t => !t.planned);
    const bytes = packed.reduce((s, x) => s + PagedEngine.bytesOf(x), 0);
    $("sample-stats").innerHTML = `
      <b>${sampleMan.model}</b><br>
      packed tensors: <b>${packed.length}</b> · real packets on this site: <b>${packed.reduce((s, x) => s + x.packets.length, 0)}</b> ·
      bytes: <b>${fmtMB(bytes)}</b> (raw codec — bit-identical to the patient)`;
    engine = new PagedEngine(sampleMan, store);
    log(`sample manifest ok (${packed.length} real tensors)`, "ok");
    $("btn-packet").disabled = false;
    $("btn-gemv").disabled = !gpu;
  } catch (e) { log(`sample load failed: ${e}`, "err"); }
}

function renderPlan() {
  const plan = new PagedEngine(lfm3Man, store).pagingPlan();
  const rows = plan.map(l => `
    <tr><td>${l.idx === -1 ? "core" : l.idx}</td>
        <td>${l.kind}</td><td>${l.tensors}</td><td>${l.packets}</td>
        <td>${fmtMB(l.bytes)}</td></tr>`).join("");
  const total = plan.reduce((s, l) => s + l.bytes, 0);
  const totalPkt = plan.reduce((s, l) => s + l.packets, 0);
  $("plan").innerHTML = `
    <table>
      <tr><th>layer</th><th>kind</th><th>tensors</th><th>packets</th><th>bytes</th></tr>
      ${rows}
      <tr class="sum"><td colspan="2">total</td><td>${lfm3Man.tensors.length}</td>
          <td>${totalPkt}</td><td>${fmtMB(total)}</td></tr>
    </table>
    <p class="hint">resident mode: ~${fmtMB(total)} in VRAM ·
       paged mode: ~${fmtMB(total)} moved per token (honest tradeoff) ·
       first packet on screen: 2 MB, not 3.46 GB.</p>`;
}

$("btn-packet").addEventListener("click", async () => {
  try {
    const t = sampleMan.tensors.find(x => !x.planned && x.packets.length && x.nbytes <= 1024 * 1024)
      || sampleMan.tensors.find(x => !x.planned && x.packets.length);
    const p = t.packets[0];
    log(`page-in: ${t.name} → ${p.f}`);
    const buf = await store.get(`samples/lfm25-layer0/${p.f}`, p.sha256);
    const d = await decodePacket(buf);
    const u16 = new Uint16Array(d.raw.buffer, d.raw.byteOffset, Math.min(d.raw.byteLength >> 1, 4096));
    const f32 = bf16ToF32(u16);
    $("packet-view").innerHTML = `
      tensor: <b>${d.name}</b> · part ${d.part}/${d.parts} · codec ${d.codec} ·
      raw ${fmtB(d.rawLen)}<br>
      sha256 verified: <code class="ok">${d.shaHex.slice(0, 32)}…</code><br>
      first bf16 values: <code>${[...f32.slice(0, 8)].map(v => v.toFixed(4)).join(", ")}</code>`;
    log(`packet verified + decoded (${fmtB(buf.byteLength)})`, "ok");
  } catch (e) { log(`packet demo failed: ${e}`, "err"); }
});

$("btn-gemv").addEventListener("click", async () => {
  try {
    const t = sampleMan.tensors.find(x => x.name.includes("feed_forward.w2"));
    const p = t.packets[0];
    log(`GPU demo: ${t.name} (shape ${t.shape}) from ${p.f}`);
    const buf = await store.get(`samples/lfm25-layer0/${p.f}`, p.sha256);
    const d = await decodePacket(buf);
    const N = t.shape[0], K = t.shape[1];
    const need = N * K;
    if (d.raw.byteLength < need * 2) throw new Error("packet does not hold full tensor");
    const W = bf16ToF32(new Uint16Array(d.raw.buffer, d.raw.byteOffset, need));
    // deterministic pseudo-random x
    const x = new Float32Array(K);
    let seed = 1337;
    for (let i = 0; i < K; i++) { seed = (seed * 1103515245 + 12345) >>> 0; x[i] = (seed / 4294967296) - 0.5; }
    const { y, ms } = await gpu.gemv(W, x, N, K);
    $("gemv-view").innerHTML = `
      W: <b>${t.name}</b> [${N}×${K}] — real patient bytes, decoded bf16→f32<br>
      y = W·x computed on <b>your GPU</b> in <b>${ms.toFixed(1)} ms</b><br>
      y[0..7] = <code>${[...y.slice(0, 8)].map(v => v.toFixed(4)).join(", ")}</code>`;
    log(`GEMV ${N}x${K} on GPU: ${ms.toFixed(1)}ms — real weights ran in the browser`, "ok");
  } catch (e) { log(`GPU demo failed: ${e}`, "err"); }
});

init();
