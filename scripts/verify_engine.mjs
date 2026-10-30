// Verify OLAI JS engine vs PyTorch reference: logits diff + greedy token match.
// Usage: node scripts/verify_engine.mjs [--speed]
import { OlaiModel, bf16BytesToF32 } from "../web/runtime/engine.js";
import { OlaiTokenizer } from "../web/runtime/bpe.js";
import { readFileSync } from "node:fs";

const DIR = "/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original";

// --- parse safetensors (all BF16) ---
const st = readFileSync(`${DIR}/model.safetensors`);
const dv = new DataView(st.buffer, st.byteOffset, st.byteLength);
const headerLen = Number(dv.getBigUint64(0, true));
const header = JSON.parse(new TextDecoder().decode(new Uint8Array(st.buffer, st.byteOffset + 8, headerLen)));
const dataStart = 8 + headerLen;

const W = {}, SHAPES = {};
const t0 = Date.now();
for (const [name, info] of Object.entries(header)) {
  if (name === "__metadata__") continue;
  const { dtype, shape, data_offsets } = info;
  if (dtype !== "BF16") throw new Error(`unexpected dtype ${dtype} for ${name}`);
  const [o0, o1] = data_offsets;
  const n = (o1 - o0) / 2;
  W[name] = bf16BytesToF32(new Uint8Array(st.buffer, st.byteOffset + dataStart + o0, o1 - o0), n);
  SHAPES[name] = shape;
}
console.log(`loaded ${Object.keys(W).length} tensors in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const cfg = JSON.parse(readFileSync(`${DIR}/config.json`, "utf8"));
const ffnDim = SHAPES["model.layers.0.feed_forward.w1.weight"][0];
console.log("ffnDim =", ffnDim);

const model = new OlaiModel(
  { layerTypes: cfg.layer_types, ffnDim, vocab: cfg.vocab_size },
  W
);
const tj = JSON.parse(readFileSync(`${DIR}/tokenizer.json`, "utf8"));
const tok = new OlaiTokenizer(tj);

const ref = JSON.parse(readFileSync("/tmp/torch_ref.json", "utf8"));

// --- prompt + prefill ---
const t1 = Date.now();
const promptIds = tok.encode(tok.renderChat([{ role: "user", content: "Hello! What is 2+2? Answer in one short sentence." }]));
console.log("our prompt ids match torch:", JSON.stringify(promptIds) === JSON.stringify(ref.ids));
const logits = model.prefill(promptIds);
const prefillS = (Date.now() - t1) / 1000;

// --- logits diff ---
let maxDiff = 0;
for (let i = 0; i < 4096; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - ref.logits_head[i]));
console.log(`prefill ${promptIds.length} tok in ${prefillS.toFixed(1)}s (${(promptIds.length / prefillS).toFixed(1)} tok/s)`);
console.log(`logits[0..4095] max|diff| vs torch bf16: ${maxDiff.toExponential(3)}`);
const am = model.argmax(logits);
console.log("argmax ours:", am, "torch:", ref.argmax, am === ref.argmax ? "MATCH" : "MISMATCH");

// --- greedy generation ---
const t2 = Date.now();
const gen = [];
let next = am;
for (let i = 0; i < 24; i++) {
  gen.push(next);
  const lg = model.step(next, true);
  if (i < 23) next = model.argmax(lg);
  if (next === 7) break; // <|im_end|>
}
const dt = (Date.now() - t2) / 1000;
console.log(`generated ${gen.length} tok in ${dt.toFixed(1)}s (${(gen.length / dt).toFixed(2)} tok/s)`);
console.log("ours :", JSON.stringify(gen));
console.log("torch:", JSON.stringify(ref.gen_ids));
console.log("GREEDY MATCH:", JSON.stringify(gen) === JSON.stringify(ref.gen_ids.slice(0, gen.length)) ? "YES ✓" : "NO ✗");
console.log("decoded:", JSON.stringify(tok.decode(gen)));
process.exit(JSON.stringify(gen) === JSON.stringify(ref.gen_ids.slice(0, gen.length)) ? 0 : 1);
