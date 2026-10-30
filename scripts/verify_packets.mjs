// End-to-end browser-pipeline test in Node:
// global fetch is patched to serve local packet files -> PacketFeed assembles tensors
// -> OlaiModel generates -> compare with PyTorch reference.
// Usage: node scripts/verify_packets.mjs
import { OlaiModel } from "../web/runtime/engine.js";
import { PacketFeed } from "../web/runtime/pktfeed.js";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

// serve packets + fzstd UMD into the shimmed browser environment
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
globalThis.fzstd = require("../web/runtime/fzstd.js");

const PKT_DIR = "/tmp/lfm25-pkts";
globalThis.fetch = async (url) => {
  const m = String(url).match(/([^\/]+\.pkt)$/);
  if (!m) throw new Error("shim fetch: unexpected url " + url);
  const data = readFileSync(`${PKT_DIR}/${m[1]}`);
  return { ok: true, status: 200, arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) };
};
globalThis.crypto ??= (await import("node:crypto")).webcrypto;

const graph = JSON.parse(readFileSync("/home/z/my-project/olai-web/graph/lfm25-full-graph.json", "utf8"));
const cfg = JSON.parse(readFileSync("/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original/config.json", "utf8"));

const t0 = Date.now();
let verified = 0;
const feed = new PacketFeed(graph, ["file:///dev/null/{f}"], {
  concurrency: 4,
  onPacket: (done, total) => { verified = done; if (done % 100 === 0) console.log(`packets ${done}/${total}`); },
});
// our shim doesn't use templates; monkey-match: PacketFeed builds URLs via templates; patch _urls
feed._urls = (file) => [`local://${file}`];
const tensors = await feed.loadAll();
console.log(`assembled ${tensors.size} tensors from ${verified} packets in ${((Date.now() - t0) / 1000).toFixed(1)}s (every packet sha256-verified via crypto.subtle)`);

const ffnDim = graph.tensors.find((t) => t.name.endsWith("feed_forward.w1.weight")).shape[0];
const model = new OlaiModel({ layerTypes: cfg.layer_types, ffnDim, vocab: cfg.vocab_size }, Object.fromEntries(tensors));

const ref = JSON.parse(readFileSync("/tmp/torch_ref.json", "utf8"));
const prompt = "<|startoftext|><|im_start|>user\nHello! What is 2+2? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n";
// encode via tokenizer
const { OlaiTokenizer } = await import("../web/runtime/bpe.js");
const tok = new OlaiTokenizer(JSON.parse(readFileSync("/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original/tokenizer.json", "utf8")));
const ids = tok.encode(prompt);
console.log("prompt ids match torch:", JSON.stringify(ids) === JSON.stringify(ref.ids));

const logits = model.prefill(ids);
const am = model.argmax(logits);
const gen = [am];
let next = am;
for (let i = 0; i < 24; i++) {
  const lg = model.step(next, true);
  if (i < 23) { next = model.argmax(lg); if (next === 7) break; gen.push(next); }
}
console.log("ours :", JSON.stringify(gen));
console.log("torch:", JSON.stringify(ref.gen_ids));
const ok = JSON.stringify(gen) === JSON.stringify(ref.gen_ids.slice(0, gen.length));
console.log(ok ? "PACKET-FED GREEDY MATCH: YES ✓" : "MISMATCH ✗");
process.exit(ok ? 0 : 1);
