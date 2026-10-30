// Verify OLAI JS tokenizer against HuggingFace token-by-token.
// Usage: node scripts/verify_tokenizer.mjs
import { OlaiTokenizer } from "../web/runtime/bpe.js";
import { readFileSync, writeFileSync } from "node:fs";

const tj = JSON.parse(readFileSync("/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original/tokenizer.json", "utf8"));
const tok = new OlaiTokenizer(tj);

const tests = [
  "Hello, world!",
  "The quick brown fox jumps over the lazy dog.",
  "ஓலை OLAI is a Tamil-first LLM. வணக்கம்!",
  "def fibonacci(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n",
  "Numbers 12345 and 67,890.00 plus 42%.",
  "I'm sure we'll've seen what they're doing — it's 100% done.",
  "  leading spaces\tand\ttabs\n\nnewlines\r\n  ",
  "email: foo.bar+baz@example.org; visit https://example.com/x?q=1",
  "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
  "emoji 🙂🚀 test and CJK 日本語 中文",
];
const script = `
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained("/home/z/my-project/olai/artifacts/patient/LFM2.5-350M/original")
tests = json.load(open("/tmp/toks.json"))
out = [tok.encode(t) for t in tests]
dec = [tok.decode(out[i], skip_special_tokens=False) for i in range(len(tests))]
json.dump({"enc": out, "dec": dec}, open("/tmp/toks_ref.json","w"))
print("OK")
`;
writeFileSync("/tmp/toks.json", JSON.stringify(tests));
writeFileSync("/tmp/ref.py", script);
writeFileSync("/tmp/olai_enc.json", JSON.stringify(tests.map((t) => tok.encode(t))));

const { execSync } = await import("node:child_process");
execSync(`/home/z/.venv/bin/python /tmp/ref.py`, { stdio: "inherit" });
const ref = JSON.parse(readFileSync("/tmp/toks_ref.json", "utf8"));
const mine = JSON.parse(readFileSync("/tmp/olai_enc.json", "utf8"));

let pass = 0, fail = 0;
for (let i = 0; i < tests.length; i++) {
  // HF encode() adds special tokens (BOS id=1); our raw BPE doesn't — compare after stripping it
  const refIds = ref.enc[i];
  const stripped = refIds.length && refIds[0] === 1 ? refIds.slice(1) : refIds;
  const a = JSON.stringify(stripped), b = JSON.stringify(mine[i]);
  if (a === b) { pass++; console.log(`PASS [${i}] len=${mine[i].length} (+BOS in ref)`); }
  else {
    fail++;
    console.log(`FAIL [${i}] "${tests[i].slice(0, 40)}"`);
    console.log("  ref :", ref.enc[i].slice(0, 30));
    console.log("  mine:", mine[i].slice(0, 30));
  }
  const d = tok.decode(ref.enc[i]);
  if (d !== ref.dec[i]) { console.log(`  DECODE MISMATCH [${i}]:`, JSON.stringify(d.slice(0, 60)), "!=", JSON.stringify(ref.dec[i].slice(0, 60))); }
}
console.log(`\n${pass}/${tests.length} encode tests passed`);
process.exit(fail ? 1 : 0);
