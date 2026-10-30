#!/usr/bin/env node
/**
 * OLAI-GRAPH v1 — Node cross-check of packet decoding (no dependencies).
 * Proves the JS packet parser in web/runtime agrees with the Python packer:
 * parses each packet header, re-hashes the RAW payload, compares to manifest.
 * (zstd packets are skipped in Node — stdlib has no zstd; browser uses fzstd.)
 */
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = process.argv[2] || join(root, "samples/lfm25-layer0-graph.json");
const packetsDir = process.argv[3] || join(root, "samples/lfm25-layer0");

const MAGIC = "OLPG";
const man = JSON.parse(readFileSync(manifestPath, "utf8"));
let ok = 0, skipped = 0, bad = 0, bytes = 0;

for (const t of man.tensors) {
  if (t.planned) continue;
  for (const p of t.packets) {
    const path = join(packetsDir, p.f);
    if (!existsSync(path)) { console.error(`MISSING ${p.f}`); bad++; continue; }
    const buf = readFileSync(path);
    const magic = buf.subarray(0, 4).toString("latin1");
    if (magic !== MAGIC) { console.error(`BAD MAGIC ${p.f}`); bad++; continue; }
    const ver = buf.readUInt16LE(4);
    const codec = buf.readUInt8(6);
    const tid = buf.readUInt32LE(8);
    const part = buf.readUInt32LE(12);
    const parts = buf.readUInt32LE(16);
    const rawLen = Number(buf.readBigUInt64LE(20));
    const nameLen = buf.readUInt16LE(28);
    const name = buf.subarray(30, 30 + nameLen).toString("utf8");
    const sha = buf.subarray(30 + nameLen, 62 + nameLen).toString("hex");
    const payload = buf.subarray(62 + nameLen);
    if (codec === 1) { skipped++; continue; } // zstd: verified by python tools
    const h = createHash("sha256").update(payload).digest("hex");
    const headerOk = name === t.name && part === p.part && rawLen === p.raw && ver === 1;
    const shaOk = h === p.sha256 && h === sha;
    if (headerOk && shaOk) { ok++; bytes += rawLen; }
    else { console.error(`MISMATCH ${p.f}`); bad++; }
  }
}
console.log(`[smoke] manifest=${manifestPath}`);
console.log(`[smoke] packets ok=${ok} zstd-skipped=${skipped} bad=${bad} raw-bytes=${(bytes / 1e6).toFixed(1)}MB`);
if (bad > 0) process.exit(1);
console.log("[smoke] PASS — JS parser agrees with Python packer");
