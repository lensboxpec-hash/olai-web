/**
 * OLAI-GRAPH v1 — JS packet codec (must agree bit-for-bit with tools/olai_pkt.py).
 *
 * Packet layout (little endian):
 *   magic 'OLPG'(4) ver u16 codec u8 flags u8 tid u32 part u32 parts u32
 *   raw_len u64 name_len u16 name[..] sha256(32) payload[..]
 */
export const MAGIC = "OLPG";
const HEADER_SIZE = 30;

const td = new TextDecoder("utf-8");
const enc = new TextEncoder();

export function parsePacket(buf) {
  const dv = new DataView(buf);
  const magic = td.decode(new Uint8Array(buf, 0, 4));
  if (magic !== MAGIC) throw new Error(`bad magic ${magic}`);
  const version = dv.getUint16(4, true);
  const codec = dv.getUint8(6);
  const tensorId = dv.getUint32(8, true);
  const part = dv.getUint32(12, true);
  const parts = dv.getUint32(16, true);
  const rawLen = Number(dv.getBigUint64(20, true));
  const nameLen = dv.getUint16(28, true);
  const name = td.decode(new Uint8Array(buf, HEADER_SIZE, nameLen));
  const shaBytes = new Uint8Array(buf, HEADER_SIZE + nameLen, 32);
  const shaHex = [...shaBytes].map(b => b.toString(16).padStart(2, "0")).join("");
  const payload = new Uint8Array(buf, HEADER_SIZE + nameLen + 32);
  return { version, codec, tensorId, part, parts, rawLen, name, shaHex, payload };
}

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, "0")).join("");
}

/** Decode payload to raw bytes; verifies sha256. codec 1 = zstd (fzstd), 0 = raw. */
export async function decodePacket(buf, fzstd) {
  const p = parsePacket(buf);
  let raw;
  if (p.codec === 1) {
    const lib = fzstd || globalThis.fzstd;
    if (!lib) throw new Error("zstd packet but fzstd not loaded (add runtime/fzstd.js script)");
    raw = lib.decompress(p.payload);
  } else {
    raw = p.payload.slice(); // copy → aligned buffer (bf16 Uint16 views need even offsets)
  }
  if (raw.length !== p.rawLen) throw new Error(`rawLen mismatch ${raw.length} != ${p.rawLen}`);
  const hex = await sha256Hex(raw);
  if (hex !== p.shaHex) throw new Error(`sha256 mismatch for ${p.name}#${p.part}`);
  return { ...p, raw };
}

/** bf16 (Uint16 little endian) -> Float32Array */
export function bf16ToF32(u16) {
  const out = new Float32Array(u16.length);
  const f16 = new Float16Array(u16.buffer, u16.byteOffset, u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16[i];
  return out;
}
