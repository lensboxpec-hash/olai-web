/**
 * OLAI BPE tokenizer — faithful JS port of the HF `tokenizers` byte-level BPE
 * used by LFM2.5-350M (tokenizer.json: ByteLevel BPE, GPT-style split regex).
 * Verified token-for-token against HuggingFace AutoTokenizer (see scripts/verify_engine.mjs).
 */
const BYTES_TO_UNICODE = (() => {
  // GPT-2 byte encoder table, exactly as HF tokenizers builds it:
  // printable bytes map to themselves, the rest map to U+0100+n in encounter order.
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const out = new Array(256);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (bs.includes(b)) out[b] = String.fromCharCode(b);
    else { out[b] = String.fromCharCode(0x100 + n); n++; }
  }
  return out;
})();
const UNICODE_TO_BYTES = (() => {
  const m = new Map();
  for (let b = 0; b < 256; b++) m.set(BYTES_TO_UNICODE[b], b);
  return m;
})();

// The exact pattern from tokenizer.json, with the Rust-scoped (?i:…) group
// rewritten for JS (JS has no scoped inline flags).
const SPLIT_RE = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export class OlaiTokenizer {
  /**
   * @param tokenizerJson parsed tokenizer.json
   */
  constructor(tj) {
    this.vocab = tj.model.vocab;           // {string -> id}
    this.merges = tj.model.merges;         // [[a,b], ...] rank = index
    this.ranks = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      const [a, b] = this.merges[i];
      this.ranks.set(a + "\u0000" + b, i);
    }
    this.added = tj.added_tokens || [];    // [{id, content, special}]
    this.addedSet = new Set(this.added.map((t) => t.content));
    // id -> piece array (decode)
    this.idToPiece = new Array(65536).fill(null);
    for (const [piece, id] of Object.entries(this.vocab)) this.idToPiece[id] = piece;
    for (const t of this.added) if (t.id < 65536) this.idToPiece[t.id] = t.content;
    // added-token literal matcher (longest first)
    const esc = this.added
      .map((t) => t.content)
      .sort((a, b) => b.length - a.length)
      .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.addedRe = new RegExp("(" + esc.join("|") + ")", "g");
    this.bosId = 1;  // <|startoftext|>
    this.eosId = 7;  // <|im_end|>
  }

  static async load(url) {
    const tj = await (await fetch(url)).json();
    return new OlaiTokenizer(tj);
  }

  _bpeWord(word) {
    // word is a string of byte-unicode chars
    if (word.length <= 1) return [word];
    let parts = [...word];
    while (true) {
      let best = Infinity, bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + "\u0000" + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; bestIdx = i; }
      }
      if (bestIdx < 0) break;
      parts = [...parts.slice(0, bestIdx), parts[bestIdx] + parts[bestIdx + 1], ...parts.slice(bestIdx + 2)];
    }
    return parts;
  }

  _encodeSegment(text, out) {
    // split regex (Isolated: every match is a chunk; we iterate all matches —
    // the pattern covers all text greedily)
    for (const m of text.matchAll(SPLIT_RE)) {
      const chunk = m[0];
      if (!chunk) continue;
      const bytes = new TextEncoder().encode(chunk);
      let wordStr = "";
      for (const b of bytes) wordStr += BYTES_TO_UNICODE[b];
      for (const piece of this._bpeWord(wordStr)) {
        const id = this.vocab[piece];
        if (id !== undefined) out.push(id);
      }
    }
  }

  encode(text) {
    const ids = [];
    let last = 0;
    for (const m of text.matchAll(this.addedRe)) {
      const idx = m.index;
      if (idx > last) this._encodeSegment(text.slice(last, idx), ids);
      const piece = m[0];
      const added = this.added.find((t) => t.content === piece);
      ids.push(added.id);
      last = idx + piece.length;
    }
    if (last < text.length) this._encodeSegment(text.slice(last), ids);
    return ids;
  }

  decode(ids) {
    let str = "";
    for (const id of ids) {
      if (id === null || id === undefined) continue;
      const piece = this.idToPiece[id];
      if (piece === null || piece === undefined) continue;
      if (this.addedSet.has(piece)) { str += piece; continue; }
      str += piece;
    }
    // byte-level reverse
    const bytes = [];
    for (const ch of str) {
      const b = UNICODE_TO_BYTES.get(ch);
      if (b !== undefined) bytes.push(b);
      else {
        // special added token text passes through as utf-8
        for (const bb of new TextEncoder().encode(ch)) bytes.push(bb);
      }
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }

  /** Render the LFM2.5 chat template for a message list (no tools / no thinking). */
  renderChat(messages, addGenerationPrompt = true) {
    let out = "<|startoftext|>";
    for (const m of messages) {
      out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
    }
    if (addGenerationPrompt) out += "<|im_start|>assistant\n";
    return out;
  }
}
