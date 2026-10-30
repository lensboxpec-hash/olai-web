/**
 * OLAI engine v1 — pure-JS LFM2.5-350M forward pass (verified vs PyTorch, greedy-exact).
 *
 * Architecture (Lfm2ForCausalLM):
 *   x = embed[ids]
 *   16 decoder layers, kinds = conv|full_attention:
 *     x += block(op_norm(x))            // gated short-conv or GQA attention
 *     x += ffn(ffn_norm(x))             // SwiGLU
 *   x = embedding_norm(x)
 *   logits = lm_head @ x                // tied with embed
 *
 * No pruning, no quantization: weights are the patient's own BF16 bytes,
 * widened bit-exactly to Float32 at load (bf16 -> f32 is lossless).
 */

const H = 1024;            // hidden
const HEADS = 16;
const KV_HEADS = 8;
const HD = 64;             // head dim
const CONV_K = 3;
const EPS = 1e-5;
const ROPE_THETA = 1e6;
const KV_MAX = 4096;

/* ---------------- bf16 -> f32 (lossless widening) ---------------- */
export function bf16BytesToF32(u8, n) {
  const out = new Float32Array(n);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) {
    u32[i] = ((u8[2 * i] | (u8[2 * i + 1] << 8)) << 16) >>> 0;
  }
  return out;
}

/* ---------------- math kernels ---------------- */
function matvec(out, W, x, rows, cols) {
  // y[o] = sum_i W[o*cols+i] * x[i]   (torch Linear weight layout [out, cols])
  for (let o = 0; o < rows; o++) {
    let s = 0;
    const base = o * cols;
    for (let i = 0; i < cols; i++) s += W[base + i] * x[i];
    out[o] = s;
  }
}

function rmsnorm(out, x, w, n) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / n + EPS);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

function siluInplace(a, n) {
  for (let i = 0; i < n; i++) a[i] = a[i] / (1 + Math.exp(-a[i]));
}

/* ---------------- model ---------------- */
export class OlaiModel {
  /**
   * @param cfg  {layerTypes: ("conv"|"full_attention")[], ffnDim, vocab}
   * @param W    dict name -> Float32Array (OLAI names e.g. "model.layers.3.conv.in_proj.weight")
   */
  constructor(cfg, W) {
    this.cfg = cfg;
    this.W = W;
    this.F = cfg.ffnDim;
    this.vocab = cfg.vocab;
    this.L = cfg.layerTypes.length;
    this.pos = 0;
    // caches
    this.convState = cfg.layerTypes.map((t) =>
      t === "conv" ? new Float32Array(H * (CONV_K - 1)) : null
    );
    this.kCache = cfg.layerTypes.map((t) =>
      t === "full_attention" ? new Float32Array(KV_HEADS * KV_MAX * HD) : null
    );
    this.vCache = cfg.layerTypes.map((t) =>
      t === "full_attention" ? new Float32Array(KV_HEADS * KV_MAX * HD) : null
    );
    // scratch (hoisted — zero per-token allocation)
    this.x = new Float32Array(H);
    this.h = new Float32Array(H);
    this.bcx = new Float32Array(3 * H);
    this.g = new Float32Array(H);
    this.y = new Float32Array(H);
    this.op = new Float32Array(H);
    this.q = new Float32Array(HEADS * HD);
    this.k = new Float32Array(KV_HEADS * HD);
    this.v = new Float32Array(KV_HEADS * HD);
    this.ao = new Float32Array(HEADS * HD);
    this.ff1 = new Float32Array(this.F);
    this.ff3 = new Float32Array(this.F);
    this.logits = new Float32Array(this.vocab);
    this.scores = new Float32Array(KV_MAX);
    this.cosB = new Float32Array(HD / 2);
    this.sinB = new Float32Array(HD / 2);
    this.invFreq = new Float32Array(HD / 2);
    for (let j = 0; j < HD / 2; j++) this.invFreq[j] = Math.pow(ROPE_THETA, (-2 * j) / HD);
  }

  _embed(id) {
    const W = this.W["model.embed_tokens.weight"];
    return W.subarray(id * H, (id + 1) * H);
  }

  _rope(arr, nHeads, pos) {
    const cos = this.cosB, sin = this.sinB;
    for (let j = 0; j < HD / 2; j++) {
      const ang = pos * this.invFreq[j];
      cos[j] = Math.cos(ang); sin[j] = Math.sin(ang);
    }
    const half = HD / 2;
    for (let hI = 0; hI < nHeads; hI++) {
      const base = hI * HD;
      for (let j = 0; j < half; j++) {
        const a = arr[base + j], b = arr[base + j + half];
        arr[base + j] = a * cos[j] - b * sin[j];
        arr[base + j + half] = b * cos[j] + a * sin[j];
      }
    }
  }

  /**
   * One decode step for token `id` at current position.
   * withLogits=false skips the lm_head matvec (used during prefill).
   * Returns logits (Float32Array) or null.
   */
  step(id, withLogits = true) {
    const pos = this.pos;
    const W = this.W;
    const x = this.x;
    x.set(this._embed(id));
    for (let l = 0; l < this.L; l++) {
      const kind = this.cfg.layerTypes[l];
      // ---- operator block ----
      rmsnorm(this.h, x, W[`model.layers.${l}.operator_norm.weight`], H);
      if (kind === "conv") {
        const p = `model.layers.${l}.conv.`;
        matvec(this.bcx, W[p + "in_proj.weight"], this.h, 3 * H, H);
        const B = this.bcx.subarray(0, H), C = this.bcx.subarray(H, 2 * H), Xi = this.bcx.subarray(2 * H, 3 * H);
        const cw = W[p + "conv.weight"]; // [H,1,3]
        const st = this.convState[l];
        for (let c = 0; c < H; c++) {
          const gv = B[c] * Xi[c];
          this.g[c] = cw[c * CONV_K] * st[c * 2] + cw[c * CONV_K + 1] * st[c * 2 + 1] + cw[c * CONV_K + 2] * gv;
          st[c * 2] = st[c * 2 + 1]; st[c * 2 + 1] = gv;
        }
        for (let c = 0; c < H; c++) this.y[c] = C[c] * this.g[c];
        matvec(this.op, W[p + "out_proj.weight"], this.y, H, H);
        for (let c = 0; c < H; c++) x[c] += this.op[c];
      } else {
        const p = `model.layers.${l}.self_attn.`;
        matvec(this.q, W[p + "q_proj.weight"], this.h, HEADS * HD, H);
        matvec(this.k, W[p + "k_proj.weight"], this.h, KV_HEADS * HD, H);
        matvec(this.v, W[p + "v_proj.weight"], this.h, KV_HEADS * HD, H);
        const qw = W[p + "q_layernorm.weight"], kw = W[p + "k_layernorm.weight"];
        for (let hI = 0; hI < HEADS; hI++) {
          const s = this.q.subarray(hI * HD, (hI + 1) * HD);
          rmsnorm(s, s, qw, HD);
        }
        for (let hI = 0; hI < KV_HEADS; hI++) {
          const s = this.k.subarray(hI * HD, (hI + 1) * HD);
          rmsnorm(s, s, kw, HD);
        }
        this._rope(this.q, HEADS, pos);
        this._rope(this.k, KV_HEADS, pos);
        const kc = this.kCache[l], vc = this.vCache[l];
        for (let hI = 0; hI < KV_HEADS; hI++) {
          const off = hI * KV_MAX * HD + pos * HD;
          kc.set(this.k.subarray(hI * HD, (hI + 1) * HD), off);
          vc.set(this.v.subarray(hI * HD, (hI + 1) * HD), off);
        }
        const scale = 1 / Math.sqrt(HD);
        for (let hI = 0; hI < HEADS; hI++) {
          const kvH = hI >> 1; // KV_GROUP = 2
          const qoff = hI * HD, kbase = kvH * KV_MAX * HD;
          let maxS = -Infinity;
          for (let t = 0; t <= pos; t++) {
            let s = 0;
            const koff = kbase + t * HD;
            for (let d = 0; d < HD; d++) s += this.q[qoff + d] * kc[koff + d];
            s *= scale;
            this.scores[t] = s;
            if (s > maxS) maxS = s;
          }
          let sum = 0;
          for (let t = 0; t <= pos; t++) {
            const e = Math.exp(this.scores[t] - maxS);
            this.scores[t] = e; sum += e;
          }
          const inv = 1 / sum;
          for (let d = 0; d < HD; d++) this.ao[qoff + d] = 0;
          for (let t = 0; t <= pos; t++) {
            const wgt = this.scores[t] * inv;
            const voff = kbase + t * HD;
            for (let d = 0; d < HD; d++) this.ao[qoff + d] += wgt * vc[voff + d];
          }
        }
        matvec(this.op, W[p + "out_proj.weight"], this.ao, H, H);
        for (let c = 0; c < H; c++) x[c] += this.op[c];
      }
      // ---- ffn ----
      rmsnorm(this.h, x, W[`model.layers.${l}.ffn_norm.weight`], H);
      matvec(this.ff1, W[`model.layers.${l}.feed_forward.w1.weight`], this.h, this.F, H);
      matvec(this.ff3, W[`model.layers.${l}.feed_forward.w3.weight`], this.h, this.F, H);
      siluInplace(this.ff1, this.F);
      for (let i = 0; i < this.F; i++) this.ff1[i] *= this.ff3[i];
      matvec(this.op, W[`model.layers.${l}.feed_forward.w2.weight`], this.ff1, H, this.F);
      for (let c = 0; c < H; c++) x[c] += this.op[c];
    }
    rmsnorm(this.h, x, W["model.embedding_norm.weight"], H);
    if (withLogits) {
      matvec(this.logits, W["model.embed_tokens.weight"], this.h, this.vocab, H);
    }
    this.pos++;
    return withLogits ? this.logits : null;
  }

  /** Prefill all but last token without logits; returns logits of last token. */
  prefill(ids) {
    for (let i = 0; i < ids.length - 1; i++) this.step(ids[i], false);
    return this.step(ids[ids.length - 1], true);
  }

  argmax(logits) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
    return bi;
  }
}
