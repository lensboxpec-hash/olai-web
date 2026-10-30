/**
 * OLAI runtime v0.1 — mirrors, cache, paging plan, WebGPU.
 *
 * PacketStore : multi-mirror fetch + Cache API + sha256 verify (per packet).
 * PagedEngine : walks the layer graph, pages tensors in/out (the plan, and a
 *               real WebGPU GEMV demo on real patient weights).
 */
import { parsePacket, decodePacket, sha256Hex, bf16ToF32 } from "./pkt.js";
import { GEMV_F32, RMSNORM, SILU_MUL, ARGMAX } from "./kernels.js";

const OWNER_REPO = "lensboxpec-hash/olai-web";

export function mirrorsFor(path) {
  return [
    `./${path}`,                                                      // local / Pages
    `../${path}`,                                                     // opened from web/
    `https://raw.githubusercontent.com/${OWNER_REPO}/weights/${path}`, // weights branch (CORS: *)
    `https://cdn.jsdelivr.net/gh/${OWNER_REPO}@weights/${path}`,       // jsDelivr CDN
    `https://raw.githubusercontent.com/${OWNER_REPO}/main/${path}`,    // main branch
  ];
}

export class PacketStore {
  constructor({ cacheName = "olai-v1", onEvent = () => {} } = {}) {
    this.cacheName = cacheName;
    this.onEvent = onEvent;
    this.stats = { hits: 0, fetches: 0, bytes: 0, verified: 0, failures: 0 };
  }

  async #cache() {
    if (!this.#c) this.#c = ("caches" in window) ? await caches.open(this.cacheName) : null;
    return this.#c;
  }
  #c = null;

  async get(path, expectSha) {
    const cache = await this.#cache();
    const mirrors = mirrorsFor(path);
    // 1. local cache
    if (cache) {
      const hit = await cache.match(mirrors[mirrors.length - 1]); // canonical URL key
      if (hit) {
        this.stats.hits++;
        const buf = await hit.arrayBuffer();
        this.onEvent({ type: "cache-hit", path });
        return buf;
      }
    }
    // 2. mirrors in order
    let lastErr = null;
    for (const url of mirrors) {
      try {
        this.onEvent({ type: "fetch", url });
        const res = await fetch(url, { mode: "cors" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();
        // verify when we know what to expect: hash the PAYLOAD (codec-aware),
        // matching both the packet's own header and the manifest entry
        if (expectSha) {
          const p = parsePacket(buf);
          let raw = p.payload;
          if (p.codec === 1) {
            const lib = globalThis.fzstd;
            if (!lib) throw new Error("zstd packet but fzstd not loaded");
            raw = lib.decompress(p.payload);
          }
          const hex = await sha256Hex(raw);
          if (hex !== p.shaHex || hex !== expectSha) {
            throw new Error(`sha mismatch from ${url}`);
          }
          this.stats.verified++;
        }
        this.stats.fetches++;
        this.stats.bytes += buf.byteLength;
        if (cache) await cache.put(mirrors[mirrors.length - 1], new Response(buf));
        this.onEvent({ type: "ok", url, bytes: buf.byteLength });
        return buf;
      } catch (e) {
        lastErr = e;
        this.onEvent({ type: "mirror-fail", url, error: String(e) });
      }
    }
    this.stats.failures++;
    throw lastErr || new Error("all mirrors failed");
  }
}

/** Builds the paging plan from a manifest (the graph walk order). */
export class PagedEngine {
  constructor(manifest, store) {
    this.man = manifest;
    this.store = store;
    this.residentBytes = 0;
    this.log = [];
  }

  static bytesOf(t) {
    if (!t.packets.length) return 0;
    // packed manifest: sum real raw sizes; plan-only manifest: count once
    if (t.packets[0].raw != null) {
      return t.packets.reduce((s, p) => s + p.raw, 0);
    }
    return t.nbytes;
  }

  totalBytes() { return this.man.tensors.reduce((s, t) => s + PagedEngine.bytesOf(t), 0); }

  /** Layer order for paging: core first, then idx 0..N-1. */
  pagingPlan() {
    const layers = [...this.man.layers].sort((a, b) => (a.idx === -1 ? -1 : a.idx) - (b.idx === -1 ? -1 : b.idx));
    return layers.map(l => {
      const tensors = l.tensors
        .map(n => this.man.tensors.find(t => t.name === n))
        .filter(Boolean);
      return {
        idx: l.idx, kind: l.kind,
        tensors: tensors.length,
        packets: tensors.reduce((s, t) => s + t.packets.length, 0),
        bytes: tensors.reduce((s, t) => s + PagedEngine.bytesOf(t), 0),
      };
    });
  }

  /** Paged mode: bytes moved per token (honest number). */
  pagedBytesPerToken() { return this.totalBytes(); }

  /**
   * Demo: page in ONE real tensor, verify, decode bf16→f32, return samples.
   * This is the "load a tiny packet → hold real weight bytes" loop.
   */
  async pageInTensor(name, { sampleCount = 8 } = {}) {
    const t = this.man.tensors.find(x => x.name === name);
    if (!t || !t.packets.length) throw new Error(`tensor not packed: ${name}`);
    const p = t.packets[0];
    const buf = await this.store.get(`samples/lfm25-layer0/${p.f}`, p.sha256);
    const pkt = parsePacket(buf);
    const decoded = await decodePacket(buf);
    const u16 = new Uint16Array(decoded.raw.buffer, decoded.raw.byteOffset,
      Math.min(decoded.raw.byteLength >> 1, 4096));
    const f32 = bf16ToF32(u16);
    this.residentBytes += buf.byteLength;
    this.log.push({ name, packet: p.f, sha: p.sha256.slice(0, 12), bytes: buf.byteLength });
    return { name, packet: p.f, sha: p.sha256, samples: [...f32.slice(0, sampleCount)] };
  }
}

/** WebGPU device + kernels. All f32 (bf16 decoded on CPU — works everywhere). */
export class Gpu {
  static async request() {
    if (!("gpu" in navigator)) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return new Gpu(device, adapter);
  }

  constructor(device, adapter) {
    this.device = device;
    this.adapter = adapter;
    this.info = adapter.info ? JSON.parse(JSON.stringify(adapter.info)) : {};
  }

  async #run(shaderCode, entry, buffers, paramsU32, workgroups) {
    const d = this.device;
    const module = d.createShaderModule({ code: shaderCode });
    const pipeline = d.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
    const entries = buffers.map((b, i) => ({
      binding: i, resource: { buffer: b.buffer, offset: 0, size: b.size },
    }));
    let paramsBuf = null;
    if (paramsU32) {
      paramsBuf = d.createBuffer({ size: 16 * Math.ceil(paramsU32.length / 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(paramsBuf, 0, new Uint32Array(paramsU32));
      entries.push({ binding: buffers.length, resource: { buffer: paramsBuf } });
    }
    const bind = d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    d.queue.submit([enc.finish()]);
  }

  /** y = W · x ; W row-major [N,K] f32. Returns {y, ms}. */
  async gemv(W, x, N, K) {
    const d = this.device;
    const mk = (arr, usage) => {
      const b = d.createBuffer({ size: arr.byteLength, usage });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    const wb = mk(W, GPUBufferUsage.STORAGE);
    const xb = mk(x, GPUBufferUsage.STORAGE);
    const yb = d.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = d.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const t0 = performance.now();
    await this.#run(GEMV_F32, "main",
      [{ buffer: wb, size: wb.size }, { buffer: xb, size: xb.size }, { buffer: yb, size: yb.size }],
      [N, K, 0, 0], [Math.ceil(N / 64), 1, 1]);
    enc2: {
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(yb, 0, rb, 0, N * 4);
      d.queue.submit([enc.finish()]);
    }
    await rb.mapAsync(GPUMapMode.READ);
    const y = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return { y, ms: performance.now() - t0 };
  }

  async rmsnorm(x, w, eps = 1e-5) {
    const d = this.device;
    const mk = (arr) => {
      const b = d.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.STORAGE });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    const xb = mk(x), wb2 = mk(w);
    const yb = d.createBuffer({ size: x.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rb = d.createBuffer({ size: x.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    await this.#run(RMSNORM, "main",
      [{ buffer: xb, size: xb.size }, { buffer: wb2, size: wb2.size }, { buffer: yb, size: yb.size }],
      [x.length, 0, 0, 0], [1, 1, 1]);
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(yb, 0, rb, 0, x.byteLength);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const y = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return y;
  }
}
