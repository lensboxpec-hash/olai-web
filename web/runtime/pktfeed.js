/**
 * OLAI PacketFeed — assembles model weights in the browser from OLAI-GRAPH packets.
 *
 * - Reads the graph manifest (tensors[] each with its packet list + sha256s).
 * - Fetches packets with bounded concurrency and mirror fallback.
 * - Verifies sha256 of every decompressed payload (crypto.subtle).
 * - Widens bf16 -> Float32 losslessly; never keeps the monolithic file in RAM.
 *
 * No backend. The whole model arrives as tiny HTTP objects, like images on a page.
 */
import { decodePacket } from "./pkt.js";
import { bf16BytesToF32 } from "./engine.js";

export class PacketFeed {
  /**
   * @param graph          parsed graph manifest json
   * @param urlTemplates   ["https://raw.githubusercontent.com/OWNER/REPO/weights/lfm25/{f}", ...]
   * @param opts           {concurrency, onPacket(done,total,bytes,totalBytes), onTensor(name,i,n), signal}
   */
  constructor(graph, urlTemplates, opts = {}) {
    this.graph = graph;
    this.templates = urlTemplates;
    this.concurrency = opts.concurrency || 6;
    this.onPacket = opts.onPacket || (() => {});
    this.onTensor = opts.onTensor || (() => {});
    this.signal = opts.signal;
    this.tensors = new Map();     // name -> Float32Array
    this.verifiedPackets = 0;
  }

  _urls(file) {
    return this.templates.map((t) => t.replace("{f}", file));
  }

  async _fetchOne(pkt) {
    let lastErr;
    for (const url of this._urls(pkt.f)) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const res = await fetch(url, { signal: this.signal });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const buf = await res.arrayBuffer();
          return { buf, url };
        } catch (e) {
          lastErr = e;
          if (this.signal?.aborted) throw e;
        }
      }
    }
    throw new Error(`all mirrors failed for ${pkt.f}: ${lastErr}`);
  }

  async loadAll() {
    const flat = [];
    for (const t of this.graph.tensors) {
      for (const p of t.packets) flat.push({ tensor: t, pkt: p });
    }
    const totalBytes = flat.reduce((s, x) => s + x.pkt.comp, 0);
    let done = 0, bytes = 0, doneBytes = 0;

    // per-tensor byte assembly
    const bufs = new Map(); // name -> {u8, filled, parts, got:Set}
    for (const t of this.graph.tensors) {
      bufs.set(t.name, { u8: new Uint8Array(t.nbytes), filled: 0, parts: t.packets.length, got: new Set() });
    }

    const applyPacket = async (t, pkt, buf) => {
      const { raw } = await decodePacket(buf, globalThis.fzstd);
      if (raw.length !== pkt.raw) throw new Error(`raw len ${raw.length} != ${pkt.raw} for ${t.name}#${pkt.part}`);
      const b = bufs.get(t.name);
      // place chunk: cumulative raw lengths of preceding parts
      let place = 0;
      for (const p2 of t.packets) {
        if (p2.part === pkt.part) break;
        place += p2.raw;
      }
      b.u8.set(raw, place);
      b.got.add(pkt.part);
      b.filled += raw.length;
      if (b.got.size === b.parts) {
        const n = t.nbytes / 2;
        this.tensors.set(t.name, bf16BytesToF32(b.u8, n));
        bufs.delete(t.name); // free raw bytes
        this.onTensor(t.name, this.tensors.size, this.graph.tensors.length);
      }
    };

    const queue = [...flat];
    const workers = Array.from({ length: this.concurrency }, async () => {
      while (queue.length) {
        if (this.signal?.aborted) throw new Error("aborted");
        const item = queue.shift();
        if (!item) break;
        const { buf, url } = await this._fetchOne(item.pkt);
        await applyPacket(item.tensor, item.pkt, buf);
        done++; doneBytes += buf.byteLength; bytes += item.pkt.comp;
        this.onPacket(done, flat.length, doneBytes, totalBytes, url);
      }
    });
    await Promise.all(workers);
    if (this.tensors.size !== this.graph.tensors.length) {
      throw new Error(`incomplete assembly: ${this.tensors.size}/${this.graph.tensors.length} tensors`);
    }
    return this.tensors;
  }
}
