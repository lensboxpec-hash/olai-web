# OLAI Web — LFM3 as tiny packets, 100% frontend

**ஓலை / OLAI** · `LiquidAI/LFM3-1.7B` delivered as a graph of tiny packets ·
no backend, no server code, no monolithic download — the browser is the runner.

> The same weights that appear as one 3.46 GB safetensors file are re-dressed
> as **1,791 self-verifying packets (~2 MB each)** + a JSON graph. The frontend
> fetches what it needs, when it needs it, verifies every packet, caches it,
> and pages layers in and out like an operating system pages memory.

```
  patient (LFM3-1.7B)                OLAI-GRAPH                     your browser
┌────────────────────┐   pack    ┌──────────────────┐   fetch   ┌──────────────────┐
│ model.safetensors  │  ───────▶ │ graph.json       │ ────────▶ │ PacketStore      │
│ 3,457,635,448 B    │           │ + 1,791 packets  │  verify   │ (Cache API+LRU)  │
│ sha256 135922ca…   │  ◀─────── │ sha-verified     │ ◀──────── │ PagedEngine      │
└────────────────────┘  unpack   └──────────────────┘  bit-exact└──────────────────┘
        bit-exact roundtrip PROVEN on the LFM2.5-350M sibling (sha256 1c9c77a4…3ea4e)
```

## What is proven (real numbers, real runs)

| Claim | Evidence |
|---|---|
| Pack LFM2.5-350M (709 MB) into 423 packets | `graph/lfm25-full-graph.json`, 9 s run |
| Rebuild the original file **bit-for-bit** | unpack sha256 = `1c9c77a4…3ea4e` = published |
| 22.25 % lossless shrink (zstd-12, no quantization) | ratio 0.7775 in the same run |
| JS parser ≡ Python packer | `node tools/smoke.mjs` — 22/22 packets PASS |
| Full LFM3 map without downloading 3.46 GB | `graph/lfm3-graph.json` (286 tensors, 1,791 packets, from a 1 MB Range request) |

## Layout

```
FORMAT.md                  ← the OLAI-GRAPH v1 wire format + honest ledger
tools/pack.py              ← streaming packer (RAM ≈ one packet, never 3.46 GB)
tools/unpack.py            ← bit-exact rebuild + sha256 proof
tools/verify.py            ← verify every packet hash in a manifest
tools/lfm3_graph.py        ← build the FULL LFM3 packet map from the HF header alone
tools/smoke.mjs            ← Node cross-check: JS parser vs Python packer
graph/lfm3-graph.json      ← the full patient map (plan; CI fills hashes)
graph/lfm25-full-graph.json← real manifest, real hashes, roundtrip-proven
samples/lfm25-layer0/      ← 22 REAL weight packets you can fetch right now
web/                       ← pure-frontend runtime (WebGPU + paged loader)
.github/workflows/pack.yml ← CI: packs LFM3, pushes packets to the `packets` branch
.github/workflows/pages.yml← CI: publishes web/ to GitHub Pages
LICENSE                    ← LFM Open License v1.0 (weights belong to Liquid AI)
```

## The three ideas

1. **Re-representation, not destruction.** Same bytes, new shape. No pruning,
   no quantization, no deletion — the roundtrip hash proves zero loss.
2. **Tiny content-addressed packets.** Each packet names itself (tensor, part,
   hash). HTTP/2 fetches dozens in parallel; Cache API keeps what arrived;
   a dead network costs only the in-flight packets.
3. **Paged inference.** The graph tells the runtime the layer order
   (13×short-conv, 6×full-attn, 5×sliding-attn for LFM3). Desktop mode pages
   everything into VRAM; phone mode pages layer-by-layer per token and offloads
   behind an LRU — slower, but it runs.

## Run the full pack (the workflow)

Actions → **pack-lfm3** → Run workflow:
- `mode=sample` — packs LFM3 layer-0 (~156 MB → ~80 packets) and pushes to the
  `packets` branch. Fast smoke of the whole pipeline.
- `mode=full` — downloads the 3.46 GB patient in CI, packs all 1,791 packets,
  verifies every hash, pushes to `packets` and rewrites `graph/lfm3-graph.json`
  with real hashes.

The runner needs no secrets for HF (public weights); it needs only the default
`GITHUB_TOKEN` (or the configured `OLAI_PAT`) to push the branch.

## Mirrors ("multiple DBs")

The runtime tries mirrors in order and remembers the fastest:

1. `https://raw.githubusercontent.com/<owner>/olai-web/packets/`
2. `https://cdn.jsdelivr.net/gh/<owner>/olai-web@packets/` (CDN, ≤20 MB files — perfect for 2 MB packets)
3. `https://<owner>.github.io/olai-web/packets/`

All three are GitHub; the graph just names packet paths, so mirrors are free.

## License / provenance (documented, licensed steal)

Weights: **LiquidAI LFM3-1.7B** and **LFM2.5-350M**, LFM Open License v1.0
(§4 permits redistribution in Object form with attribution — packetizing is a
mechanical transformation, i.e. Object form). This repo redistributes those
weights as packets, keeps the license file, and names the source. The packer,
format and runtime are ours. Commercial-use conditions (revenue threshold) live
in the license text — read it before you sell anything around this.

## Status (honest)

- v0.1. Format: **frozen enough to trust** (bit-exact roundtrip proven).
- `web/` runtime: packet store, mirrors, cache, paging plan, WebGPU device +
  GEMV/RMSNorm/SwiGLU/argmax WGSL kernels wired; attention & conv kernels are
  stubbed with defined interfaces (marked in code). First full LFM3 token in a
  browser = next milestone, after CI lands the full packet set.
