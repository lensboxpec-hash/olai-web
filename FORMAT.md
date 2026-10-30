# OLAI-GRAPH v1 — Weight Delivery Format

**One sentence:** the exact same weights, re-dressed — a monolithic 3.46 GB
safetensors becomes a graph of ~1,791 tiny, self-describing, content-addressed
packets that a pure-frontend browser runtime pages in and out on demand.

## Doctrine

- **No pruning.** Every byte of the patient's weight file lands in exactly one
  packet. Unclaimed bytes (padding, mysterious tails) become `__gap__` /
  `__file_tail__` opaque nodes — nothing is discarded, even what we cannot name.
- **No quantization.** Payload bytes are the patient's own bytes. The optional
  zstd wrapper is bit-exact on decode (proven: sha256 roundtrip).
- **No "impossible".** The format is a *re-representation*: same information,
  new shape, chosen so HTTP, CDNs, browsers and WebGPU can all chew it.

## Wire layout (little endian)

```
offset  size  field
0       4     magic 'OLPG'
4       2     version = 1
6       1     codec: 0 = raw, 1 = zstd (payload decodes bit-exact either way)
7       1     flags (reserved)
8       4     tensor_id (u32)
12      4     part (u32, 1-based)
16      4     parts (u32)
20      8     raw_len (u64, uncompressed payload bytes)
28      2     name_len (u16)
30      N     name (utf8 tensor name, e.g. "model.layers.0.conv.conv.weight")
30+N    32    sha256 of the UNCOMPRESSED payload
30+N+32 ...   payload (codec bytes)
```

Every packet is self-describing: lose the manifest and a packet still tells you
who it is. Every packet is independently verifiable: hash it, compare, ship.

## Manifest (the graph)

`graph.json` nodes and edges:

```
nodes   tensors[]   { id, name, dtype, shape, offset, nbytes, packets[], planned }
nodes   packets[]   { f, part, raw, comp, sha256 }        ← content-addressed
edges   layers[]    { idx, kind: conv|full_attention|sliding_attention|…,
                      tensors: [names] }                  ← execution order
extra   header_b64z (zstd+base64 of the original safetensors header →
        bit-exact rebuild of the source file is possible from packets alone)
extra   stats, fingerprint.model.safetensors, mirrors[]
```

`planned: true` entries are packet *slots* laid out by the header map before
weights are fetched; CI fills real `sha256`/`comp` when it packs. A manifest
with `plan_only: true` is the map of the patient taken with a needle (a 1 MB
HTTP Range request), not the 3.46 GB patient itself.

## Why tiny packets beat a monolith

| Property | 3.46 GB monolith | OLAI-GRAPH @2 MB |
|---|---|---|
| First byte of progress | after ~100% buffered | after first 2 MB |
| HTTP parallelism | 1 stream (or few big ranges) | 1,791 independent streams |
| Cache granularity | all-or-nothing | per-packet (Cache API / IndexedDB) |
| Resume after network death | restart or byte-range bookkeeping | skip cached packets |
| CDN mirrors | one origin | raw.githubusercontent + jsDelivr + Pages |
| Verify integrity | hash 3.46 GB at the end | verify every 2 MB on arrival |
| Phone-friendly paging | no | page layers in, decode, page out |

## Compression honesty (the ledger)

- zstd-12 on real LFM2.5-350M bf16 weights: **ratio 0.7775 → 22.25% smaller,
  lossless, bit-exact roundtrip proven** (`graph/lfm25-full-graph.json`).
- Expect similar single-digit-to-20% gains on LFM3. The *transport* win
  (parallel, resumable, cacheable, paged) is much bigger than the *ratio* win —
  that is the honest ledger.
- Codec id 2 is reserved for the synaptic re-representation codec (brain-node /
  synapse graph form of the same weights) — a future slot, no lies attached.

## Sizes

- LFM3-1.7B: 3,457,635,448 B → 286 tensors → **1,791 packets @ 2 MB payload**.
- Packet size is a knob: 0.25 MB → ~13k packets (manifest ~2 MB), 2 MB → 1.8k
  packets (manifest ~250 KB), 8 MB → ~450 packets.
