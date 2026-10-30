#!/usr/bin/env python3
"""
OLAI-GRAPH v1 packer: safetensors -> tiny graph packets + a graph manifest.

Doctrine (OLAI Surgeon): no pruning, no quantization, no deletion. Every byte
of every parameter lands in exactly one packet. The manifest is the graph:
nodes = tensors/packets, edges = layer membership. The frontend runtime walks
that graph to page weights in and out of the browser.

Streams the source file: RAM cost is one packet (~2MB), never the 3.46GB model.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import struct
import sys
import time

import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olai_pkt import pack_bytes, CODEC_RAW, CODEC_ZSTD


def log(m):
    print(f"[pack] {m}", flush=True)


def build_layers(tensor_names, config):
    """Graph edges: which tensors belong to which layer block."""
    layer_types = (config or {}).get("layer_types", [])
    per_layer, core = {}, []
    rx = re.compile(r"^model\.layers\.(\d+)\.")
    for nm in tensor_names:
        m = rx.match(nm)
        if m:
            per_layer.setdefault(int(m.group(1)), []).append(nm)
        else:
            core.append(nm)
    layers = []
    for i, kind in enumerate(layer_types):
        layers.append({"idx": i, "kind": kind, "tensors": sorted(per_layer.get(i, []))})
    if core:
        layers.append({"idx": -1, "kind": "core (embed / global norm)", "tensors": sorted(core)})
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="patient safetensors file")
    ap.add_argument("--out", required=True, help="output packets dir")
    ap.add_argument("--manifest", required=True, help="output graph json")
    ap.add_argument("--model", default="", help="model id, e.g. LiquidAI/LFM3-1.7B")
    ap.add_argument("--config", default=None, help="model config.json (for layer graph)")
    ap.add_argument("--payload-mb", type=float, default=2.0, help="target raw bytes per packet")
    ap.add_argument("--zstd-level", type=int, default=12)
    ap.add_argument("--codec", choices=["zstd", "raw"], default="zstd",
                    help="raw = bit-identical payloads (best for browser demos/CDNs)")
    ap.add_argument("--match", default=None,
                    help="regex: pack only matching tensors; rest recorded as planned")
    ap.add_argument("--expect-sha256", default=None,
                    help="sha256 of the FULL original file (full mode, for unpack proof)")
    ap.add_argument("--license", default="LFM Open License v1.0")
    args = ap.parse_args()

    payload_target = int(args.payload_mb * 1024 * 1024)
    os.makedirs(args.out, exist_ok=True)
    config = json.load(open(args.config)) if args.config else {}
    want_codec = CODEC_ZSTD if args.codec == "zstd" else CODEC_RAW

    t_entries = []
    total_raw = total_comp = total_pkts = 0
    t0 = time.time()
    opaque_pkts = 0

    def pack_range(fh, name, dtype, start, end, tid, out_entries):
        """Pack [start,end) of the open file into packets; append entry."""
        nonlocal total_raw, total_comp, total_pkts, opaque_pkts
        parts = max(1, (end - start + payload_target - 1) // payload_target)
        pkts = []
        for p in range(parts):
            s = start + p * payload_target
            e = min(start + (p + 1) * payload_target, end)
            fh.seek(s)
            raw = fh.read(e - s)
            if len(raw) != e - s:
                raise EOFError(f"short read on {name}")
            sha = hashlib.sha256(raw).hexdigest()
            blob = pack_bytes(tid, p + 1, parts, name, raw, want_codec, args.zstd_level)
            fname = f"{tid:04d}-{p + 1:03d}.pkt"
            with open(os.path.join(args.out, fname), "wb") as pf:
                pf.write(blob)
            pkts.append({"f": fname, "part": p + 1, "raw": len(raw),
                         "comp": len(blob), "sha256": sha})
            total_raw += len(raw)
            total_comp += len(blob)
            total_pkts += 1
            opaque_pkts += 1
        out_entries.append({"id": tid, "name": name, "dtype": dtype,
                            "shape": [end - start], "offset": start,
                            "nbytes": end - start, "packets": pkts,
                            "planned": False, "opaque": True})

    with open(args.src, "rb") as fh:
        fh.seek(0, 2)
        fsize = fh.tell()
        fh.seek(0)
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header_blob = fh.read(hlen)
        # safetensors data_offsets are RELATIVE to the end of the header.
        # All of our packing operates in ABSOLUTE file offsets.
        data_start = 8 + hlen
        meta = json.loads(header_blob)
        meta.pop("__metadata__", None)
        tensors = sorted(meta.items(), key=lambda kv: kv[1]["data_offsets"][0])
        rx = re.compile(args.match) if args.match else None

        for tid, (name, t) in enumerate(tensors):
            rel_start, rel_end = t["data_offsets"]
            start = data_start + rel_start   # absolute
            end = data_start + rel_end       # absolute
            nbytes = end - start
            selected = (rx is None) or bool(rx.search(name))
            entry = {"id": tid, "name": name, "dtype": t["dtype"], "shape": t["shape"],
                     "offset": start, "nbytes": nbytes, "packets": [], "planned": not selected}
            if selected:
                parts = max(1, (nbytes + payload_target - 1) // payload_target)
                for p in range(parts):
                    s = start + p * payload_target
                    e = min(start + (p + 1) * payload_target, end)
                    fh.seek(s)
                    raw = fh.read(e - s)
                    if len(raw) != e - s:
                        raise EOFError(f"short read on {name}")
                    sha = hashlib.sha256(raw).hexdigest()
                    blob = pack_bytes(tid, p + 1, parts, name, raw, want_codec, args.zstd_level)
                    fname = f"{tid:04d}-{p + 1:03d}.pkt"
                    with open(os.path.join(args.out, fname), "wb") as pf:
                        pf.write(blob)
                    entry["packets"].append(
                        {"f": fname, "part": p + 1, "raw": len(raw),
                         "comp": len(blob), "sha256": sha})
                    total_raw += len(raw)
                    total_comp += len(blob)
                    total_pkts += 1
                    if total_pkts % 25 == 0:
                        el = time.time() - t0
                        log(f"{total_pkts} packets  {total_raw / 1e6:.0f}MB raw  {el:.0f}s")
            t_entries.append(entry)

        # Doctrine: cover EVERY byte of the patient. Unclaimed regions (gaps
        # between tensors, file tail) become opaque pseudo-tensor nodes.
        covered_end = data_start
        opaque_entries = []
        next_opaque_id = len(tensors)
        for entry in sorted(t_entries, key=lambda e: e["offset"]):
            if entry["offset"] > covered_end:
                gs, ge = covered_end, entry["offset"]
                pack_range(fh, f"__gap__@{gs}", "BYTES", gs, ge,
                           next_opaque_id, opaque_entries)
                next_opaque_id += 1
            covered_end = max(covered_end, entry["offset"] + entry["nbytes"])
        if covered_end < fsize:
            pack_range(fh, "__file_tail__", "BYTES", covered_end, fsize,
                       next_opaque_id, opaque_entries)
        if opaque_entries:
            log(f"opaque coverage: {sum(e['nbytes'] for e in opaque_entries)} bytes "
                f"in {len(opaque_entries)} unclaimed region(s) — nothing discarded")
        t_entries.extend(opaque_entries)
        t_entries.sort(key=lambda e: e["offset"])

    manifest = {
        "format": "OLAI-GRAPH",
        "version": 1,
        "model": args.model,
        "license": args.license,
        "fingerprint": {"model.safetensors": args.expect_sha256},
        "packet_payload_target": payload_target,
        "codec_policy": args.codec,
        "partial": rx is not None,
        "match": args.match,
        "full_file_coverage": rx is None,
        "header_b64z": base64.b64encode(
            zstd.ZstdCompressor(level=19).compress(header_blob)).decode(),
        "stats": {
            "tensors": len(t_entries),
            "packets": total_pkts,
            "raw_bytes": total_raw,
            "comp_bytes": total_comp,
            "file_bytes": fsize,
            "ratio": (total_comp / total_raw) if total_raw else None,
            "seconds": round(time.time() - t0, 1),
        },
        "tensors": t_entries,
        "layers": build_layers([t["name"] for t in t_entries], config),
    }
    mdir = os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(mdir, exist_ok=True)
    with open(args.manifest, "w") as mf:
        json.dump(manifest, mf)
    log(f"DONE tensors={len(t_entries)} packets={total_pkts} "
        f"raw={total_raw / 1e9:.3f}GB packed={total_comp / 1e9:.3f}GB "
        f"ratio={(total_comp / total_raw) if total_raw else 0:.4f} "
        f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
