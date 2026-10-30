#!/usr/bin/env python3
"""
Build the FULL LFM3-1.7B graph manifest WITHOUT downloading 3.46GB.

Trick: a safetensors file starts with an 8-byte length + a JSON header that
lists every tensor (name, dtype, shape, byte offsets). We Range-fetch only the
first ~1MB from Hugging Face, parse that header, and lay out the complete
packet plan: every tensor, every packet slot, real filenames, sha256=null.
CI (pack.yml, mode=full) later runs tools/pack.py which fills real hashes.

This is the map of the patient's organs, taken with a needle, not a scalpel.
"""
import argparse
import base64
import json
import re
import struct
import urllib.request

DEFAULT_URL = "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/model.safetensors"
DEFAULT_CFG = "https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/main/config.json"


def fetch_range(url, end_byte=1048575):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{end_byte}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def build_layers(tensor_names, config):
    layer_types = (config or {}).get("layer_types", [])
    per_layer, core = {}, []
    rx = re.compile(r"^model\.layers\.(\d+)\.")
    for nm in tensor_names:
        m = rx.match(nm)
        if m:
            per_layer.setdefault(int(m.group(1)), []).append(nm)
        else:
            core.append(nm)
    layers = [{"idx": i, "kind": k, "tensors": sorted(per_layer.get(i, []))}
              for i, k in enumerate(layer_types)]
    if core:
        layers.append({"idx": -1, "kind": "core (embed / global norm)", "tensors": sorted(core)})
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--config-url", default=DEFAULT_CFG)
    ap.add_argument("--out", default="graph/lfm3-graph.json")
    ap.add_argument("--payload-mb", type=float, default=2.0)
    ap.add_argument("--expect-sha256",
                    default="135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330")
    args = ap.parse_args()

    blob = fetch_range(args.url)
    hlen = struct.unpack("<Q", blob[:8])[0]
    header_blob = blob[8:8 + hlen]
    meta = json.loads(header_blob)
    meta.pop("__metadata__", None)
    tensors = sorted(meta.items(), key=lambda kv: kv[1]["data_offsets"][0])

    try:
        config = json.load(urllib.request.urlopen(args.config_url, timeout=30))
    except Exception as e:
        print(f"[graph] config fetch failed ({e}); layer graph without kinds")
        config = {}

    payload_target = int(args.payload_mb * 1024 * 1024)
    t_entries = []
    planned_pkts = 0
    for tid, (name, t) in enumerate(tensors):
        start, end = t["data_offsets"]
        nbytes = end - start
        parts = max(1, (nbytes + payload_target - 1) // payload_target)
        pkts = [{"f": f"{tid:04d}-{p + 1:03d}.pkt", "part": p + 1, "raw": None,
                 "comp": None, "sha256": None} for p in range(parts)]
        planned_pkts += parts
        t_entries.append({"id": tid, "name": name, "dtype": t["dtype"], "shape": t["shape"],
                          "offset": start, "nbytes": nbytes, "packets": pkts, "planned": True})

    manifest = {
        "format": "OLAI-GRAPH", "version": 1,
        "model": "LiquidAI/LFM3-1.7B", "license": "LFM Open License v1.0",
        "fingerprint": {"model.safetensors": args.expect_sha256},
        "packet_payload_target": payload_target, "codec_policy": "zstd",
        "partial": False, "match": None, "plan_only": True,
        "header_b64z": base64.b64encode(header_blob).decode() if False else
        base64.b64encode(__import__("zstandard").ZstdCompressor(level=19).compress(header_blob)).decode(),
        "stats": {"tensors": len(t_entries), "packets": planned_pkts, "planned_only": True},
        "tensors": t_entries,
        "layers": build_layers([t["name"] for t in t_entries], config),
    }
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(manifest, open(args.out, "w"))
    print(f"[graph] plan written: {args.out}")
    print(f"[graph] tensors={len(t_entries)} planned packets={planned_pkts} "
          f"file_bytes={tensors[-1][1]['data_offsets'][1] + 8 + hlen:,}")


if __name__ == "__main__":
    main()
