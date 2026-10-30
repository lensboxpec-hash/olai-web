#!/usr/bin/env python3
"""
OLAI-GRAPH v1 unpacker: rebuild the ORIGINAL safetensors bit-for-bit from
packets + manifest. This is the losslessness proof: if the rebuilt file's
sha256 equals the patient's published sha256, the packet graph carried the
weights with zero damage — no pruning, no quantization, nothing lost.
"""
import argparse
import base64
import hashlib
import json
import os
import struct
import sys

import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olai_pkt import parse_packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-sha256", default=None)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    header = zstd.ZstdDecompressor().decompress(base64.b64decode(man["header_b64z"]))
    h = hashlib.sha256()
    with open(args.out, "wb") as out:
        head8 = struct.pack("<Q", len(header))
        out.write(head8)
        out.write(header)
        h.update(head8)
        h.update(header)
        n = 0
        for t in man["tensors"]:
            for p in sorted(t["packets"], key=lambda q: q["part"]):
                blob = open(os.path.join(args.packets, p["f"]), "rb").read()
                d = parse_packet(blob)
                if d["name"] != t["name"] or d["part"] != p["part"]:
                    raise ValueError(f"packet identity mismatch: {p['f']}")
                out.write(d["raw"])
                h.update(d["raw"])
                n += 1
                if n % 50 == 0:
                    print(f"[unpack] {n} packets rebuilt...", flush=True)
    digest = h.hexdigest()
    print(f"[unpack] rebuilt {n} packets -> {args.out}")
    print(f"[unpack] file sha256  = {digest}")
    if args.expect_sha256:
        print(f"[unpack] expected     = {args.expect_sha256}")
        ok = digest == args.expect_sha256
        print(f"[unpack] BIT-EXACT ROUNDTRIP: {'YES — nothing lost' if ok else 'NO — DO NOT SHIP'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
