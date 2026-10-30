#!/usr/bin/env python3
"""
OLAI-GRAPH v1 verifier: walk the manifest, re-hash every packet on disk,
compare against the manifest. Prints coverage + PASS/FAIL. Does not rebuild.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olai_pkt import parse_packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--packets", required=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    ok = bad = missing = 0
    planned_tensors = 0
    raw = comp = 0
    for t in man["tensors"]:
        if t.get("planned"):
            planned_tensors += 1
            continue
        for p in t["packets"]:
            path = os.path.join(args.packets, p["f"])
            if not os.path.exists(path):
                missing += 1
                print(f"[verify] MISSING {p['f']}")
                continue
            d = parse_packet(open(path, "rb").read())
            if d["sha256"] == p["sha256"] and d["part"] == p["part"]:
                ok += 1
                raw += len(d["raw"])
                comp += os.path.getsize(path)
            else:
                bad += 1
                print(f"[verify] BAD {p['f']}")
    total_pkts = sum(len(t["packets"]) for t in man["tensors"])
    print(f"[verify] packets ok={ok} bad={bad} missing={missing} "
          f"planned-not-packed-tensors={planned_tensors}")
    print(f"[verify] raw={raw / 1e6:.1f}MB on-disk={comp / 1e6:.1f}MB")
    verdict = (bad == 0 and missing == 0)
    print(f"[verify] VERDICT: {'PASS — graph intact' if verdict else 'FAIL'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
