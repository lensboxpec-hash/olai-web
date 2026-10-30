#!/bin/bash
# Upload LFM2.5 packets to orphan branch weights-lfm25 in batches (~60 pkts / ~75MB per push).
set -e
cd /tmp/wtree-lfm25
SRC=/tmp/lfm25-pkts
mkdir -p pkts
n=0
batch=0
for f in $(ls "$SRC"/*.pkt | sort); do
  cp "$f" pkts/
  n=$((n+1))
  if [ $((n % 60)) -eq 0 ] || [ "$n" = "423" ]; then
    batch=$((batch+1))
    git add pkts >/dev/null
    git commit -q -m "LFM2.5-350M packets batch ${batch} (${n}/423) — sha256 chain per graph/lfm25-full-graph.json"
    echo "pushing batch ${batch} (${n} packets)..."
    git push -q origin weights-lfm25
    echo "  pushed."
  fi
done
echo "DONE: ${n} packets in ${batch} batches"
