"""
OLAI-GRAPH v1 — packet codec core (magic 'OLPG').

A packet is a self-describing, content-addressed slice of ONE tensor of a
safetensors file. Nothing is deleted, no precision is changed: the payload
bytes are the patient's own weight bytes (optionally zstd-wrapped, which is
bit-exact on decode). The manifest is the graph; packets are the nodes.

Wire layout (little endian):
  magic   4s   'OLPG'
  version H    1
  codec   B    0=raw  1=zstd (payload bytes decode bit-exact either way)
  flags   B    reserved
  tensor_id  I
  part    I    1-based
  parts   I
  raw_len Q    uncompressed payload length in bytes
  name_len H   utf8 tensor name length
  name    ...  utf8 tensor name
  sha256  32B  sha256 of the UNCOMPRESSED payload
  payload ...  codec bytes
"""
import hashlib
import struct

import zstandard as zstd

MAGIC = b"OLPG"
VERSION = 1
CODEC_RAW = 0
CODEC_ZSTD = 1

_HDR = struct.Struct("<4sHBBIIIQH")  # 30 bytes before name


def pack_bytes(tensor_id, part, parts, name, raw, want_codec=CODEC_ZSTD, level=12):
    """Return one packet blob. Falls back to raw if zstd does not shrink."""
    if want_codec == CODEC_ZSTD:
        payload = zstd.ZstdCompressor(level=level).compress(raw)
        if len(payload) < len(raw):
            codec = CODEC_ZSTD
        else:
            codec, payload = CODEC_RAW, raw
    else:
        codec, payload = CODEC_RAW, raw
    nb = name.encode("utf-8")
    sha = hashlib.sha256(raw).digest()
    head = _HDR.pack(MAGIC, VERSION, codec, 0, tensor_id, part, parts, len(raw), len(nb))
    return head + nb + sha + payload


def parse_packet(blob):
    """Decode a packet; returns dict with uncompressed raw bytes. Raises on corruption."""
    magic, ver, codec, flags, tid, part, parts, raw_len, nlen = _HDR.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    off = _HDR.size
    name = blob[off:off + nlen].decode("utf-8")
    off += nlen
    sha = blob[off:off + 32]
    off += 32
    payload = blob[off:]
    if codec == CODEC_ZSTD:
        raw = zstd.ZstdDecompressor().decompress(payload, max_output_size=raw_len)
    else:
        raw = payload
    if len(raw) != raw_len:
        raise ValueError(f"raw_len mismatch: header={raw_len} got={len(raw)}")
    if hashlib.sha256(raw).digest() != sha:
        raise ValueError(f"sha256 mismatch in packet tensor={name} part={part}")
    return {
        "version": ver, "codec": codec, "tensor_id": tid,
        "part": part, "parts": parts, "name": name, "sha256": sha.hex(), "raw": raw,
    }
