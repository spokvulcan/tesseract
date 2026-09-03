#!/usr/bin/env python3
"""Graft a bf16 MTP head into an MLX-quantized Qwen3.5-family checkpoint.

mlx-community quants drop the `mtp.*` tensors the vendor MTP drafter needs.
This script copies them from an official bf16 shard into the quantized model
directory as an extra safetensors shard and patches the weight index. The
loader applies quantization only to modules with `.scales` tensors
(Vendor/mlx-swift-lm Load.swift), so the bf16 head loads as-is beside the
4-bit weights — no config.json changes required.

Usage:
  graft_mtp_head.py --quant-dir <model dir> --bf16-shard <shard with mtp.*>

Writes `model-mtp-head.safetensors` into the quant dir and rewrites
`model.safetensors.index.json` (a backup of the original index is kept as
`model.safetensors.index.json.pre-mtp`).

Single-file checkpoints without an index (the z-lab PARO releases ship one
`model.safetensors`) get the extra shard only: every loader that matters —
the app's `checkpointShipsMTPHead` header scan, the vendor weight loader,
and the PARO loader's `sourceURLs` — enumerates `*.safetensors` in the
directory when no index exists, so no index is fabricated. Note the PARO
loader's Prepared Checkpoint manifest covers every source shard, so the
graft invalidates it and the next load re-converts once.
"""

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def extract_mtp_tensors(shard: Path) -> dict:
    """Return {name: raw bytes + metadata} for every mtp.* tensor in shard."""
    header = read_header(shard)
    data_start = 8 + struct.unpack("<Q", open(shard, "rb").read(8))[0]
    out = {}
    with open(shard, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__" or not name.startswith("mtp."):
                continue
            begin, end = meta["data_offsets"]
            f.seek(data_start + begin)
            out[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "bytes": f.read(end - begin),
            }
    return out


def write_safetensors(path: Path, tensors: dict) -> None:
    header = {}
    offset = 0
    for name, t in sorted(tensors.items()):
        size = len(t["bytes"])
        header[name] = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    # safetensors pads the header to an 8-byte boundary with spaces
    pad = (8 - len(blob) % 8) % 8
    blob += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, t in sorted(tensors.items()):
            f.write(t["bytes"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", type=Path, required=True)
    ap.add_argument("--bf16-shard", type=Path, required=True)
    ap.add_argument("--shard-name", default="model-mtp-head.safetensors")
    args = ap.parse_args()

    index_path = args.quant_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    already = [
        k
        for shard in sorted(args.quant_dir.glob("*.safetensors"))
        for k in read_header(shard)
        if k.startswith("mtp.")
    ]
    if already:
        print(f"directory already has {len(already)} mtp.* entries; nothing to do")
        return 0

    tensors = extract_mtp_tensors(args.bf16_shard)
    if not tensors:
        sys.exit(f"no mtp.* tensors found in {args.bf16_shard}")
    print(f"extracted {len(tensors)} mtp.* tensors from {args.bf16_shard.name}")

    out_shard = args.quant_dir / args.shard_name
    write_safetensors(out_shard, tensors)
    print(f"wrote {out_shard} ({out_shard.stat().st_size / 1e6:.0f} MB)")

    if index is None:
        print("no index in the quant dir (single-file checkpoint); shard-only graft")
    else:
        shutil.copy2(index_path, index_path.with_suffix(".json.pre-mtp"))
        for name in tensors:
            index["weight_map"][name] = args.shard_name
        if "metadata" in index and "total_size" in index["metadata"]:
            index["metadata"]["total_size"] += sum(
                len(t["bytes"]) for t in tensors.values()
            )
        index_path.write_text(json.dumps(index, indent=2))
        print(f"patched {index_path.name} (+{len(tensors)} entries)")

    # Round-trip verification: re-read what we wrote.
    check = read_header(out_shard)
    written = [k for k in check if k != "__metadata__"]
    assert sorted(written) == sorted(tensors), "written shard header mismatch"
    print("verified: shard header round-trips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
