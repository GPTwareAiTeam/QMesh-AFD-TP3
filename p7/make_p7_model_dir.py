#!/usr/bin/env python3
"""Assemble a loadable HF model dir for the P7 AFD slice.

Symlinks the pack's shards into DEST, writes model.safetensors.index.json from
the real per-file key lists, and symlinks the tokenizer/config files from the
NVFP4 checkpoint. No weights are copied or rewritten.

Requires the `safetensors` package (pip install safetensors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


COPY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def keys_of(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="np", device="cpu") as f:
        return list(f.keys())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assemble a loadable HF model dir for the P7 AFD slice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pack", default="/models/dsv41/packs/afd-tp3-576-864-864/p7",
                    help="P7 rank-pack directory written by pack/build_packs.py")
    ap.add_argument("--src", default="/models/dsv41/DeepSeek-V4.1-Flash-1017-NVFP4",
                    help="NVFP4 checkpoint directory supplying config/tokenizer files")
    ap.add_argument("--dest", default="/models/dsv41/p7-model",
                    help="model directory to create/update (symlinks and index only)")
    args = ap.parse_args()
    try:
        import safetensors  # noqa: F401
    except ModuleNotFoundError:
        print("make_p7_model_dir.py requires the safetensors package", file=sys.stderr)
        return 2

    pack, src, dest = (Path(p).resolve() for p in (args.pack, args.src, args.dest))
    dest.mkdir(parents=True, exist_ok=True)

    shards = sorted(pack.glob("*.safetensors"))
    if not shards:
        print(f"no shards in {pack}", file=sys.stderr)
        return 1

    weight_map: dict[str, str] = {}
    total = 0
    for shard in shards:
        link = dest / shard.name
        if link.is_symlink() or link.exists():
            if not link.is_symlink():
                print(f"refusing to overwrite non-symlink {link}", file=sys.stderr)
                return 1
            link.unlink()
        os.symlink(shard, link)
        ks = keys_of(shard)
        dup = [k for k in ks if k in weight_map]
        if dup:
            print(f"duplicate keys in {shard.name}: {dup[:5]}", file=sys.stderr)
            return 1
        for k in ks:
            weight_map[k] = shard.name
        total += shard.stat().st_size

    index = {
        "metadata": {"total_size": total, "format": "pt"},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n")

    for name in COPY_FILES:
        s = src / name
        if not s.exists():
            continue
        d = dest / name
        if d.is_symlink() or d.exists():
            d.unlink()
        os.symlink(s, d)

    print(f"{len(shards)} shards, {len(weight_map)} tensors, {total / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
