"""W1: build AFD-TP3 rank packs from the LibertAI NVFP4 checkpoint.

Each rank pack is a directory of safetensors files plus MANIFEST.json:
  p7  : every non-expert, non-Engram-table, non-vision tensor (attention, shared experts, router,
        HC, Engram projections, embed, head, norms, DSpark mtp.*) + routed-expert columns [0:576)
  gx1 : routed-expert columns [576:1440)
  gx2 : routed-expert columns [1440:2304)

Routed experts are stacked per layer and projection so a rank loads one contiguous tensor per
(layer, projection, kind):
  L{l}.w13.weight        U8      [E=384, 2*C, 2560]   gate rows then up rows, C = slice width
  L{l}.w13.weight_scale  F8_E4M3 [E, 2*C, 320]
  L{l}.w2.weight         U8      [E, 5120, C/2]
  L{l}.w2.weight_scale   F8_E4M3 [E, 5120, C/16]
  L{l}.w13.weight_scale_2 / L{l}.w2.weight_scale_2   F32 [E, 2] / [E] (w1 and w3 share; asserted)
  L{l}.w13.input_scale    / L{l}.w2.input_scale       F32 [E, 2] / [E] (W4A4 per-tensor activation
      scales; vLLM's ModelOpt NVFP4 MoE registers w13_input_scale [E, 2] and w2_input_scale [E] and
      reads them at process_weights_after_loading, so a pack without them loads uninitialised garbage)
NVFP4 codes are 2 per byte along K, 16-element E4M3 block scales along K, one FP32 global scale per
projection. Slicing w1/w3 by rows and w2 by K-columns keeps every byte and scale block intact.

Paths are generic defaults that match the layout created by tools/download.sh
(MODEL_ROOT=/models/dsv41). Pass --src/--out for any other checkout:
  --src  NVFP4 checkpoint directory containing model.safetensors.index.json and its shards
         (default: /models/dsv41/DeepSeek-V4.1-Flash-1017-NVFP4)
  --out  pack root, created as needed
         (default: /models/dsv41/packs/afd-tp3-576-864-864)
The builder is specific to the pinned checkpoint geometry (384 experts, 40 layers); it does not
convert or requantize weights. Engram embed tables and vision tensors are not copied into the P7
dense files.

Verification: for every (layer, projection) the concatenation of the three ranks' slices is
byte-identical to the source tensors (checked while writing).
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULT_SRC = "/models/dsv41/DeepSeek-V4.1-Flash-1017-NVFP4"
DEFAULT_OUT = "/models/dsv41/packs/afd-tp3-576-864-864"
SPLIT = {"p7": (0, 576), "gx1": (576, 1440), "gx2": (1440, 2304)}
N_EXP, INTER, HID, LAYERS = 384, 2304, 5120, 40
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|weight_scale|weight_scale_2|input_scale)$")
DT = {"U8": np.uint8, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.int8, "BF16": np.uint16,
      "F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64}


def read_headers(root):
    idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    T = {}
    for s in sorted(set(idx.values())):
        with open(f"{root}/{s}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            a, b = v["data_offsets"]
            T[k] = dict(shard=s, dtype=v["dtype"], shape=v["shape"], off=8 + n + a, nbytes=b - a)
    return T


class Reader:
    def __init__(self, root):
        self.root, self.fds = root, {}

    def raw(self, t):
        fd = self.fds.get(t["shard"])
        if fd is None:
            fd = self.fds[t["shard"]] = os.open(f"{self.root}/{t['shard']}", os.O_RDONLY)
        return os.pread(fd, t["nbytes"], t["off"])

    def arr(self, t):
        a = np.frombuffer(self.raw(t), dtype=DT[t["dtype"]])
        return a.reshape(t["shape"]) if t["shape"] else a


class Writer:
    """Writes one safetensors file per layer chunk; tensors are appended in order."""

    def __init__(self, path):
        self.path, self.items = path, []

    def add(self, name, dtype, arr):
        a = np.asarray(arr)
        self.items.append((name, dtype, np.ascontiguousarray(a) if a.ndim else a))

    def close(self):
        header, off = {}, 0
        for name, dtype, a in self.items:
            header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
            off += a.nbytes
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        h = hashlib.sha256()
        tmp = self.path + ".part"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            for _, _, a in self.items:
                b = a.tobytes()
                f.write(b)
                h.update(b)
        os.replace(tmp, self.path)
        sizes = {n: a.nbytes for n, _, a in self.items}
        self.items = []
        return h.hexdigest(), sizes


PER_EXPERT_RANKS = {"p7"}


def emit_rank(layer, r, a, b, src, w):
    """Write one rank's slice of a layer into ``w``."""
    if r in PER_EXPERT_RANKS:
        # vLLM's DeepSeek-V4.1 loader maps checkpoint keys per expert
        # (experts.{E}.w1.*) onto the fused w13_/w2_ parameters; fused
        # [E, ...] checkpoint keys are not accepted on that path.
        for e in range(N_EXP):
            for p in ("w1", "w3"):
                w.add(f"L{layer}.e{e}.{p}.weight", "U8", src[(p, "weight")][e, a:b])
                w.add(f"L{layer}.e{e}.{p}.weight_scale", "F8_E4M3", src[(p, "weight_scale")][e, a:b])
                w.add(f"L{layer}.e{e}.{p}.weight_scale_2", "F32", src[(p, "weight_scale_2")][e])
                w.add(f"L{layer}.e{e}.{p}.input_scale", "F32", src[(p, "input_scale")][e])
            w.add(f"L{layer}.e{e}.w2.weight", "U8", src[("w2", "weight")][e, :, a // 2 : b // 2])
            w.add(f"L{layer}.e{e}.w2.weight_scale", "F8_E4M3", src[("w2", "weight_scale")][e, :, a // 16 : b // 16])
            w.add(f"L{layer}.e{e}.w2.weight_scale_2", "F32", src[("w2", "weight_scale_2")][e])
            w.add(f"L{layer}.e{e}.w2.input_scale", "F32", src[("w2", "input_scale")][e])
        return
    for kind, dt in (("weight", "U8"), ("weight_scale", "F8_E4M3")):
        w13 = np.concatenate([src[("w1", kind)][:, a:b], src[("w3", kind)][:, a:b]], axis=1)
        w.add(f"L{layer}.w13.{kind}", dt, w13)
    w.add(f"L{layer}.w2.weight", "U8", src[("w2", "weight")][:, :, a // 2 : b // 2])
    w.add(f"L{layer}.w2.weight_scale", "F8_E4M3", src[("w2", "weight_scale")][:, :, a // 16 : b // 16])
    # vLLM keeps both GEMM1 shards' global scales and a per-tensor activation
    # scale per shard; w1 and w3 share both (asserted above).
    w.add(f"L{layer}.w13.weight_scale_2", "F32", np.stack([src[("w1", "weight_scale_2")]] * 2, axis=1))
    w.add(f"L{layer}.w13.input_scale", "F32", np.stack([src[("w1", "input_scale")]] * 2, axis=1))
    w.add(f"L{layer}.w2.weight_scale_2", "F32", src[("w2", "weight_scale_2")])
    w.add(f"L{layer}.w2.input_scale", "F32", src[("w2", "input_scale")])


def check_rank(layer, r, a, b, src, w):
    """Slice check against the source; valid for any rank subset."""
    got = dict((n, x) for n, _, x in w.items)
    c = b - a
    per_ex = r in PER_EXPERT_RANKS
    want = (
        2 * (c * (HID // 2) * N_EXP + c * (HID // 16) * N_EXP)  # w1+w3 weight + weight_scale
        + HID * (c // 2) * N_EXP + HID * (c // 16) * N_EXP  # w2 weight + weight_scale
        + (6 if per_ex else 4) * 4 * N_EXP  # per-expert/per-shard global + activation scales
    )
    assert sum(v.nbytes for v in got.values()) == want, f"L{layer} {r} byte total"
    e = 0
    for p in ("w1", "w3"):
        for kind in ("weight", "weight_scale"):
            name = f"L{layer}.e{e}.{p}.{kind}" if per_ex else f"L{layer}.{p}.{kind}"
            arr = got[name]
            sl = arr if per_ex else (arr[:, :c] if p == "w1" else arr[:, c:])
            assert np.array_equal(sl, src[(p, kind)][e, a:b]), f"L{layer} {r} {p}.{kind} slice"
    for kind, div in (("weight", 2), ("weight_scale", 16)):
        name = f"L{layer}.e{e}.w2.{kind}" if per_ex else f"L{layer}.w2.{kind}"
        arr = got[name][None]
        assert np.array_equal(arr, src[("w2", kind)][e : e + 1, :, a // div : b // div]), f"L{layer} {r} w2.{kind} slice"
    if per_ex:
        e = N_EXP - 1
        for p in ("w1", "w3"):
            assert np.array_equal(got[f"L{layer}.e{e}.{p}.weight"], src[(p, "weight")][e, a:b]), f"L{layer} {r} last {p}"
        assert got[f"L{layer}.e{e}.w2.weight"].shape == (HID, c // 2)
        for p in ("w1", "w3", "w2"):
            assert got[f"L{layer}.e{e}.{p}.weight_scale_2"].shape == ()
            assert got[f"L{layer}.e{e}.{p}.input_scale"].shape == ()


def build_layer(layer, T, R, out, split=SPLIT):
    """Slice one MoE layer for every rank in `split`; returns {rank: (file, sha, sizes)}."""
    w = {r: Writer(f"{out}/{r}/experts-L{layer:02d}.safetensors") for r in split}
    g = lambda e, p, kind: T[f"layers.{layer}.ffn.experts.{e}.{p}.{kind}"]
    src = {}
    for p in ("w1", "w3", "w2"):
        for kind in ("weight", "weight_scale"):
            src[(p, kind)] = np.stack([R.arr(g(e, p, kind)) for e in range(N_EXP)])
        src[(p, "weight_scale_2")] = np.array([R.arr(g(e, p, "weight_scale_2"))[0] for e in range(N_EXP)], dtype=np.float32)
        src[(p, "input_scale")] = np.array([R.arr(g(e, p, "input_scale"))[0] for e in range(N_EXP)], dtype=np.float32)
    for kind in ("weight_scale_2", "input_scale"):
        assert np.array_equal(src[("w1", kind)], src[("w3", kind)]), f"L{layer} w1/w3 {kind} differ"
    assert src[("w1", "weight")].shape == (N_EXP, INTER, HID // 2) and src[("w2", "weight")].shape == (N_EXP, HID, INTER // 2)

    for r, (a, b) in split.items():
        emit_rank(layer, r, a, b, src, w[r])
    for r, (a, b) in split.items():
        check_rank(layer, r, a, b, src, w[r])

    res = {}
    for r in split:
        sha, sizes = w[r].close()
        res[r] = (f"experts-L{layer:02d}.safetensors", sha, sizes)
    return res


def build_p7_dense(T, R, out):
    """Copy every non-routed-expert tensor P7 needs, unchanged, into p7/dense-*.safetensors."""
    skip = lambda k: EXPERT_RE.match(k) or ".engram.embed." in k or k.startswith(("vision.", "aligner.", "image_"))
    names = sorted(k for k in T if not skip(k))
    files, chunk, cur, n = {}, [], 0, 0
    def flush():
        nonlocal chunk, cur, n
        if not chunk:
            return
        wr = Writer(f"{out}/p7/dense-{n:02d}.safetensors")
        for k in chunk:
            t = T[k]
            wr.add(k, t["dtype"], np.frombuffer(R.raw(t), dtype=np.uint8).reshape(-1))
        sha, sizes = wr.close()
        files[f"dense-{n:02d}.safetensors"] = dict(sha256=sha, tensors={k: dict(dtype=T[k]["dtype"], shape=T[k]["shape"], nbytes=T[k]["nbytes"]) for k in chunk})
        chunk, cur, n = [], 0, n + 1
    for k in names:
        chunk.append(k)
        cur += T[k]["nbytes"]
        if cur >= 4 << 30:
            flush()
    flush()
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Build AFD-TP3 rank packs from the pinned NVFP4 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="NVFP4 checkpoint directory (model.safetensors.index.json + shards)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="rank-pack output root; created as needed")
    ap.add_argument("--layers", default="all", help="all, or a comma list of layer indices")
    ap.add_argument("--workers", type=int, default=4, help="parallel layer builders")
    ap.add_argument("--ranks", default="all", help="comma list of ranks to rebuild, or all")
    ap.add_argument("--skip-dense", action="store_true", help="leave p7/dense-*.safetensors alone")
    args = ap.parse_args()
    src, out = args.src, args.out
    T = read_headers(src)
    layers = range(LAYERS) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    ranks = list(SPLIT) if args.ranks == "all" else args.ranks.split(",")
    split = {r: SPLIT[r] for r in ranks}
    for r in split:
        os.makedirs(f"{out}/{r}", exist_ok=True)
    t0 = time.time()
    # Partial rebuilds keep the untouched entries of the existing manifest.
    manifest = {}
    for r in split:
        p = f"{out}/{r}/MANIFEST.json"
        prev = json.load(open(p)) if os.path.exists(p) else {}
        manifest[r] = dict(prev, split=SPLIT[r], files=dict(prev.get("files", {})))
    def one(layer):
        res = build_layer(layer, T, Reader(src), out, split)
        print(f"[{time.time()-t0:7.1f}s] layer {layer} ok", flush=True)
        return layer, res
    with ThreadPoolExecutor(args.workers) as ex:
        for layer, res in ex.map(one, layers):
            for r, (fn, sha, sizes) in res.items():
                manifest[r]["files"][fn] = dict(sha256=sha, bytes=sum(sizes.values()))
    if args.layers == "all" and not args.skip_dense:
        dense = build_p7_dense(T, Reader(src), out)
        for fn, d in dense.items():
            manifest["p7"]["files"][fn] = d
        print(f"[{time.time()-t0:7.1f}s] p7 dense ok ({len(dense)} files)", flush=True)
    meta = dict(source=src, source_repo="LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4", source_rev="5be99cd15675589fb6693d0a8651e413bcb9a396",
                layout_version=2, split=SPLIT, built=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for r in split:
        with open(f"{out}/{r}/MANIFEST.json", "w") as f:
            json.dump({**manifest[r], **meta}, f, indent=1)
    print("DONE", {r: round(sum(v.get("bytes", 0) for v in manifest[r]["files"].values()) / 2**30, 2) for r in split})


if __name__ == "__main__":
    sys.exit(main())
