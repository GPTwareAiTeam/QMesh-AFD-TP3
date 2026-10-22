"""The v7 swizzled NVFP4 decode body: build recipe, import, and body factory.

`gx/kernels/fp4_decode_moe_swz.cu` (KernelFinish) computes a decode-row MoE step
straight from the arena's vLLM-swizzled block scales -- no row-major scale copy,
no activation re-quant, two kernel launches. This module owns everything that is
not the kernel itself:

  build   one torch extension from the vendored sources, into a cache dir keyed
          by a hash of the sources, so a checkout is self-contained (the bench
          scratch dir /w/k4 is NOT a dependency).
  load    import that .so, or a prebuilt one from $GX_SWZ_SO / a sibling path.
  body    the `body(li, r, x, ids, tw, out, ws13, ws2, a1g)` callable the graph
          ingest replays, using the arena tensors in place.

The kernel reads `ids` directly, so the sentinel policy is the ingest's: `-1`
padding and any corrupt id are stored as expert 0 with weight 0.0, which the
kernel turns into an exact zero contribution (`g2[0] * 0 == 0`). An id outside
[0, E) would instead index the arena out of bounds, which is why the host
refuses a corrupt post before any replay (`gx_expert_worker.routing_gate`) and
`gx_ingest` never stores one.

Build (inside the container, offline, one nvcc run):

    python3 /w/gx_decode_swz.py build     # -> /w/_ext/fp4_v7/fp4_v7.so
    python3 /w/gx_decode_swz.py check     # load + shape/dtype report

Loading picks, in order: $GX_SWZ_SO, <this dir>/fp4_v7.so, the `_ext/<name>`
build the command above leaves next to this file, then a `$GX_SWZ_CACHE`
(default `~/.cache/gx_swz/v7-<sources hash>`) build; if none exists it builds
into `_ext/`. `$GX_SWZ_SRC` points the source lookup at another directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

K = 5120                # hidden (x, out width; w13's split axis)
C = 864                 # this host's SwiGLU column slice
CP = 896                # C rounded up to the kernel's 128-row blocks
TOPK = 6
NAME = "fp4_v7"
GLUE = "ext_variant.cu"
KERNEL = "fp4_decode_moe_swz.cu"
# GB10. The kernel uses sm_121a-only instructions (e2m1x2 / e4m3x2 converts).
ARCH = ("-gencode", "arch=compute_121a,code=sm_121a")

HERE = Path(__file__).resolve().parent


def source_dir() -> Path:
    """Directory holding both vendored sources (env override, then siblings)."""
    env = os.environ.get("GX_SWZ_SRC")
    cands = [Path(env)] if env else []
    cands += [HERE / "kernels", HERE / "swz", HERE]
    for d in cands:
        if (d / GLUE).is_file() and (d / KERNEL).is_file():
            return d
    raise FileNotFoundError(
        f"{GLUE} + {KERNEL} not found; looked in "
        + ", ".join(str(d) for d in cands)
        + " (or set GX_SWZ_SRC)")


def _key(src: Path) -> str:
    h = hashlib.sha256()
    for name in (GLUE, KERNEL):
        h.update((src / name).read_bytes())
    h.update(" ".join(ARCH).encode())
    h.update(torch.__version__.encode())
    h.update(f"cuda{torch.version.cuda}".encode())
    return h.hexdigest()[:16]


def cache_dir(src: Path) -> Path:
    base = os.environ.get("GX_SWZ_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "gx_swz"
    return root / f"v7-{_key(src)}"


def build(out: str | Path | None = None, verbose: bool = False) -> Path:
    """Compile the extension; return the .so path. `out` = a directory for it.

    Default build directory is `<this dir>/_ext/fp4_v7`, next to the deployed
    sources, so a host is provisioned by one offline nvcc run and `load()` then
    finds it by itself (nothing outside the checkout is consulted).
    """
    from torch.utils.cpp_extension import load

    src = source_dir()
    bdir = Path(out) if out else (HERE / "_ext" / NAME)
    bdir.mkdir(parents=True, exist_ok=True)
    kernel = str(src / KERNEL)
    mod = load(
        name=NAME,
        sources=[str(src / GLUE)],
        extra_cuda_cflags=[
            *ARCH, "-O3", f'-DKERNEL_FILE="\\"{kernel}\\""',
        ],
        build_directory=str(bdir),
        verbose=verbose,
    )
    so = Path(mod.__file__)
    print(f"built {NAME} {so} ({so.stat().st_size} B)", flush=True)
    return so


def candidates() -> list[Path]:
    """Where a built extension may already be, most specific first."""
    out = []
    env = os.environ.get("GX_SWZ_SO")
    if env:
        out.append(Path(env))
    out += [HERE / f"{NAME}.so", HERE / "_ext" / NAME / f"{NAME}.so"]
    try:
        out.append(cache_dir(source_dir()) / f"{NAME}.so")
    except FileNotFoundError:
        pass
    return out


def load(path: str | Path | None = None) -> object:
    """Import the extension module, building it once if no .so is staged."""
    if NAME in sys.modules:
        return sys.modules[NAME]
    so = Path(path) if path else next(
        (p for p in candidates() if p.is_file()), None)
    if so is None:
        so = build()
    spec = importlib.util.spec_from_file_location(NAME, str(so))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {so} as a torch extension module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[NAME]
        raise
    return mod


def make_body(arena, swiglu_limit: float, mod=None):
    """`body(li, r, x, ids, tw, out, ws13, ws2, a1g)` on the arena's own tensors.

    `a1g`/`ws2` are unused: the kernel takes the arena's global scales raw and
    its own fp32 activation scratch, which is a view of `ws13` (the clamped body
    uses the same buffer for bf16 intermediates; only one of the two runs).
    Capture-safe: metadata views only, no allocation, no host sync.
    """
    mod = mod or load()
    run = mod.run
    s13 = [t.view(torch.uint8) if t is not None else None for t in arena.s13]
    s2 = [t.view(torch.uint8) if t is not None else None for t in arena.s2]
    has_limit = bool(swiglu_limit > 0)
    limit = float(max(swiglu_limit, 0.0))

    def body(li: int, r: int, x, ids, tw, out, ws13, ws2, a1g) -> None:
        w13, s13l, g13 = arena.w13[li], s13[li], arena.g13[li]
        w2, s2l, g2 = arena.w2[li], s2[li], arena.g2[li]
        if w13 is None or s13l is None or w2 is None or s2l is None:
            raise RuntimeError(f"layer {li} is not loaded in the arena")
        # The ingest guarantees every stored id names an expert; a corrupt post
        # is refused host-side before this graph can replay.
        if (ids.dtype is not torch.int32 or tw.dtype is not torch.float32
                or x.dtype is not torch.bfloat16
                or out.dtype is not torch.bfloat16):
            raise TypeError(
                f"decode-swz body: ids/tw/x/out must be int32/f32/bf16/bf16, "
                f"got {ids.dtype}/{tw.dtype}/{x.dtype}/{out.dtype}")
        act = ws13.view(torch.float32)[: r * TOPK * C]
        run(out, x, ids, tw, w13, s13l, g13, w2, s2l, g2, act, limit,
            has_limit)
    return body


def check() -> int:
    """Load the extension and report its one entry point and the arena sizes."""
    mod = load()
    sig = (mod.run.__doc__ or "").strip().splitlines()
    first = sig[0].strip() if sig else "run(?)"
    print(f"{NAME}: {mod.__file__}\n  {first}")
    print(f"  w13 [E, {2 * CP}, {K // 2}] u8; s13 [E, {2 * CP}, {K // 16}] u8 "
          f"swizzled; w2 [E, {K}, {CP // 2}] u8; s2 [E, {K}, {CP // 16}] u8; "
          f"act fp32 [{TOPK}*r, {C}]")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "build":
        out = argv[2] if len(argv) > 2 else None
        print(build(out, verbose=True))
        return 0
    if cmd == "check":
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
