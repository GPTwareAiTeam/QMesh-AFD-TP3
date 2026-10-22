"""Python-side proof of the AFD-TP3 transport seam (W5).

Drives one or more live GX edges through `PeerSession` exactly as an MoE layer
will: post(layer, x, ids, tw) -> wait(layer) -> fetch(layer), for every layer of
the model, and reports the per-phase cost.

Two modes:
  identity  the GX worker copies its in-slot to its out-slot, so every fetched
            partial must be bit-identical to what was posted for THAT layer.
            This checks the whole Python -> verbs -> worker -> verbs -> Python
            path, including per-layer slot addressing, without loading weights.
  moe       the real 40-layer arena answers. The partials are the experts'
            output; routing is verified cross-end against the worker's
            `rout_recv isum/wsum` round line.

Peers are given as repeatable `--peer dev:gid:mgmt_ip:port`; with several peers
one layer posts to all of them before waiting, which is the deployment order
(GX1 and GX2 are independent links and independent GPUs).
Run the GX side first (worker --iters 0, then the serve edge), on that GX, e.g.:
    setsid nohup qmesh_afd_edge.$(uname -m) --role server \
        --dev <gx-rdma-dev> --gid-idx <gid> --mgmt-ip <gx-address> --mgmt-port 47101 \
        --layers 40 --rows 1 --iters 0 --shm /dev/shm/qafd --serve > /tmp/serve_server.log 2>&1 &
Peers are explicit (`--peer`); nothing here assumes a host name or address.
`--help` prints the interface without importing torch, so it runs on a CPU-only host.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# The two GX rails use distinct GID indexes: the wrong index fails at ``qp rtr``
# with ENODATA, and the edge's schedule check then reports a layer mismatch, so
# keep the rails distinct. GID values are per fabric, hence no built-in peer.


def parse_peers(specs: list[str]) -> list[tuple[str, int, str, int]]:
    out = []
    for spec in specs:
        dev, gid, ip, port = spec.split(":")
        out.append((dev, int(gid), ip, int(port)))
    return out


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)

    sessions = [PeerSession(dev, gid, ip, port, layers=args.layers, rows=args.rows,
                            iters=0, device=args.device)
                for dev, gid, ip, port in parse_peers(args.peer)]

    post_us, wait_us, fetch_us, tot_us = [], [], [], []
    polls = 0
    mismatches = 0
    isum = 0
    wsum = 0.0
    sums: list[float] = []

    for r in range(args.rounds):
        # Per-layer deterministic activations: a swapped slot or a stale flag
        # would return another layer's tensor, which the comparison catches.
        g = torch.Generator(device="cpu").manual_seed(1000 + r)
        xs = [torch.randn(args.rows, K, generator=g).to(torch.bfloat16).to(args.device)
              for _ in range(args.layers)]
        ids = torch.randint(0, 384, (args.rows, TOPK), generator=g).to(args.device).to(torch.int32)
        tw = torch.rand(args.rows, TOPK, generator=g, dtype=torch.float32)
        tw = (tw / tw.sum(-1, keepdim=True)).to(args.device)
        risum = int(ids.sum(dtype=torch.int64).item())
        rwsum = float(tw.sum(dtype=torch.float64).item())
        isum += risum
        wsum += rwsum

        for s in sessions:
            s.begin_round()
        parts = None
        for L in range(args.layers):
            t0 = time.perf_counter()
            for s in sessions:
                s.post(L, xs[L], ids, tw)
            t1 = time.perf_counter()
            for s in sessions:
                polls += s.wait(L)
            t2 = time.perf_counter()
            parts = [s.fetch(L) for s in sessions]
            t3 = time.perf_counter()
            if r >= args.warmup:
                post_us.append((t1 - t0) * 1e6 / len(sessions))
                wait_us.append((t2 - t1) * 1e6)
                fetch_us.append((t3 - t2) * 1e6 / len(sessions))
                tot_us.append((t3 - t0) * 1e6)
            if args.mode == "identity":
                # The wire carries e4m3 under a per-post scale, so identity mode
                # returns the posted rows quantized, not bit-identical: the
                # reference is the same cast the session does.
                sc = FP8_MAX / float(xs[L].abs().amax().clamp_min(1e-6))
                ref = ((xs[L] * sc).clamp(-FP8_MAX, FP8_MAX)).to(torch.float8_e4m3fn)
                ref = ref.to(torch.bfloat16).div_(sc)
                for i, part in enumerate(parts):
                    if not torch.equal(part, ref):
                        mismatches += 1
                        d = (part.float() - xs[L].float()).abs().max().item()
                        print(f"MISMATCH layer={L} round={r} peer={i} max|d|={d:.4g}")
            elif len(sessions) > 1:
                summed = parts[0]
                for part in parts[1:]:
                    summed = summed + part
                sums.append(float(summed.float().abs().sum().item()))
        print(f"round {r}: {args.layers} layers done, "
              f"{tot_us[-1] if tot_us else 0:.1f} us/layer (last); "
              f"rout_sent isum={risum} wsum={rwsum:.6f}"
              + "".join(f"; peer{i} |partial|={p.float().abs().sum().item():.4f}"
                        for i, p in enumerate(parts))
              + (f"; sum |partials|={sums[-1]:.4f}" if args.mode == "moe" and sums else ""))

    def stats(v: list[float]) -> str:
        if not v:
            return "n=0"
        return (f"mean={statistics.mean(v):.2f} p50={statistics.median(v):.2f} "
                f"p95={sorted(v)[int(len(v) * 0.95)]:.2f} max={max(v):.2f}")

    n = len(tot_us)
    print(f"RESULT mode={args.mode} peers={len(sessions)} layers={args.layers} rows={args.rows} "
          f"rounds={args.rounds} n={n} polls_per_layer={polls / max(1, n):.2f}")
    print(f"  post  {stats(post_us)}  (per peer)")
    print(f"  wait  {stats(wait_us)}  (all peers)")
    print(f"  fetch {stats(fetch_us)}  (per peer)")
    print(f"  total {stats(tot_us)}  -> {n / (sum(tot_us) / 1e6) if n else 0:.0f} layers/s")
    print(f"  rout_sent isum={isum} wsum={wsum:.6f}")
    if args.mode == "identity":
        print(f"RESULT identity mismatches={mismatches} of {n * len(sessions)} layer-fetches")
    for s in sessions:
        s.close()


def main() -> None:
    global torch, FP8_MAX, K, LAYERS, TOPK, PeerSession
    p = argparse.ArgumentParser(
        description="AFD-TP3 Python transport smoke (identity or moe)")
    p.add_argument("--mode", choices=("identity", "moe"), default="identity")
    p.add_argument("--peer", action="append", metavar="DEV:GID:MGMT_IP:PORT",
                   help="repeatable; one GX edge management endpoint each "
                        "(required)")
    p.add_argument("--layers", type=int, default=None,
                   help="layers per round (default: wire geometry from afd_session)")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    if not args.peer:
        p.error("at least one --peer DEV:GID:MGMT_IP:PORT is required")

    # Imported only after argument parsing, so --help needs no torch or GPU.
    import torch  # noqa: PLC0415
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "afd_overlay"))
    from afd_session import FP8_MAX, K, LAYERS, TOPK, PeerSession  # noqa: PLC0415,E402

    if args.layers is None:
        args.layers = LAYERS
    run(args)


if __name__ == "__main__":
    main()
