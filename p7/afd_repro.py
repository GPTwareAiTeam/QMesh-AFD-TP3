"""Transport-only repro: N rounds of post/wait/fetch over both AFD peers.

Mirrors what the MoE seam does per layer, with no vLLM in the loop, so a hang
points at the wire (or at this container's ibverbs) rather than at the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time



def main() -> int:
    ap = argparse.ArgumentParser(
        description="AFD-TP3 transport-only post/wait/fetch repro (no vLLM)")
    ap.add_argument("--config", default=os.environ.get("AFD_CONFIG", "afd.json"),
                    help="peer JSON (default: $AFD_CONFIG or ./afd.json)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None,
                    help="layers per round (default: wire geometry from afd_session)")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--split", action="store_true",
                    help="post every layer, then wait every layer")
    args = ap.parse_args()

    # Imported only after argument parsing, so --help needs no torch or GPU.
    import torch  # noqa: PLC0415
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from afd_session import K, LAYERS, TOPK, PeerSession  # noqa: PLC0415,E402

    if args.layers is None:
        args.layers = LAYERS


    with open(args.config) as f:
        cfg = json.load(f)
    sessions = []
    for p in cfg["peers"]:
        t0 = time.time()
        s = PeerSession(p["dev"], p["gid"], p["ip"], p["port"],
                        layers=cfg["layers"], rows=args.rows, iters=0, device=0)
        print(f"peer {p['ip']}:{p['port']} up in {time.time() - t0:.2f}s", flush=True)
        sessions.append(s)

    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(args.rows, K, generator=g).to("cuda").to(torch.bfloat16)
    ids = torch.randint(0, 384, (args.rows, TOPK), generator=g).to("cuda", torch.int32)
    tw = torch.rand(args.rows, TOPK, generator=g, dtype=torch.float32).to("cuda")
    tw = tw / tw.sum(-1, keepdim=True)

    for r in range(args.rounds):
        t0 = time.time()
        for s in sessions:
            s.begin_round()
        if args.split:
            for li in range(args.layers):
                for s in sessions:
                    s.post(li, x, ids, tw)
            for li in range(args.layers):
                for s in sessions:
                    s.wait(li)
                    part = s.fetch(li)
        else:
            for li in range(args.layers):
                for s in sessions:
                    s.post(li, x, ids, tw)
                    s.wait(li)
                    part = s.fetch(li)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1e3
        print(f"round {r} {args.layers} layers in {dt:.1f} ms "
              f"({dt / args.layers:.3f} ms/layer) sum={float(part.float().abs().sum()):.3f}",
              flush=True)

    for s in sessions:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
