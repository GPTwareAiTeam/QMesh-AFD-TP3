"""Gate A transport benchmark for AFD-TP3.

Two independent 2-rank NCCL worlds, one per physical rail (the triangle is not
a flat subnet). Each rail is a dedicated point-to-point link; addresses and
device names come from the environment, never from this file.
  link 1: P7 GPU0 <-> GX1
  link 2: P7 GPU0 <-> GX2
P7 runs one process per link, both on GPU0 (only GPU0 is visible in the
container).

Per MoE layer, P7 sends x_norm + routing meta (R x (5120+24) bf16) and receives
an FP32 partial (R x 5120); the GX "computes" a copy. 40 layers = one decode
step. Measured eager and as one captured CUDA graph per step. The two links run
concurrently, as they would in serving.
Prints one JSON line per R.

Environment (set by the launcher, see run_xfer.sh):
  ROLE         p7 | gx
  LINK         rail index (free-form, copied to the output row)
  MASTER_ADDR, MASTER_PORT   TCPStore/NCCL bootstrap address on this rail
  LAYERS       layers per step (default 40)
  ROWS         comma-separated row counts (default 1,6,48)
  ITERS        timed iterations per row (default 200)
  WARM         warmup iterations (default 20)
NCCL_* variables are read by NCCL itself.

`--help` only documents this interface and needs no GPU, torch, or network.
"""

import argparse
import json
import os
import time

H, META = 5120, 24

HELP_EPILOG = """\
environment:
  ROLE=p7|gx  LINK=<rail>  MASTER_ADDR=<addr>  MASTER_PORT=<port>
  LAYERS=40  ROWS=1,6,48  ITERS=200  WARM=20
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gate A transport benchmark (two isolated 2-rank NCCL rails)",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.parse_args(argv)  # -h/--help exits here, before torch is imported

    import torch  # noqa: PLC0415  (imported after --help so CPU-only hosts can print usage)
    import torch.distributed as dist  # noqa: PLC0415

    role = os.environ["ROLE"]  # "p7" or "gx"
    link = int(os.environ["LINK"])
    layers = int(os.environ.get("LAYERS", "40"))
    rows = [int(r) for r in os.environ.get("ROWS", "1,6,48").split(",")]
    iters = int(os.environ.get("ITERS", "200"))
    warm = int(os.environ.get("WARM", "20"))
    rank = 0 if role == "p7" else 1
    peer = 1 - rank

    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=2,
        device_id=dev,
    )

    def step(xs, parts):
        # Both ranks issue matching ops in matching groups: group 1 = x (P7 -> GX), group 2 = partial (GX -> P7).
        for layer in range(layers):
            if role == "p7":
                dist.send(xs[layer], peer)
                dist.recv(parts[layer], peer)
            else:
                dist.recv(xs[layer], peer)
                parts[layer].copy_(xs[layer][:, :H])
                dist.send(parts[layer], peer)

    def timed(fn):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    for R in rows:
        xs = [torch.randn(R, H + META, device=dev).to(torch.bfloat16) for _ in range(layers)]
        parts = [torch.zeros(R, H, device=dev, dtype=torch.float32) for _ in range(layers)]
        fn = lambda: step(xs, parts)
        for _ in range(warm):
            fn()
        eager_ms = timed(fn)
        ok = True
        if role == "p7":  # data check: returned partial equals what we sent
            ok = all(torch.equal(parts[l], xs[l][:, :H].float()) for l in range(layers))

        graph_ms, err = None, None
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            dist.barrier()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(warm):
                g.replay()
            graph_ms = timed(g.replay)
            if role == "p7":
                for l in range(layers):
                    parts[l].zero_()
                xs[0].normal_()
                g.replay()
                torch.cuda.synchronize()
                ok = ok and all(torch.equal(parts[l], xs[l][:, :H].float()) for l in range(layers))
            else:
                g.replay()
                torch.cuda.synchronize()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]

        print(json.dumps({
            "role": role, "link": link, "R": R, "layers": layers,
            "send_B": R * (H + META) * 2, "return_B": R * H * 4,
            "eager_ms_step": round(eager_ms, 3), "eager_us_layer": round(eager_ms * 1e3 / layers, 1),
            "graph_ms_step": None if graph_ms is None else round(graph_ms, 3),
            "graph_us_layer": None if graph_ms is None else round(graph_ms * 1e3 / layers, 1),
            "data_ok": ok, "graph_error": err,
        }), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
