"""Gate A memory probe for a GX expert worker.

Allocates the planned arena (default 106.8 GiB for the 864-column slice) plus worker buffers in
1 GiB chunks, touches every page, then reports MemAvailable and CUDA mem_get_info at each stage.
Stops early (and reports) if host MemAvailable falls below the floor. Frees everything on exit.

Configuration is CLI flags or the same values as environment variables
(flags win; defaults are the measured 864-column plan):
  ARENA_GIB / --arena-gib   arena size in GiB (default 106.8)
  EXTRA_GIB / --extra-gib   worker buffers in GiB (default 1.0)
  FLOOR_GIB / --floor-gib   stop below this host MemAvailable (default 4.0)

`--help` only documents this interface and needs no GPU or torch.
"""

import argparse
import json
import os
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Gate A CUDA arena/memory probe (GX expert worker)")
    ap.add_argument("--arena-gib", type=float,
                    default=float(os.environ.get("ARENA_GIB", "106.8")))
    ap.add_argument("--extra-gib", type=float,
                    default=float(os.environ.get("EXTRA_GIB", "1.0")))
    ap.add_argument("--floor-gib", type=float,
                    default=float(os.environ.get("FLOOR_GIB", "4.0")),
                    help="stop if host MemAvailable would fall below this")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)  # -h/--help exits here, before torch is imported

    import torch  # noqa: PLC0415  (so --help works on a CPU-only host)

    GiB = 1 << 30
    target = args.arena_gib
    extra = args.extra_gib
    floor = args.floor_gib

    def mem_available():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / (1 << 20)

    def snap(stage, n):
        free, total = torch.cuda.mem_get_info()
        return {"stage": stage, "allocated_gib": round(n, 2), "host_memavail_gib": round(mem_available(), 2),
                "cuda_free_gib": round(free / GiB, 2), "cuda_total_gib": round(total / GiB, 2),
                "torch_reserved_gib": round(torch.cuda.memory_reserved() / GiB, 2)}

    torch.cuda.set_device(0)
    log = [snap("start", 0)]
    chunks, n, stopped = [], 0.0, None
    goal = target + extra
    t0 = time.time()
    while n < goal:
        sz = min(1.0, goal - n)
        if mem_available() < floor + sz:
            stopped = f"host MemAvailable {mem_available():.2f} GiB < floor {floor} + {sz}"
            break
        c = torch.empty(int(sz * GiB), dtype=torch.uint8, device="cuda")
        c.fill_(1)  # touch every page
        chunks.append(c)
        n += sz
        if int(n) % 10 == 0 and abs(n - round(n)) < 1e-6:
            log.append(snap("progress", n))
    torch.cuda.synchronize()
    log.append(snap("full", n))
    # Hold briefly, then a bandwidth read of the whole arena (what decode streams).
    time.sleep(5)
    log.append(snap("held_5s", n))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    s = sum(int(c[:: 1 << 20].sum()) for c in chunks)  # light touch to confirm residency
    torch.cuda.synchronize()
    big = chunks[0]
    for _ in range(3):
        big.add_(1)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rw = 0
    for _ in range(10):
        big.add_(1)
    torch.cuda.synchronize()
    bw = 10 * 2 * big.numel() / (time.perf_counter() - t2) / 1e9
    print(json.dumps({"target_gib": target, "extra_gib": extra, "floor_gib": floor, "reached_gib": round(n, 2),
                      "stopped": stopped, "secs": round(time.time() - t0, 1), "rw_bw_GBps_1GiB": round(bw, 1), "log": log}, indent=1))
    del chunks, big
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
