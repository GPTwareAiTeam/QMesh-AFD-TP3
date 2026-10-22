#!/usr/bin/env bash
# Control the GX expert worker inside the long-lived ``gxw1`` container.
#   gx_worker.sh <host|gx1|gx2> up     # (re)start, appends to /w/serve40g.log
#   gx_worker.sh <host|gx1|gx2> down   # stop; the container keeps running
#   gx_worker.sh <host|gx1|gx2> wait   # block until *this* launch has the arena
#
# ``gx1`` and ``gx2`` resolve to GX1_HOST and GX2_HOST (scripts/local.env).
# Other knobs: GX_WORKER_FLAGS, GX_LAYERS, GX_ROWS, GX_SHM, GX_A2, GX_LOG.
# DSV41_SKIP_LOCAL_ENV=1 ignores scripts/local.env for hermetic checks.
#
# ``up`` is what an engine launch needs on both hosts, but a worker restart
# costs a ~106 GiB arena reload (5-7 min), so only restart a host whose pack
# or code changed.  ``/w`` is the container's own copy, NOT the host tree:
# ``/tmp/gxpy`` on the host is only a staging area, and every code change must
# be copied in as ``/tmp/gxpy/gx_expert_worker.py -> gxw1:/w/w.py`` (plus the
# ``gx_*.py`` modules and the vendored ``kernels/`` sources) before ``up``, or
# the container runs the previous build while the host looks up to date.
# ``gx_decode_swz.py`` builds ``--decode-kernel cuda-swz`` on demand from
# ``/w/kernels``; that tree is staged even though the fast kernel stays opt-in.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${DSV41_SKIP_LOCAL_ENV:-0}" != 1 && -f "$SCRIPT_DIR/local.env" ]]; then
  . "$SCRIPT_DIR/local.env"
fi

host=${1:-}
action=${2:-}
case "$host" in
  gx1)
    [[ -n "${GX1_HOST:-}" ]] || { echo "gx_worker: GX1_HOST is not set (scripts/local.env or environment)" >&2; exit 2; }
    host=$GX1_HOST
    ;;
  gx2)
    [[ -n "${GX2_HOST:-}" ]] || { echo "gx_worker: GX2_HOST is not set (scripts/local.env or environment)" >&2; exit 2; }
    host=$GX2_HOST
    ;;
  "")
    echo "usage: gx_worker.sh <host|gx1|gx2> up|down|wait" >&2
    exit 2
    ;;
esac
case "$action" in
  up|down|wait) ;;
  *)
    echo "usage: gx_worker.sh <host|gx1|gx2> up|down|wait" >&2
    exit 2
    ;;
esac
log=${GX_LOG:-/w/serve40g.log}
safe_host=${host//[^A-Za-z0-9_.@-]/_}
off_file=${GX_STATE_DIR:-/tmp}/gxworker.$safe_host.off

case "$action" in
  down)
    # The bracket keeps the pattern from matching the ``sh -c`` that runs it:
    # a literal ``-f /w/w.py`` selects that shell too, pkill kills it, and the
    # caller sees exit 143 instead of a clean stop.
    ssh "$host" "docker exec -u root gxw1 sh -c 'pkill -f \"python3 -u /w/w[.]py\" || true'"
    ;;
  up)
    # The log is append-only across restarts, so remember where this launch
    # starts; ``wait`` greps only the bytes written after it. A host whose
    # ``/tmp`` was wiped has no log yet, and ``set -e`` must not turn that into
    # a silent no-op.
    ssh "$host" "docker exec -u root gxw1 sh -c 'wc -c < $log 2>/dev/null || echo 0'" > "$off_file"
    # ``--a2`` reuses the SwiGLU scales measured on a previous run's real
    # activations. ``--calib N`` would re-measure them over the first N rounds,
    # at the cost of one GEMM1 + device sync per (row, expert) — 6144 of them per
    # layer at the 1024-row capacity, ~1 s per layer — so it stays off: the
    # sidecar already carries scales measured on real traffic, and the engine's
    # first forward is a warmup whose payload must not define them.
    # ``--rows`` is the wire's row capacity, mirrored as R_MAX in
    # src/qmesh_afd_peer.cpp, --rows in restart_afd_edges.sh and AFD_ROWS in the
    # engine; the row count of each post rides in the routing header.
    # Same bracket as ``down``: the plain pattern would also match this shell.
    # ``GX_WORKER_FLAGS`` selects the serving path; the default is the in-graph
    # ingestion with the checkpoint's SwiGLU clamp (rows=1 single-peer A/B:
    # 438.1 us/layer per-layer graphs vs 424.8 round graph at 1 row vs 496.8
    # round graph captured at 16 rows, 154 ms/token before).
    # ``--round-graph --round-rows 1`` is within noise of the per-layer path and
    # only pays off once a round's 40 posts are the bottleneck, so it stays off
    # by default; it must never be captured wider than the posted n (the ingest
    # grid and the CUTLASS body are baked at capture) and the worker falls back
    # to the per-layer path for any other n.
    # The fast ``--decode-kernel cuda-swz`` body remains an explicit override:
    # stage its sources for a fresh build, but do not enable it by default
    # until it is verified end-to-end through the served path.
    # Deploy the staged build before launching: a host-side scp alone leaves
    # the container running its previous copy of any file.
    ssh "$host" 'cd /tmp/gxpy && for f in gx_expert_worker.py gx_ingest.py gx_spin.py gx_decode_swz.py; do
        [ -f "$f" ] || { echo "missing staging file /tmp/gxpy/$f" >&2; exit 1; }
        d=$f; [ "$f" = gx_expert_worker.py ] && d=w.py
        docker cp "$f" "gxw1:/w/$d" >/dev/null
      done
      [ -d kernels ] || { echo "missing staging tree /tmp/gxpy/kernels" >&2; exit 1; }
      # gx_decode_swz.py compiles kernels/{ext_variant.cu,fp4_decode_moe_swz.cu}
      # into /w/_ext on first use. Replace the tree as a unit so a .so built
      # from an older source hash can never outlive its sources.
      docker exec -u root gxw1 rm -rf /w/kernels.stage
      docker cp kernels "gxw1:/w/kernels.stage" >/dev/null
      docker exec -u root gxw1 sh -c "rm -rf /w/kernels && mv /w/kernels.stage /w/kernels"
      docker exec gxw1 md5sum /w/w.py /w/gx_ingest.py /w/gx_spin.py /w/gx_decode_swz.py \
        /w/kernels/fp4_decode_moe_swz.cu /w/kernels/ext_variant.cu
      md5sum /tmp/gxpy/gx_expert_worker.py /tmp/gxpy/gx_ingest.py /tmp/gxpy/gx_spin.py \
        /tmp/gxpy/gx_decode_swz.py /tmp/gxpy/kernels/fp4_decode_moe_swz.cu \
        /tmp/gxpy/kernels/ext_variant.cu'
    flags=${GX_WORKER_FLAGS:---ingest graph --swiglu-limit 10}
    ssh "$host" "docker exec -u root gxw1 sh -c 'pkill -f \"python3 -u /w/w[.]py\" || true'; docker exec -d -u root gxw1 sh -c 'cd /w && nohup python3 -u /w/w.py --serve --mode moe --layers ${GX_LAYERS:-40} --rows ${GX_ROWS:-4096} --iters 0 --shm ${GX_SHM:-/dev/shm/qafd} --calib 0 --a2 ${GX_A2:-/w/a2.json} $flags >> $log 2>&1'"
    ;;
  wait)
    off=$(cat "$off_file")
    ssh "$host" "docker exec -u root gxw1 sh -c 'tail -c +$((off + 1)) $log'" | grep -q "arena loaded" ||
      { echo "arena not loaded yet on $host (see $log)" >&2; exit 1; }
    ;;
  *)
    echo "unknown action $action" >&2; exit 2
    ;;
esac
