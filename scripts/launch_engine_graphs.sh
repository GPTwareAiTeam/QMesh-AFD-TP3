#!/usr/bin/env bash
# Reproducible AFD-TP3 production engine launch: P7 GPU0, FULL decode CUDA
# graphs, real Seam transport over fresh AFD edges. DSpark is off by default;
# SPEC_TOKENS>0 enables exact DSpark with the validated retention overlay.
#
#   bash scripts/launch_engine_graphs.sh
#
# This is the experiment-tree successor of the earlier profile-tree launcher.
# Defaults are the
# verified graph-safe seam recipe (EAGER=0 AFD_TRANSPORT=1 AFD_SEAM=1
# capture=1 SPEC=0 GPU0). The container is never replaced: if NAME already
# exists (running or stopped), this exits nonzero and leaves it alone.
#
# Env knobs (all optional unless noted):
#   SPEC_TOKENS=N          0 (default, non-spec rollback), 1..5, or 7. 1/3/5 are
#                          the validated points; 2/4 need an explicit
#                          CAPTURE_SIZES; 7 is new (verify width 8) and defaults
#                          to 1,7,8. 6 is rejected. >0 requires explicit
#                          GPU_MEM_UTIL and DSPARK_EXTRA_GIB.
#   GPU_MEM_UTIL=0.xx      REQUIRED when SPEC_TOKENS>0. The default 0.90 applies
#                          only to SPEC_TOKENS=0. Checked against the measured
#                          baseline + DSpark drafter + DSPARK_EXTRA_GIB.
#   DSPARK_EXTRA_GIB=G     REQUIRED when SPEC_TOKENS>0: draft KV, verify
#                          workspace and capture headroom, stated explicitly.
#   DSPARK_BASELINE_GIB=G  measured non-spec engine footprint (default 82.16)
#   DSPARK_DRAFTER_GIB=G   DSpark drafter footprint (default 7.388)
#   DSPARK_MEM_ACK=1       required to accept GPU_MEM_UTIL>0.97
#   GPU_TOTAL_MIB=N        override GPU0 memory.total (default: nvidia-smi -i 0)
#   MAX_MODEL_LEN=N        model context/admission cap (default 8192). The
#                          model's native max is 1048576; this is an admission
#                          cap, not a proven 1M-prompt run.
#   MAX_NUM_SEQS=N         max resident sequences (default 1). >1 is explicit
#                          opt-in and derives exact target capture widths for
#                          C=1,2,4,8 (other C falls back to eager).
#   MAX_NUM_BATCHED_TOKENS=N  scheduler prefill token budget (default 4096).
#   KV_CACHE_GIB=G         explicit KV reservation, positive integer GiB.
#   KV_CACHE_BYTES=B       explicit KV reservation, positive integer bytes.
#                          GIB/BYTES are one setting; conflicting values fail.
#                          Passed as --kv-cache-memory-bytes; vLLM then skips
#                          KV profiling and reserves exactly B. The startup
#                          free-memory gate still applies; CUDA-graph memory is
#                          not subtracted by that gate. No implicit KV size.
#   DSPARK_ENGINE_EVENT_FILE=P  opt-in engine event JSONL; must be under
#                          /tmp/afd_dump (the host/container shared mount).
#   CAPTURE_SIZES=a,b,...  decode capture sizes. Default by SPEC: 0 -> 1,
#                          1 -> 1,2, 3 -> 1,3,4, 5 -> 1,5,6, 7 -> 1,7,8.
#                          With MAX_NUM_SEQS>1 the default adds the exact
#                          target verify widths for C=1,2,4,8 (k=5, C=8 ->
#                          1,5,6,12,24,48; no padded target rows). Any explicit
#                          list must contain 1, SPEC, SPEC+1 and those widths.
#   MAX_CAPTURE=N          scalar --max-cudagraph-capture-size; defaults to the
#                          largest CAPTURE_SIZES entry and must equal it (vLLM
#                          rejects a scalar that differs from the filtered list max)
#   EAGER=1                explicit experiment: --enforce-eager, no CUDA graphs
#   AFD_TRANSPORT=0        DIAGNOSTIC ONLY: P7-only floor, no seam, no wire,
#                          edges untouched (cannot serve real requests)
#   AFD_SEAM=0             legacy Python seam instead of the graph-safe Seam
#                          (A/B only; default is the Seam)
#   SKIP_EDGES=1           do not restart edges (they must already be waiting)
#   AFD_SEAM_STATUS_DIR=D  optional existing host directory mounted at /prof;
#                          enables AFD_SEAM_STATUS_LOG=/prof/seam_status.jsonl
#   MHC_QUANT=0            boot the image's own mHC kernel/boundary files instead
#                          of p7/mhc_quant (A/B; default mounts the fused quant)
#   DSPARK_RETENTION_DIR=D package root (default p7/dspark_retention); only
#                          mounted when SPEC_TOKENS>0
#   NATIVE_CTX_MEM=1       mount p7/native_context_memory's exact-bound indexer
#                          and in-place MXFP8 dequant overlays (hash-pinned;
#                          default 0 leaves the image files untouched). The
#                          measured native-1M / 3 GiB KV / C=8 recipe also needs
#                          MAX_NUM_BATCHED_TOKENS=2048 for JIT-warmup headroom.
#   DRY_RUN=1              validate everything and print the env/argv without
#                          restarting edges or starting a container
#   NAME=... PORT=... CACHE_DIR=... EXTRA_DOCKER_ARGS=...
#
# Site configuration: environment wins; defaults may come from the gitignored
# scripts/local.env (copy scripts/local.env.example once per host).
#   EXP=DIR              experiment root (default: parent of this script)
#   AFD_NATIVE_DIR=DIR   native transport source (default: $EXP/native)
#   AFD_BIN_DIR=DIR      built transport artifacts (default: $AFD_NATIVE_DIR/bin);
#                        mounted at /opt/qafd and exported as QMESH_AFD_BIN
#   EDGES=FILE           edge restart script (default: native/scripts, then
#                        $EXP/scripts/restart_afd_edges.sh)
#   VLLM_SRC=DIR         vLLM checkout with docker/sm120/run-dsv41.sh (required)
#   MODEL_DIR=DIR        served model directory (required, absolute)
#   MODEL_MOUNT=DIR      host tree bind-mounted at the same path (default:
#                        /models when MODEL_DIR is under it, else its parent)
#   IMAGE=TAG            engine image (default vllm-moet:dsv41-213d5fa)
#   AFD_CONFIG_FILE=FILE peer JSON (dev/gid/ip/port) mounted as
#                        /opt/afd/afd.json (default: afd.local.json if present,
#                        else the tracked public sample)
#   GX1_HOST=H GX2_HOST=H  passed to the edge restart script
#   DSV41_SKIP_LOCAL_ENV=1  ignore scripts/local.env (hermetic tests)
#
# Edge stderr/stdout goes to /tmp/<NAME>-edges.log (override with EDGE_LOG);
# a nonzero edge restart aborts this launch with that same status. Nothing is
# piped through tail before the status is checked.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP=${EXP:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
EXP=$(cd -- "$EXP" && pwd) || { echo "launch: EXP is not a directory: ${EXP}" >&2; exit 2; }

die() { echo "launch: $*" >&2; exit 2; }

# Site values are defaults only: anything already exported wins.
if [[ "${DSV41_SKIP_LOCAL_ENV:-0}" != 1 && -f "$SCRIPT_DIR/local.env" ]]; then
  # shellcheck source=local.env.example
  . "$SCRIPT_DIR/local.env"
fi

AFD_NATIVE_DIR=${AFD_NATIVE_DIR:-$EXP/native}
AFD_BIN_DIR=${AFD_BIN_DIR:-}
if [[ -z "$AFD_BIN_DIR" ]]; then
  [[ -d "$AFD_NATIVE_DIR" ]] ||
    die "AFD_NATIVE_DIR not found: $AFD_NATIVE_DIR (set AFD_NATIVE_DIR or AFD_BIN_DIR, or create scripts/local.env from local.env.example)"
  AFD_BIN_DIR=$AFD_NATIVE_DIR/bin
fi
AFD_BIN_DIR=$(cd -- "$AFD_BIN_DIR" && pwd) ||
  die "AFD_BIN_DIR not found: ${AFD_BIN_DIR}"

EDGES=${EDGES:-}
if [[ -z "$EDGES" ]]; then
  for _candidate in \
    "$AFD_NATIVE_DIR/scripts/restart_afd_edges.sh" \
    "$EXP/scripts/restart_afd_edges.sh"; do
    if [[ -f "$_candidate" ]]; then
      EDGES=$_candidate
      break
    fi
  done
fi

[[ -n "${VLLM_SRC:-}" ]] ||
  die "VLLM_SRC is not set (copy scripts/local.env.example to scripts/local.env, or export VLLM_SRC)"
VLLM_SRC=$(cd -- "$VLLM_SRC" && pwd) || die "VLLM_SRC is not a directory: ${VLLM_SRC}"
[[ -f "$VLLM_SRC/docker/sm120/run-dsv41.sh" ]] ||
  die "run-dsv41.sh not found under VLLM_SRC=$VLLM_SRC"

[[ -n "${MODEL_DIR:-}" ]] ||
  die "MODEL_DIR is not set (copy scripts/local.env.example to scripts/local.env, or export MODEL_DIR)"
[[ "$MODEL_DIR" == /* ]] || die "MODEL_DIR must be absolute, got '$MODEL_DIR'"
[[ -f "$MODEL_DIR/config.json" ]] || die "no config.json in MODEL_DIR=$MODEL_DIR"
MODEL_MOUNT=${MODEL_MOUNT:-}
if [[ -z "$MODEL_MOUNT" ]]; then
  if [[ "$MODEL_DIR" == /models/* ]]; then
    MODEL_MOUNT=/models
  else
    MODEL_MOUNT=$(dirname -- "$MODEL_DIR")
  fi
fi
[[ -d "$MODEL_MOUNT" ]] || die "MODEL_MOUNT is not a directory: $MODEL_MOUNT"

IMAGE=${IMAGE:-vllm-moet:dsv41-213d5fa}
GX1_HOST=${GX1_HOST:-}
GX2_HOST=${GX2_HOST:-}
AFD_LAYERS=${AFD_LAYERS:-40}
AFD_ROWS=${AFD_ROWS:-4096}
[[ "$AFD_LAYERS" =~ ^[1-9][0-9]*$ ]] || die "AFD_LAYERS must be a positive integer, got '$AFD_LAYERS'"
[[ "$AFD_ROWS" =~ ^[1-9][0-9]*$ ]] || die "AFD_ROWS must be a positive integer, got '$AFD_ROWS'"

AFD_CONFIG_FILE=${AFD_CONFIG_FILE:-}
if [[ -z "$AFD_CONFIG_FILE" ]]; then
  if [[ -f "$EXP/p7/afd_overlay/afd.local.json" ]]; then
    AFD_CONFIG_FILE=$EXP/p7/afd_overlay/afd.local.json
  else
    AFD_CONFIG_FILE=$EXP/p7/afd_overlay/afd.example.json
  fi
fi
_cfg_dir=$(cd -- "$(dirname -- "$AFD_CONFIG_FILE")" 2>/dev/null && pwd) ||
  die "AFD_CONFIG_FILE directory not found: $AFD_CONFIG_FILE"
AFD_CONFIG_FILE=$_cfg_dir/$(basename -- "$AFD_CONFIG_FILE")
[[ -f "$AFD_CONFIG_FILE" ]] ||
  die "AFD_CONFIG_FILE not found: $AFD_CONFIG_FILE (copy p7/afd_overlay/afd.example.json to afd.local.json and edit the peers)"

python3 - "$AFD_CONFIG_FILE" "${DRY_RUN:-0}" "$AFD_LAYERS" "$AFD_ROWS" <<'PY' || exit 2
import json
import sys

path, dry_run, want_layers, want_rows = sys.argv[1:5]
try:
    with open(path) as handle:
        cfg = json.load(handle)
except OSError as exc:
    sys.exit(f"launch: AFD_CONFIG_FILE unreadable: {path}: {exc}")
except json.JSONDecodeError as exc:
    sys.exit(f"launch: AFD_CONFIG_FILE is not valid JSON: {path}: {exc}")

if not isinstance(cfg, dict):
    sys.exit(f"launch: AFD_CONFIG_FILE must be a JSON object: {path}")
for key, want in (("layers", want_layers), ("rows", want_rows)):
    got = cfg.get(key)
    if isinstance(got, bool) or got != int(want):
        sys.exit(
            f"launch: AFD_CONFIG_FILE {key}={got!r} but AFD_{key.upper()}={want}; "
            "the wire geometry must match"
        )
peers = cfg.get("peers")
if not isinstance(peers, list) or len(peers) != 2:
    sys.exit("launch: AFD_CONFIG_FILE peers must be exactly two objects (GX1, GX2)")
for i, peer in enumerate(peers, 1):
    if not isinstance(peer, dict):
        sys.exit(f"launch: AFD_CONFIG_FILE peer {i} is not an object")
    dev, gid, ip, port = peer.get("dev"), peer.get("gid"), peer.get("ip"), peer.get("port")
    if not isinstance(dev, str) or not dev.strip():
        sys.exit(f"launch: AFD_CONFIG_FILE peer {i} needs a non-empty 'dev'")
    if isinstance(gid, bool) or not isinstance(gid, int) or gid < 0:
        sys.exit(f"launch: AFD_CONFIG_FILE peer {i} 'gid' must be a non-negative integer")
    if not isinstance(ip, str) or not ip.strip():
        sys.exit(f"launch: AFD_CONFIG_FILE peer {i} needs a non-empty 'ip'")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        sys.exit(f"launch: AFD_CONFIG_FILE peer {i} 'port' must be 1..65535")
if cfg.get("public_sample") is True:
    message = (
        f"launch: AFD_CONFIG_FILE={path} is the public sample (RFC5737 addresses); "
        "copy p7/afd_overlay/afd.example.json to the gitignored afd.local.json "
        "and set each peer's dev/gid/ip/port, or pass AFD_CONFIG_FILE"
    )
    if dry_run == "1":
        print(message + " (DRY_RUN accepts it for validation only)", file=sys.stderr)
    else:
        sys.exit(message + "; refusing a real launch")
print(
    f"launch: peer config ok: layers={want_layers} rows={want_rows} peers="
    + ", ".join(f"{p['dev']}/{p['ip']}:{p['port']}" for p in peers),
    file=sys.stderr,
)
PY

if [[ "${AFD_TRANSPORT:-1}" == 1 ]]; then
  compgen -G "$AFD_BIN_DIR/_qmesh_afd*.so" >/dev/null ||
    die "AFD_BIN_DIR=$AFD_BIN_DIR has no _qmesh_afd*.so (run native/build.sh ext)"
fi
NAME=${NAME:-dsv41-p7}
PORT=${PORT:-8001}
CACHE_DIR=${CACHE_DIR:-/tmp/cache-dsv41-p7}
EDGE_LOG=${EDGE_LOG:-/tmp/${NAME}-edges.log}

SPEC_TOKENS=${SPEC_TOKENS:-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-}
GPU_TOTAL_MIB=${GPU_TOTAL_MIB:-}
DSPARK_BASELINE_GIB=${DSPARK_BASELINE_GIB:-82.16}
DSPARK_DRAFTER_GIB=${DSPARK_DRAFTER_GIB:-7.388}
DSPARK_EXTRA_GIB=${DSPARK_EXTRA_GIB:-}
DSPARK_RETENTION_DIR=${DSPARK_RETENTION_DIR:-$EXP/p7/dspark_retention}
NATIVE_CTX_MEM=${NATIVE_CTX_MEM:-0}
NATIVE_CTX_MEM_DIR=${NATIVE_CTX_MEM_DIR:-$EXP/p7/native_context_memory}
CAPTURE_SIZES=${CAPTURE_SIZES:-}
MAX_CAPTURE=${MAX_CAPTURE:-}
DRY_RUN=${DRY_RUN:-0}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
KV_CACHE_GIB=${KV_CACHE_GIB:-}
KV_CACHE_BYTES=${KV_CACHE_BYTES:-}
DSPARK_ENGINE_EVENT_FILE=${DSPARK_ENGINE_EVENT_FILE:-}


EAGER=${EAGER:-0}
AFD_TRANSPORT=${AFD_TRANSPORT:-1}
AFD_SEAM=${AFD_SEAM:-1}

case "$EAGER" in 0|1) ;; *) die "EAGER must be 0 or 1, got '$EAGER'";; esac
case "$AFD_TRANSPORT" in
  0|1) ;;
  *) die "AFD_TRANSPORT must be 0 or 1, got '$AFD_TRANSPORT'" ;;
esac
case "$AFD_SEAM" in 0|1) ;; *) die "AFD_SEAM must be 0 or 1, got '$AFD_SEAM'";; esac
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'";; esac
case "$NATIVE_CTX_MEM" in 0|1) ;; *) die "NATIVE_CTX_MEM must be 0 or 1, got '$NATIVE_CTX_MEM'";; esac
case "$SPEC_TOKENS" in
  0|1|2|3|4|5|7) ;;
  *) die "SPEC_TOKENS must be 0..5 or 7 (validated: 0/1/3/5), got '$SPEC_TOKENS'" ;;
esac
if [[ "$SPEC_TOKENS" == 2 || "$SPEC_TOKENS" == 4 ]]; then
  echo "launch: SPEC_TOKENS=$SPEC_TOKENS is measurable but not a validated point; explicit CAPTURE_SIZES is required" >&2
fi
_is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
_is_pos_int "$MAX_MODEL_LEN" ||
  die "MAX_MODEL_LEN must be a positive integer, got '$MAX_MODEL_LEN'"
(( MAX_MODEL_LEN <= 1048576 )) ||
  die "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native max_position_embeddings 1048576 (admission cap; no 1M prompt is implied)"
_is_pos_int "$MAX_NUM_SEQS" ||
  die "MAX_NUM_SEQS must be a positive integer, got '$MAX_NUM_SEQS'"
_is_pos_int "$MAX_NUM_BATCHED_TOKENS" ||
  die "MAX_NUM_BATCHED_TOKENS must be a positive integer, got '$MAX_NUM_BATCHED_TOKENS'"
(( MAX_NUM_BATCHED_TOKENS >= MAX_NUM_SEQS )) ||
  die "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS must be >= MAX_NUM_SEQS=$MAX_NUM_SEQS"
(( MAX_NUM_BATCHED_TOKENS <= MAX_NUM_SEQS * MAX_MODEL_LEN )) ||
  die "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS exceeds MAX_NUM_SEQS*MAX_MODEL_LEN=$((MAX_NUM_SEQS * MAX_MODEL_LEN))"

_kv_from_gib=""
if [[ -n "$KV_CACHE_GIB" ]]; then
  _is_pos_int "$KV_CACHE_GIB" ||
    die "KV_CACHE_GIB must be a positive integer, got '$KV_CACHE_GIB'"
  (( KV_CACHE_GIB <= 1024 )) ||
    die "KV_CACHE_GIB=$KV_CACHE_GIB is implausibly large for one GPU"
  _kv_from_gib=$((KV_CACHE_GIB * 1073741824))
fi
if [[ -n "$KV_CACHE_BYTES" ]]; then
  _is_pos_int "$KV_CACHE_BYTES" ||
    die "KV_CACHE_BYTES must be a positive integer, got '$KV_CACHE_BYTES'"
  if [[ -n "$_kv_from_gib" && "$KV_CACHE_BYTES" != "$_kv_from_gib" ]]; then
    die "KV_CACHE_GIB=$KV_CACHE_GIB ($_kv_from_gib bytes) disagrees with KV_CACHE_BYTES=$KV_CACHE_BYTES"
  fi
  _kv_from_gib=$KV_CACHE_BYTES
fi
KV_CACHE_BYTES=$_kv_from_gib
if [[ -n "$KV_CACHE_BYTES" ]]; then
  _kv_gib=$(awk -v b="$KV_CACHE_BYTES" 'BEGIN { printf "%.3f", b / 1073741824 }')
  echo "launch: explicit KV cache bytes=$KV_CACHE_BYTES (${_kv_gib} GiB); vLLM skips KV profiling and reserves exactly this; the startup free-memory gate still applies and CUDA-graph memory is not subtracted" >&2
fi
if [[ -n "$DSPARK_ENGINE_EVENT_FILE" ]]; then
  [[ "$DSPARK_ENGINE_EVENT_FILE" == /tmp/afd_dump/* ]] ||
    die "DSPARK_ENGINE_EVENT_FILE must be under /tmp/afd_dump (host/container shared mount), got '$DSPARK_ENGINE_EVENT_FILE'"
  echo "launch: opt-in engine event file $DSPARK_ENGINE_EVENT_FILE" >&2
fi

if [[ "$EAGER" == 1 ]]; then
  echo "launch: EAGER=1 explicit experiment (no CUDA graphs)" >&2
  CG_ARGS="--enforce-eager"
  MAX_CAPTURE=0
else
  # CAPTURE_SIZES is a vLLM list; --max-cudagraph-capture-size is one scalar.
  # Feeding the list text to the scalar option is the bug this guards against.
  # DSpark needs one graph at the draft width (SPEC_TOKENS rows) and one at the
  # target verify width (SPEC_TOKENS+1 rows); every decode graph needs row 1.
  if [[ -z "$CAPTURE_SIZES" ]]; then
    if (( MAX_NUM_SEQS > 1 )); then
      # Exact target verify widths for the measured concurrency points 1,2,4,8.
      # Target graphs round a smaller width up to decode_query_len=SPEC+1, so
      # these widths are the ones that avoid padded sparse-MLA target rows.
      _want=(1)
      if (( SPEC_TOKENS > 0 )); then
        _want+=("$SPEC_TOKENS" "$((SPEC_TOKENS + 1))")
      fi
      _c=1
      while (( _c <= MAX_NUM_SEQS )); do
        if (( SPEC_TOKENS > 0 )); then
          _want+=("$((_c * (SPEC_TOKENS + 1)))")
        else
          _want+=("$_c")
        fi
        (( _c *= 2 ))
      done
      CAPTURE_SIZES=$(printf '%s\n' "${_want[@]}" | sort -n -u | paste -sd, -)
      echo "launch: derived exact capture sizes [$CAPTURE_SIZES] for MAX_NUM_SEQS=$MAX_NUM_SEQS (target widths for C=1,2,4,8; other C falls back to eager)" >&2
    else
      case "$SPEC_TOKENS" in
        0) CAPTURE_SIZES=1 ;;
        1) CAPTURE_SIZES=1,2 ;;
        3) CAPTURE_SIZES=1,3,4 ;;
        5) CAPTURE_SIZES=1,5,6 ;;
        7) CAPTURE_SIZES=1,7,8 ;;
        *) die "SPEC_TOKENS=$SPEC_TOKENS has no validated capture default; set CAPTURE_SIZES explicitly (must contain 1, $SPEC_TOKENS and $((SPEC_TOKENS+1)))" ;;
      esac
    fi
  fi
  IFS=',' read -r -a _raw_sizes <<< "$CAPTURE_SIZES"
  _sizes=()
  _largest=0
  _normalized=""
  for _s in "${_raw_sizes[@]}"; do
    _s=${_s//[[:space:]]/}
    [[ "$_s" =~ ^[1-9][0-9]*$ ]] ||
      die "CAPTURE_SIZES must be comma-separated positive integers, got '$CAPTURE_SIZES'"
    if (( _s > _largest )); then _largest=$_s; fi
    _sizes+=("$_s")
    _normalized+="${_normalized:+,}$_s"
  done
  (( _largest > 0 )) || die "CAPTURE_SIZES is empty"
  _has_size() {
    local want=$1 have
    for have in "${_sizes[@]}"; do [[ "$have" == "$want" ]] && return 0; done
    return 1
  }
  _has_size 1 || die "CAPTURE_SIZES must include 1 (the single-row decode graph), got '$CAPTURE_SIZES'"
  if [[ "$SPEC_TOKENS" != 0 ]]; then
    _has_size "$SPEC_TOKENS" ||
      die "CAPTURE_SIZES must include the draft width $SPEC_TOKENS, got '$CAPTURE_SIZES'"
    _has_size "$((SPEC_TOKENS + 1))" ||
      die "CAPTURE_SIZES must include the verify width $((SPEC_TOKENS + 1)), got '$CAPTURE_SIZES'"
  fi
  _c=1
  while (( _c <= MAX_NUM_SEQS )); do
    if [[ "$SPEC_TOKENS" == 0 ]]; then
      _width=$_c
    else
      _width=$((_c * (SPEC_TOKENS + 1)))
    fi
    _has_size "$_width" ||
      die "CAPTURE_SIZES must include the exact target width $_width for concurrency $_c (MAX_NUM_SEQS=$MAX_NUM_SEQS), got '$CAPTURE_SIZES'"
    (( _c *= 2 ))
  done
  (( MAX_NUM_BATCHED_TOKENS >= _largest )) ||
    die "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS is smaller than the largest emitted capture size $_largest; vLLM filters captures above max_num_batched_tokens and then rejects a max_cudagraph_capture_size that differs from the filtered list max"
  if [[ -z "$MAX_CAPTURE" ]]; then
    MAX_CAPTURE=$_largest
  else
    [[ "$MAX_CAPTURE" =~ ^[1-9][0-9]*$ ]] ||
      die "MAX_CAPTURE must be a positive integer, got '$MAX_CAPTURE'"
    (( MAX_CAPTURE == _largest )) ||
      die "MAX_CAPTURE=$MAX_CAPTURE must equal the largest capture size ($_largest); vLLM rejects a scalar that differs from the filtered list max"
  fi
  CG_ARGS="--compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${_normalized}]} --max-cudagraph-capture-size ${MAX_CAPTURE}"
fi

# Memory budget. SPEC_TOKENS=0 keeps the verified 0.90 default. DSpark needs a
# stated budget: a drafter, its KV/verify workspace and the extra graph captures
# sit on top of the measured non-spec footprint, so the old 0.90 default leaves
# ~0.36 GiB and OOMs. There is deliberately no implicit 0.98.
_is_gib() { [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; }
if [[ "$SPEC_TOKENS" == 0 ]]; then
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
else
  [[ -n "$GPU_MEM_UTIL" ]] ||
    die "SPEC_TOKENS=$SPEC_TOKENS requires an explicit GPU_MEM_UTIL (0.90 is the non-spec default and has no DSpark headroom)"
  [[ "$GPU_MEM_UTIL" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "GPU_MEM_UTIL must be a decimal fraction, got '$GPU_MEM_UTIL'"
  awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u < 1) }' ||
    die "GPU_MEM_UTIL must be in (0,1), got '$GPU_MEM_UTIL'"
  [[ -n "$DSPARK_EXTRA_GIB" ]] ||
    die "SPEC_TOKENS=$SPEC_TOKENS requires explicit DSPARK_EXTRA_GIB (draft KV + verify workspace + capture headroom)"
  _is_gib "$DSPARK_BASELINE_GIB" || die "DSPARK_BASELINE_GIB must be a number, got '$DSPARK_BASELINE_GIB'"
  _is_gib "$DSPARK_DRAFTER_GIB" || die "DSPARK_DRAFTER_GIB must be a number, got '$DSPARK_DRAFTER_GIB'"
  _is_gib "$DSPARK_EXTRA_GIB" || die "DSPARK_EXTRA_GIB must be a number, got '$DSPARK_EXTRA_GIB'"
  if [[ -z "$GPU_TOTAL_MIB" ]]; then
    GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d '[:space:]') || true
    [[ "$GPU_TOTAL_MIB" =~ ^[1-9][0-9]*$ ]] ||
      die "cannot query GPU0 memory.total; set GPU_TOTAL_MIB explicitly"
  fi
  [[ "$GPU_TOTAL_MIB" =~ ^[1-9][0-9]*$ ]] ||
    die "GPU_TOTAL_MIB must be a positive integer, got '$GPU_TOTAL_MIB'"
  awk -v u="$GPU_MEM_UTIL" -v t="$GPU_TOTAL_MIB" \
      -v b="$DSPARK_BASELINE_GIB" -v d="$DSPARK_DRAFTER_GIB" \
      -v e="$DSPARK_EXTRA_GIB" 'BEGIN {
        budget = t / 1024.0 * u
        need = b + d + e
        printf "launch: DSpark memory: budget %.3f GiB (util %s x %s MiB) vs need %.3f GiB (baseline %s + drafter %s + extra %s), headroom %.3f GiB\n",
               budget, u, t, need, b, d, e, budget - need > "/dev/stderr"
        exit !(budget + 1e-9 >= need)
      }' ||
    die "GPU_MEM_UTIL=$GPU_MEM_UTIL does not cover baseline $DSPARK_BASELINE_GIB + drafter $DSPARK_DRAFTER_GIB + extra $DSPARK_EXTRA_GIB GiB"
  if [[ -n "$KV_CACHE_BYTES" ]]; then
    # The explicit reservation replaces vLLM's profiled KV, so this is a
    # conservative extra-headroom read, not a second hard gate: the measured
    # baseline may already include a smaller profiled KV. Never trade the
    # requested KV/context down here; the allocator and startup gate decide.
    awk -v u="$GPU_MEM_UTIL" -v t="$GPU_TOTAL_MIB" \
        -v b="$DSPARK_BASELINE_GIB" -v d="$DSPARK_DRAFTER_GIB" \
        -v e="$DSPARK_EXTRA_GIB" -v kv="$KV_CACHE_BYTES" 'BEGIN {
          budget = t / 1024.0 * u
          kv_gib = kv / 1073741824.0
          total = b + d + e + kv_gib
          printf "launch: explicit KV %.3f GiB: estimate %.3f GiB vs budget %.3f GiB, headroom %.3f GiB\n",
                 kv_gib, total, budget, budget - total > "/dev/stderr"
          if (total > budget + 1e-9)
            printf "launch: WARNING explicit KV estimate exceeds the util budget (conservative; baseline may include profiled KV); the launcher does not lower KV or cap context, actual allocator/startup failure is authoritative\n" > "/dev/stderr"
        }'
  fi
  if awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0.97) }'; then
    [[ "${DSPARK_MEM_ACK:-0}" == 1 ]] ||
      die "GPU_MEM_UTIL=$GPU_MEM_UTIL (>0.97) needs DSPARK_MEM_ACK=1 after a measured headroom run"
    echo "launch: WARNING GPU_MEM_UTIL=$GPU_MEM_UTIL accepted with DSPARK_MEM_ACK=1; watch for allocator retries" >&2
  fi
fi

if [[ "$AFD_TRANSPORT" == 0 ]]; then
  AFD_SEAM=0
  echo "launch: DIAGNOSTIC MODE: AFD_TRANSPORT=0 (P7-only floor, no Seam, no wire, edges untouched)" >&2
elif [[ "$AFD_SEAM" == 1 ]]; then
  echo "launch: transport=real edges + graph-safe Seam" >&2
else
  echo "launch: transport=real edges + legacy Python seam (AFD_SEAM=0)" >&2
fi


# DSpark is only exact if the served pack really carries the 3-stage draft
# (mtp.0/1/2), the trained block, and the 128-expert top-3 draft MoEs. Fail
# closed before spending a GPU window on a mispacked model.
if [[ "$SPEC_TOKENS" != 0 ]]; then
  if ! python3 - "$MODEL_DIR" "$SPEC_TOKENS" <<'PY'
import json
import os
import sys

model_dir, k = sys.argv[1], int(sys.argv[2])
with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
text = cfg.get("text_config") or cfg


def get(key):
    return text.get(key, cfg.get(key))


expect = {
    "num_hidden_layers": 40,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "n_routed_experts": 384,
}
for key, want in expect.items():
    got = get(key)
    if got != want:
        sys.exit(f"model config {key}={got!r}, expected {want!r}")
# dspark_block_size is the checkpoint's default drafting width, not an
# architectural cap: vllm's DSpark config only uses it to default
# num_speculative_tokens when that is omitted, and every draft buffer, mask,
# and position is sized from the requested width. The launcher allowlist
# bounds SPEC_TOKENS to 7 (verify width 8); the learned weights are unchanged.

index_path = os.path.join(model_dir, "model.safetensors.index.json")
with open(index_path) as f:
    weight_map = json.load(f)["weight_map"]
mtp = [name for name in weight_map if name.startswith("mtp.")]
stages = sorted({name.split(".")[1] for name in mtp})
files = sorted({weight_map[name] for name in mtp})
if len(mtp) != 2401:
    sys.exit(f"draft weights incomplete: {len(mtp)} mtp.* entries, expected 2401")
if stages != ["0", "1", "2"]:
    sys.exit(f"draft stages {stages}, expected ['0','1','2']")
missing = [f for f in files if not os.path.isfile(os.path.join(model_dir, f))]
if missing:
    sys.exit(f"draft shards missing: {missing}")
print(
    f"launch: DSpark pack ok: k={k} block=5 mtp=2401 stages={stages} "
    f"files={files} draft_experts=128 top3 targets=[37,38,39]",
    file=sys.stderr,
)
PY
  then
    die "MODEL_DIR=$MODEL_DIR is not the exact DSpark-1017 pack (see the check above)"
  fi
fi

# Refuse to touch an existing container. run-dsv41.sh removes a stopped
# same-name container itself, so this check has to happen before it runs.
if docker container inspect "$NAME" >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "launch: DRY_RUN=1 note: container '$NAME' exists; a real launch would refuse it" >&2
  else
    {
      echo "launch: container '$NAME' already exists; refusing to stop or remove it."
      echo "        If this container is yours to replace, do it explicitly first:"
      echo "          docker stop -t 60 '$NAME' && docker rm '$NAME'"
      echo "        or choose another name with NAME=..."
    } >&2
    exit 1
  fi
fi

DOCKER_EXTRA=(
  -v "$MODEL_MOUNT:$MODEL_MOUNT:ro"
  -v "$EXP/p7/afd_overlay:/opt/afd:ro"
  -v "$AFD_CONFIG_FILE:/opt/afd/afd.json:ro"
  -v "$AFD_BIN_DIR:/opt/qafd:ro"
  -v /tmp/afd_dump:/tmp/afd_dump
  -e PYTHONPATH=/opt/afd
  -e AFD_ENABLE=1
  -e AFD_ROUTED_INTERMEDIATE=576
  -e AFD_TRANSPORT="$AFD_TRANSPORT"
  -e AFD_SEAM="$AFD_SEAM"
  -e AFD_ROWS="$AFD_ROWS"
  -e AFD_LAYERS="$AFD_LAYERS"
  -e AFD_CONFIG=/opt/afd/afd.json
  -e QMESH_AFD_BIN=/opt/qafd
  -e AFD_TIMING=1
  -e AFD_A2_SCALE_FIX=1
  --device /dev/infiniband
)

# Opt-in only: no default event file, so the hook stays off in prod. The host
# directory backing it is already bind-mounted at /tmp/afd_dump above.
if [[ -n "$DSPARK_ENGINE_EVENT_FILE" ]]; then
  DOCKER_EXTRA+=(-e DSPARK_ENGINE_EVENT_FILE="$DSPARK_ENGINE_EVENT_FILE")
fi

# sm_120 mHC fused MXFP8 quant: the attention boundary's mHC kernel writes the
# projection's MXFP8 payload/scales (p7/mhc_quant/). The kernel dir and the three
# files generated by p7/mhc_quant/patch_vllm_mhc_quant_sm120.py are mounted over
# the image's copies, so this launch is the served recipe, not an EXTRA arg.
# MHC_QUANT=0 boots the image's own files for an A/B.
MHC_QUANT_DIR=${MHC_QUANT_DIR:-$EXP/p7/mhc_quant}
if [[ "${MHC_QUANT:-1}" == 1 ]]; then
  _vllm=/usr/local/lib/python3.12/dist-packages/vllm
  for _f in \
    "$MHC_QUANT_DIR/kernel/mhc_post_norm_sm120.cu" \
    "$MHC_QUANT_DIR/kernel/mhc_post_norm_sm120.py" \
    "$MHC_QUANT_DIR/overlay/models/deepseek_v41/nvidia/ops/mhc.py" \
    "$MHC_QUANT_DIR/overlay/models/deepseek_v41/nvidia/model.py" \
    "$MHC_QUANT_DIR/overlay/models/deepseek_v41/attention.py"; do
    [[ -f "$_f" ]] ||
      die "mHC fused quant file missing: $_f (regenerate with p7/mhc_quant/patch_vllm_mhc_quant_sm120.py)"
  done
  DOCKER_EXTRA+=(
    -v "$MHC_QUANT_DIR/kernel:/opt/vllm-moet/dsv41_sm120/mhc_overlap:ro"
    -v "$MHC_QUANT_DIR/overlay/models/deepseek_v41/nvidia/ops/mhc.py:$_vllm/models/deepseek_v41/nvidia/ops/mhc.py:ro"
    -v "$MHC_QUANT_DIR/overlay/models/deepseek_v41/nvidia/model.py:$_vllm/models/deepseek_v41/nvidia/model.py:ro"
    -v "$MHC_QUANT_DIR/overlay/models/deepseek_v41/attention.py:$_vllm/models/deepseek_v41/attention.py:ro"
  )
  echo "launch: mHC fused MXFP8 quant overlay on (MHC_QUANT_DIR=$MHC_QUANT_DIR)" >&2
fi

# DSpark retention overlay. The two single-file overrides bound the target's
# retained suffix at num_speculative_tokens+1 and null the draft SWA block 0
# guard. They are packaged from this exact image by
# p7/dspark_retention/package.sh; never mount a /tmp copy. Non-spec boots mount
# nothing here, so the default rollback is untouched.
DSPARK_IMAGE_ID=sha256:9a21de6e8ba7ade4df4590841401edd09c406638ad21bb22d6f170c297d0f404
DSPARK_BUILD_COMMIT=49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
if [[ "$SPEC_TOKENS" != 0 ]]; then
  _vllm=/usr/local/lib/python3.12/dist-packages/vllm
  [[ -f "$DSPARK_RETENTION_DIR/package.sh" ]] ||
    die "retention package missing at $DSPARK_RETENTION_DIR (run bash $DSPARK_RETENTION_DIR/package.sh or set DSPARK_RETENTION_DIR)"
  _ret_rel=(
    vllm/v1/core/kv_cache_planning.py
    vllm/v1/attention/backends/mla/sparse_swa.py
  )
  _ret_src_sha=(
    dd03a0eb2a75eebdc6b912f0cffa50bb6d3e06f0810c4d964bc9de30dc2a3360
    66b246e8d2d1ea5fbd192da1c9250c2e47bc9824c774cb5e22bf093277bc1fe4
  )
  _ret_overlay_sha=(
    103daabfd0df9e9a8e52938a781e119c3da6cafce08308190d5672a0631db06a
    363268c27d32d652bca0c3b17098574cdaf029255e81dd17c8cc465e723b16f2
  )
  for _i in "${!_ret_rel[@]}"; do
    _rel=${_ret_rel[$_i]}
    for _kind in src overlay; do
      _file="$DSPARK_RETENTION_DIR/$_kind/$_rel"
      [[ -f "$_file" ]] ||
        die "retention $_kind file missing: $_file (regenerate with p7/dspark_retention/package.sh)"
    done
    _got=$(sha256sum "$DSPARK_RETENTION_DIR/src/$_rel")
    _got=${_got%% *}
    [[ "$_got" == "${_ret_src_sha[$_i]}" ]] ||
      die "retention source hash mismatch for $_rel: $_got != ${_ret_src_sha[$_i]}"
    _got=$(sha256sum "$DSPARK_RETENTION_DIR/overlay/$_rel")
    _got=${_got%% *}
    [[ "$_got" == "${_ret_overlay_sha[$_i]}" ]] ||
      die "retention overlay hash mismatch for $_rel: $_got != ${_ret_overlay_sha[$_i]}"
  done
  _image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) ||
    die "IMAGE=$IMAGE not found; the retention overlay is pinned to $DSPARK_IMAGE_ID"
  [[ "$_image_id" == "$DSPARK_IMAGE_ID" ]] ||
    die "retention overlay is pinned to $DSPARK_IMAGE_ID ($DSPARK_BUILD_COMMIT), got $_image_id"
  _commit=$(docker image inspect --format '{{index .Config.Labels "ai.vllm.build.commit"}}' "$IMAGE" 2>/dev/null) ||
    _commit=""
  [[ "$_commit" == "$DSPARK_BUILD_COMMIT" ]] ||
    die "image build commit mismatch: '$_commit' != '$DSPARK_BUILD_COMMIT'"
  if ! python3 - "$DSPARK_RETENTION_DIR/overlay/vllm/v1/core/kv_cache_planning.py" \
                  "$DSPARK_RETENTION_DIR/overlay/vllm/v1/attention/backends/mla/sparse_swa.py" <<'PY'
import sys

for path in sys.argv[1:]:
    with open(path) as f:
        compile(f.read(), path, "exec")
PY
  then
    die "retention overlay does not compile"
  fi
  DOCKER_EXTRA+=(
    -v "$DSPARK_RETENTION_DIR/overlay/vllm/v1/core/kv_cache_planning.py:$_vllm/v1/core/kv_cache_planning.py:ro"
    -v "$DSPARK_RETENTION_DIR/overlay/vllm/v1/attention/backends/mla/sparse_swa.py:$_vllm/v1/attention/backends/mla/sparse_swa.py:ro"
    -e AFD_NUM_EXPERTS=384
  )
  echo "launch: DSpark retention overlay on (k=$SPEC_TOKENS, extra_retained_tokens=$((SPEC_TOKENS + 1)), package=$DSPARK_RETENTION_DIR)" >&2
else
  echo "launch: DSpark off (SPEC_TOKENS=0); retention overlay not mounted" >&2
fi

# Native 1M context-memory overlay (opt-in, default off). Two single-file
# overrides: the sparse indexer's prefill K workspace uses the exact per-chunk
# bound (factor * max_model_len rows) instead of the 8.7x flashmla_sparse
# envelope (saving 0.93 GiB at max_model_len=1M with the MXFP4 indexer cache),
# and MXFP8 dequant reuses its float32 weight copy instead of allocating a
# second 128 MiB temporary for wo_a. Both are packaged from this exact image by
# p7/native_context_memory/package.sh; never mount a /tmp copy. With the flag
# off, nothing is mounted and the image files are untouched.
NATIVE_CTX_IMAGE_ID=sha256:9a21de6e8ba7ade4df4590841401edd09c406638ad21bb22d6f170c297d0f404
NATIVE_CTX_BUILD_COMMIT=49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
if [[ "$NATIVE_CTX_MEM" == 1 ]]; then
  # Container root of the vLLM package. Every `_ctx_rel` already starts with
  # `vllm/`, so this must NOT include `vllm` itself: the engine imports
  # `/usr/local/lib/python3.12/dist-packages/vllm/...`, and a doubled
  # `vllm/vllm` path mounts a file nothing imports (silent no-op).
  _vllm=/usr/local/lib/python3.12/dist-packages
  _ctx_rels=(
    vllm/v1/attention/backends/mla/indexer.py
    vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
  )
  _ctx_src_shas=(
    3150214befd174998b02cd2fc254a9b03966b20f4da2372af0c5fc5a956d61fe
    03a16184c905b3a5f0f4aa52736cefeb0528d97fa3d79a85da08d10b68b86c09
  )
  _ctx_overlay_shas=(
    6c4185759db3172a6cd1dc625b8e7cca083d49ccbd9ea07991f14cfda8305ed6
    d70568eeb878bfc0b2f3ce1b49ce4f76a5520711d2cf54218adbd68481c744e9
  )
  _ctx_overlays=()
  [[ -f "$NATIVE_CTX_MEM_DIR/package.sh" ]] ||
    die "native-context-memory package missing at $NATIVE_CTX_MEM_DIR (run bash $NATIVE_CTX_MEM_DIR/package.sh or set NATIVE_CTX_MEM_DIR)"
  _image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) ||
    die "IMAGE=$IMAGE not found; the native-context-memory overlay is pinned to $NATIVE_CTX_IMAGE_ID"
  [[ "$_image_id" == "$NATIVE_CTX_IMAGE_ID" ]] ||
    die "native-context-memory overlay is pinned to $NATIVE_CTX_IMAGE_ID ($NATIVE_CTX_BUILD_COMMIT), got $_image_id"
  _commit=$(docker image inspect --format '{{index .Config.Labels "ai.vllm.build.commit"}}' "$IMAGE" 2>/dev/null) ||
    _commit=""
  [[ "$_commit" == "$NATIVE_CTX_BUILD_COMMIT" ]] ||
    die "image build commit mismatch: '$_commit' != '$NATIVE_CTX_BUILD_COMMIT'"
  for _i in "${!_ctx_rels[@]}"; do
    _ctx_rel=${_ctx_rels[$_i]}
    _ctx_src_sha=${_ctx_src_shas[$_i]}
    _ctx_overlay_sha=${_ctx_overlay_shas[$_i]}
    _ctx_dest="$_vllm/$_ctx_rel"
    for _kind in src overlay; do
      _file="$NATIVE_CTX_MEM_DIR/$_kind/$_ctx_rel"
      [[ -f "$_file" ]] ||
        die "native-context-memory $_kind file missing: $_file (regenerate with p7/native_context_memory/package.sh)"
    done
    _got=$(sha256sum "$NATIVE_CTX_MEM_DIR/src/$_ctx_rel")
    _got=${_got%% *}
    [[ "$_got" == "$_ctx_src_sha" ]] ||
      die "native-context-memory source hash mismatch for $_ctx_rel: $_got != $_ctx_src_sha"
    _got=$(sha256sum "$NATIVE_CTX_MEM_DIR/overlay/$_ctx_rel")
    _got=${_got%% *}
    [[ "$_got" == "$_ctx_overlay_sha" ]] ||
      die "native-context-memory overlay hash mismatch for $_ctx_rel: $_got != $_ctx_overlay_sha"
    # Preflight the exact path that will be shadowed, inside the image, with no
    # GPU and no engine start: `cat` (CPU-only, ephemeral) must return the pinned
    # source bytes. This proves each mount destination is a real, imported file
    # before the warmup/JIT phase can depend on it, and it fails loudly instead
    # of mounting into a path that does not exist in this image.
    _got=$(docker run --rm --runtime runc --network none --entrypoint cat \
      "$IMAGE" "$_ctx_dest" 2>/dev/null | sha256sum) || true
    _got=${_got%% *}
    [[ "$_got" == "$_ctx_src_sha" ]] ||
      die "image file $_ctx_dest is missing or not the pinned image source ($_got != $_ctx_src_sha); the native-context-memory overlay would not be imported"
    DOCKER_EXTRA+=(-v "$NATIVE_CTX_MEM_DIR/overlay/$_ctx_rel:$_ctx_dest:ro")
    _ctx_overlays+=("$NATIVE_CTX_MEM_DIR/overlay/$_ctx_rel")
    echo "launch: native-context-memory mount $NATIVE_CTX_MEM_DIR/overlay/$_ctx_rel -> $_ctx_dest" >&2
  done
  # Each image invalidates its .pyc by source mtime, and Python then recompiles
  # the single-file mounts at import; compile both up front.
  if ! python3 - "${_ctx_overlays[@]}" <<'PY'
import sys

for path in sys.argv[1:]:
    with open(path) as f:
        compile(f.read(), path, "exec")
PY
  then
    die "native-context-memory overlay does not compile"
  fi
  # Echo the resulting rows so the boot log proves which bound was used (factor
  # mirrors vllm.config.vllm: 5 at <=256Ki tokens, else 2).
  _ctx_factor=2
  (( MAX_MODEL_LEN <= 262144 )) && _ctx_factor=5
  _ctx_rows=$((_ctx_factor * MAX_MODEL_LEN))
  _ctx_k_mib=$((_ctx_rows * 68 / 1048576))
  echo "launch: native-context-memory overlay on (factor=$_ctx_factor, max_prefill_rows=$_ctx_rows, indexer K <= ${_ctx_k_mib} MiB at ratio 1, package=$NATIVE_CTX_MEM_DIR)" >&2
  if (( MAX_MODEL_LEN >= 524288 && MAX_NUM_BATCHED_TOKENS > 2048 )); then
    echo "launch: native-context-memory note: the measured native-1M / 3 GiB KV / C=8 recipe uses MAX_NUM_BATCHED_TOKENS=2048; $MAX_NUM_BATCHED_TOKENS leaves less JIT-warmup headroom" >&2
  fi
else
  echo "launch: native-context-memory overlay off (NATIVE_CTX_MEM=0)" >&2
fi

# Status logging is opt-in: the seam already understands AFD_SEAM_STATUS_LOG;
# only the host directory that backs it is a launch concern. Do not create or
# mount anything when the caller has not supplied an existing directory.
if [[ -n "${AFD_SEAM_STATUS_DIR:-}" ]]; then
  [[ -d "$AFD_SEAM_STATUS_DIR" ]] ||
    die "AFD_SEAM_STATUS_DIR is not an existing directory: $AFD_SEAM_STATUS_DIR"
  DOCKER_EXTRA+=(-v "$AFD_SEAM_STATUS_DIR:/prof")
  DOCKER_EXTRA+=(-e AFD_SEAM_STATUS_LOG=/prof/seam_status.jsonl)
fi

RUN_ENV=(
  MODEL_DIR="$MODEL_DIR" IMAGE="$IMAGE" NAME="$NAME" GPUS=0 TP=1 PORT="$PORT"
  SPEC_TOKENS="$SPEC_TOKENS" LANGUAGE_ONLY=1 GPU_MEM_UTIL="$GPU_MEM_UTIL"
  MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS="$MAX_NUM_SEQS" MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
  CACHE_DIR="$CACHE_DIR" EXTRA_ARGS="$CG_ARGS --no-enable-prefix-caching${KV_CACHE_BYTES:+ --kv-cache-memory-bytes $KV_CACHE_BYTES}"
  EXTRA_DOCKER_ARGS="${DOCKER_EXTRA[*]} ${EXTRA_DOCKER_ARGS:-}"
)
if [[ "$SPEC_TOKENS" != 0 ]]; then
  # Pin the served KV recipe instead of inheriting image defaults, and keep the
  # KV tier on the GPU: the draft needs its blocks resident across verify rows.
  RUN_ENV+=(KV_RECORD=nvfp4 INDEXER_KV_DTYPE=mxfp4 KV_MODE=auto KV_OFFLOAD_GIB=0)
fi

echo "launch: NAME=$NAME PORT=$PORT eager=$EAGER transport=$AFD_TRANSPORT seam=$AFD_SEAM capture=[$CAPTURE_SIZES] max_capture=$MAX_CAPTURE spec=$SPEC_TOKENS gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS kv_cache_bytes=${KV_CACHE_BYTES:-unset}" >&2

if [[ "$DRY_RUN" == 1 ]]; then
  echo "launch: DRY_RUN=1 (nothing restarted or launched)" >&2
  for _e in "${RUN_ENV[@]}"; do printf 'ENV %q\n' "$_e"; done
  for _d in "${DOCKER_EXTRA[@]}"; do printf 'DOCKER_EXTRA %q\n' "$_d"; done
  exit 0
fi

if [[ "$AFD_TRANSPORT" == 1 && "${SKIP_EDGES:-0}" != 1 ]]; then
  [[ -n "$EDGES" && -f "$EDGES" ]] ||
    die "edge restart script not found: set EDGES=FILE (default: \$AFD_NATIVE_DIR/scripts/restart_afd_edges.sh or \$EXP/scripts/restart_afd_edges.sh)"
  _edge_env=(AFD_CONFIG_FILE="$AFD_CONFIG_FILE" AFD_NATIVE_DIR="$AFD_NATIVE_DIR"
             AFD_BIN_DIR="$AFD_BIN_DIR" AFD_LAYERS="$AFD_LAYERS" AFD_ROWS="$AFD_ROWS")
  if [[ -n "$GX1_HOST" ]]; then _edge_env+=(GX1_HOST="$GX1_HOST"); fi
  if [[ -n "$GX2_HOST" ]]; then _edge_env+=(GX2_HOST="$GX2_HOST"); fi
  _edge_rc=0
  env "${_edge_env[@]}" bash "$EDGES" >"$EDGE_LOG" 2>&1 || _edge_rc=$?
elif [[ "$AFD_TRANSPORT" == 0 ]]; then
  echo "launch: diagnostic run, edges not restarted" >&2
else
  echo "launch: SKIP_EDGES=1, assuming both edges are already waiting" >&2
fi

cd "$VLLM_SRC"
# Canonical verified recipe. GPUS=0 and TP=1 are fixed: GPU1/GPU2 are never
# used, and there is no fake or remote-engine default.
exec env "${RUN_ENV[@]}" bash docker/sm120/run-dsv41.sh
