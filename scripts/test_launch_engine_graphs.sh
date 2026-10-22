#!/usr/bin/env bash
# CPU proof of launcher argument/validation behavior and the edge helper's
# DRY_RUN network guard. Every case is offline: no edges, ssh/scp, container,
# or GPU.
# GPU_TOTAL_MIB is pinned so the memory arithmetic is host-independent.
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
LAUNCH=$ROOT/scripts/launch_engine_graphs.sh
PKG=$ROOT/p7/dspark_retention
UNSETS=(
  -u SPEC_TOKENS -u GPU_MEM_UTIL -u GPU_TOTAL_MIB -u DSPARK_BASELINE_GIB
  -u DSPARK_DRAFTER_GIB -u DSPARK_EXTRA_GIB -u DSPARK_RETENTION_DIR
  -u CAPTURE_SIZES -u MAX_CAPTURE -u DSPARK_MEM_ACK -u EAGER
  -u AFD_TRANSPORT -u AFD_SEAM -u DRY_RUN -u NAME -u PORT -u CACHE_DIR
  -u EXP -u AFD_NATIVE_DIR -u AFD_BIN_DIR -u EDGES -u VLLM_SRC
  -u MODEL_DIR -u MODEL_MOUNT -u IMAGE -u AFD_CONFIG_FILE
  -u AFD_LAYERS -u AFD_ROWS -u GX1_HOST -u GX2_HOST -u DSV41_SKIP_LOCAL_ENV
  -u MAX_MODEL_LEN -u MAX_NUM_SEQS -u MAX_NUM_BATCHED_TOKENS
  -u KV_CACHE_GIB -u KV_CACHE_BYTES -u DSPARK_ENGINE_EVENT_FILE
  -u NATIVE_CTX_MEM -u NATIVE_CTX_MEM_DIR
)
TMPDIRS=()
cleanup() { for d in "${TMPDIRS[@]:-}"; do [[ -n "$d" ]] && rm -rf "$d"; done; }
trap cleanup EXIT
# Hermetic site fixture: no dependency on this workstation's checkout, model,
# transport build, or real peer addresses.
FIX=$(mktemp -d)
TMPDIRS+=("$FIX")
mkdir -p "$FIX/vllm/docker/sm120" "$FIX/model" "$FIX/bin" "$FIX/native"
: > "$FIX/vllm/docker/sm120/run-dsv41.sh"
: > "$FIX/bin/_qmesh_afd.cpython-312-x86_64-linux-gnu.so"
cat > "$FIX/bin/docker" <<'SH'
#!/usr/bin/env bash
# Dry-run fixture: the pinned image exists, no container does.
case "$1 $2" in
  "image inspect")
    case "$4" in
      *Labels*) echo 49cb2bbf71cfd36ee5aa33d1e184b1cddff55294 ;;
      *\.Id*) echo sha256:9a21de6e8ba7ade4df4590841401edd09c406638ad21bb22d6f170c297d0f404 ;;
      *) exit 1 ;;
    esac
    ;;
  "run --rm")
    # `docker run --rm ... --entrypoint cat IMAGE PATH` in the launcher: serve
    # the fixture image tree so only a real container path resolves.
    for _arg in "$@"; do _path=$_arg; done
    exec cat "${STUB_IMAGE:?}$_path"
    ;;
  *) exit 1 ;;
esac
SH
chmod +x "$FIX/bin/docker"
python3 - "$FIX/model" <<'PY'
import json
import os
import sys

model = sys.argv[1]
text = {
    "num_hidden_layers": 40,
    "num_nextn_predict_layers": 3,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "n_routed_experts": 384,
}
with open(os.path.join(model, "config.json"), "w") as handle:
    json.dump({"text_config": text}, handle)
shards = [f"shard{s}.safetensors" for s in range(3)]
weight_map = {}
for i in range(2401):
    stage = i % 3
    weight_map[f"mtp.{stage}.w{i}"] = shards[stage]
with open(os.path.join(model, "model.safetensors.index.json"), "w") as handle:
    json.dump({"weight_map": weight_map}, handle)
for shard in shards:
    open(os.path.join(model, shard), "w").close()
PY
CONFIG=$FIX/afd.json
cat > "$CONFIG" <<'JSON'
{
  "layers": 40,
  "rows": 4096,
  "peers": [
    {"dev": "mlx5_0", "gid": 3, "ip": "192.0.2.11", "port": 47101},
    {"dev": "mlx5_1", "gid": 4, "ip": "192.0.2.12", "port": 47120}
  ]
}
JSON
BASE=(DRY_RUN=1 SKIP_EDGES=1 GPU_TOTAL_MIB=97887 NAME=launch-test
      DSV41_SKIP_LOCAL_ENV=1 AFD_NATIVE_DIR="$FIX/native" AFD_BIN_DIR="$FIX/bin"
      VLLM_SRC="$FIX/vllm" MODEL_DIR="$FIX/model" IMAGE=test-image:v1
      AFD_CONFIG_FILE="$CONFIG" AFD_LAYERS=40 AFD_ROWS=4096 PATH="$FIX/bin:$PATH"
      STUB_IMAGE="$FIX/image")

fail=0
OUT=""
FLAT=""
RC=0
pass() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

run() {
  OUT=$(env "${UNSETS[@]}" "${BASE[@]}" "$@" bash "$LAUNCH" 2>&1)
  RC=$?
  FLAT=${OUT//\\/}
}

expect_ok() {
  local name=$1; shift
  run "$@"
  if (( RC == 0 )); then pass "$name"; else
    bad "$name (rc=$RC)"; printf '%s\n' "$OUT" | tail -n 6
  fi
}

expect_fail() {
  local name=$1 pat=$2; shift 2
  run "$@"
  if (( RC != 0 )) && [[ "$FLAT" == *"$pat"* ]]; then pass "$name"; else
    bad "$name (rc=$RC, want substring '$pat')"; printf '%s\n' "$OUT" | tail -n 6
  fi
}

env_is() { grep -Fxq "ENV $1" <<<"$OUT"; }
env_absent() { ! grep -q "^ENV $1" <<<"$OUT"; }
flat_has() { [[ "$FLAT" == *"$1"* ]]; }
flat_count() { grep -o -- "$1" <<<"$FLAT" | wc -l; }

# Edge-helper seam: any ssh/scp attempt leaves a marker and fails. The helper's
# DRY_RUN path must exit before this directory is used.
NO_SSH=$FIX/no-ssh
SSH_MARKER=$FIX/ssh-called
mkdir -p "$NO_SSH"
for tool in ssh scp; do
  cat > "$NO_SSH/$tool" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$0 $*" >> "${SSH_MARKER:?}"
exit 1
SH
  chmod +x "$NO_SSH/$tool"
done
: > "$FIX/bin/edge"

edge_run() {
  rm -f "$SSH_MARKER"
  OUT=$(env -u GX1_HOST -u GX2_HOST -u AFD_LAYERS -u AFD_ROWS -u DRY_RUN \
    DSV41_SKIP_LOCAL_ENV=1 AFD_BIN_DIR="$FIX/bin" AFD_EDGE_BIN="$FIX/bin/edge" \
    AFD_CONFIG_FILE="$CONFIG" PATH="$NO_SSH:$PATH" SSH_MARKER="$SSH_MARKER" \
    "$@" bash "$ROOT/scripts/restart_afd_edges.sh" 2>&1)
  RC=$?
  FLAT=${OUT//\\/}
}
edge_ok() {
  local name=$1; shift
  edge_run "$@"
  if (( RC == 0 )) && flat_has "DRY_RUN=1"; then pass "$name"; else
    bad "$name (rc=$RC)"; printf '%s\n' "$OUT" | tail -n 6
  fi
  [[ ! -e "$SSH_MARKER" ]] || bad "$name attempted ssh/scp"
}
edge_fail() {
  local name=$1 pat=$2; shift 2
  edge_run "$@"
  if (( RC != 0 )) && [[ "$FLAT" == *"$pat"* ]]; then pass "$name"; else
    bad "$name (rc=$RC, want substring '$pat')"; printf '%s\n' "$OUT" | tail -n 6
  fi
  [[ ! -e "$SSH_MARKER" ]] || bad "$name attempted ssh/scp"
}

# --- non-spec rollback -------------------------------------------------------
expect_ok "SPEC_TOKENS=0 dry run"
env_is "SPEC_TOKENS=0" || bad "non-spec SPEC_TOKENS=0 not passed through"
env_is "GPU_MEM_UTIL=0.90" || bad "non-spec GPU_MEM_UTIL=0.90 default lost"
env_absent "KV_" || bad "non-spec launch sets a KV_* override"
flat_has "capture_sizes\":[1]" || bad "non-spec capture list is not [1]"
flat_has "--max-cudagraph-capture-size 1" || bad "non-spec max capture is not 1"
env_is "MAX_MODEL_LEN=8192" || bad "default MAX_MODEL_LEN is not 8192"
env_is "MAX_NUM_SEQS=1" || bad "default MAX_NUM_SEQS is not 1"
env_is "MAX_NUM_BATCHED_TOKENS=4096" || bad "default MAX_NUM_BATCHED_TOKENS is not 4096"
[[ "$OUT" != *"--kv-cache-memory-bytes"* ]] || bad "default launch emits an explicit KV size"
[[ "$OUT" != *"DSPARK_ENGINE_EVENT_FILE"* ]] || bad "default launch enables the engine event hook"
[[ "$OUT" != *"DOCKER_EXTRA AFD_NUM_EXPERTS=384"* ]] || bad "non-spec launch gained AFD_NUM_EXPERTS"
[[ "$OUT" != *"$PKG/overlay"* ]] || bad "non-spec launch mounts the retention overlay"
pass "non-spec args unchanged (no retention, no KV_*, util .90, capture 1)"
flat_has "$CONFIG:/opt/afd/afd.json:ro" || bad "peer config is not bind-mounted"
flat_has "$FIX/bin:/opt/qafd:ro" || bad "AFD_BIN_DIR is not mounted at /opt/qafd"
flat_has "-v $FIX:$FIX:ro" || bad "MODEL_MOUNT does not follow MODEL_DIR"
flat_has "DOCKER_EXTRA AFD_CONFIG=/opt/afd/afd.json" || bad "AFD_CONFIG container path changed"
flat_has "DOCKER_EXTRA QMESH_AFD_BIN=/opt/qafd" || bad "QMESH_AFD_BIN container path changed"
flat_has "DOCKER_EXTRA AFD_ROWS=4096" || bad "AFD_ROWS not passed"
flat_has "DOCKER_EXTRA AFD_LAYERS=40" || bad "AFD_LAYERS not passed"
pass "portable mounts, config path, transport bin path"

# --- validated points --------------------------------------------------------
expect_ok "SPEC_TOKENS=1 defaults" SPEC_TOKENS=1 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
flat_has "capture_sizes\":[1,2]" || bad "k=1 capture is not [1,2]"
flat_has "--max-cudagraph-capture-size 2" || bad "k=1 max capture is not 2"

expect_ok "SPEC_TOKENS=3 defaults" SPEC_TOKENS=3 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
flat_has "capture_sizes\":[1,3,4]" || bad "k=3 capture is not [1,3,4]"
flat_has "--max-cudagraph-capture-size 4" || bad "k=3 max capture is not 4"
env_is "SPEC_TOKENS=3" || bad "k=3 SPEC_TOKENS not passed through"
env_is "GPU_MEM_UTIL=0.97" || bad "k=3 GPU_MEM_UTIL not passed through"
env_is "KV_RECORD=nvfp4" || bad "k=3 KV_RECORD not pinned"
env_is "INDEXER_KV_DTYPE=mxfp4" || bad "k=3 INDEXER_KV_DTYPE not pinned"
env_is "KV_MODE=auto" || bad "k=3 KV_MODE not pinned"
env_is "KV_OFFLOAD_GIB=0" || bad "k=3 KV_OFFLOAD_GIB not pinned"
flat_has "DOCKER_EXTRA AFD_NUM_EXPERTS=384" || bad "k=3 AFD_NUM_EXPERTS not set"
flat_has "$PKG/overlay/vllm/v1/core/kv_cache_planning.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_planning.py:ro" ||
  bad "retention kv_cache_planning.py mount missing"
flat_has "$PKG/overlay/vllm/v1/attention/backends/mla/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro" ||
  bad "retention sparse_swa.py mount missing"
[[ "$OUT" == *"extra_retained_tokens=4"* ]] || bad "k=3 retention bound not logged as k+1"
pass "k=3 retention mounts + KV recipe + capture"

expect_ok "SPEC_TOKENS=5 defaults" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
flat_has "capture_sizes\":[1,5,6]" || bad "k=5 capture is not [1,5,6]"

expect_ok "SPEC_TOKENS=7 defaults" SPEC_TOKENS=7 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
flat_has "capture_sizes\":[1,7,8]" || bad "k=7 capture is not [1,7,8]"
flat_has "--max-cudagraph-capture-size 8" || bad "k=7 max capture is not 8"
env_is "SPEC_TOKENS=7" || bad "k=7 SPEC_TOKENS not passed through"
env_is "GPU_MEM_UTIL=0.97" || bad "k=7 GPU_MEM_UTIL not passed through"
env_is "KV_RECORD=nvfp4" || bad "k=7 KV_RECORD not pinned"
env_is "INDEXER_KV_DTYPE=mxfp4" || bad "k=7 INDEXER_KV_DTYPE not pinned"
env_is "KV_MODE=auto" || bad "k=7 KV_MODE not pinned"
env_is "KV_OFFLOAD_GIB=0" || bad "k=7 KV_OFFLOAD_GIB not pinned"
flat_has "DOCKER_EXTRA AFD_NUM_EXPERTS=384" || bad "k=7 AFD_NUM_EXPERTS not set"
flat_has "$PKG/overlay/vllm/v1/core/kv_cache_planning.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_planning.py:ro" ||
  bad "k=7 retention kv_cache_planning.py mount missing"
flat_has "$PKG/overlay/vllm/v1/attention/backends/mla/sparse_swa.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro" ||
  bad "k=7 retention sparse_swa.py mount missing"
[[ "$OUT" == *"extra_retained_tokens=8"* ]] || bad "k=7 retention bound not logged as k+1"
pass "k=7 default capture [1,7,8], exact-8 retention, KV recipe"

expect_ok "SPEC_TOKENS=2 explicit captures" SPEC_TOKENS=2 CAPTURE_SIZES=1,2,3 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
flat_has "capture_sizes\":[1,2,3]" || bad "k=2 explicit capture list not passed"

expect_ok "util .98 with DSPARK_MEM_ACK=1" SPEC_TOKENS=3 GPU_MEM_UTIL=0.98 \
  DSPARK_EXTRA_GIB=1.0 DSPARK_MEM_ACK=1

# --- explicit context/cache/concurrency (benchmark contract) -----------------
expect_ok "k=5 C=2 derives exact target captures" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=2
flat_has "capture_sizes\":[1,5,6,12]" || bad "k=5 C=2 capture is not [1,5,6,12]"
flat_has "--max-cudagraph-capture-size 12" || bad "k=5 C=2 max capture is not 12"

expect_ok "k=5 C=4 derives exact target captures" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=4
flat_has "capture_sizes\":[1,5,6,12,24]" || bad "k=5 C=4 capture is not [1,5,6,12,24]"
flat_has "--max-cudagraph-capture-size 24" || bad "k=5 C=4 max capture is not 24"

expect_ok "k=5 C=8 derives exact target captures" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=8
flat_has "capture_sizes\":[1,5,6,12,24,48]" || bad "k=5 C=8 capture is not [1,5,6,12,24,48]"
flat_has "--max-cudagraph-capture-size 48" || bad "k=5 C=8 max capture is not 48"

expect_ok "native 1M C=8 k=5 benchmark render" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=4096 KV_CACHE_GIB=3 \
  CAPTURE_SIZES=1,5,6,12,24,48 MAX_CAPTURE=48 \
  DSPARK_ENGINE_EVENT_FILE=/tmp/afd_dump/cwb-engine-events.jsonl
env_is "MAX_MODEL_LEN=1048576" || bad "native 1M MAX_MODEL_LEN not passed"
env_is "MAX_NUM_SEQS=8" || bad "C=8 MAX_NUM_SEQS not passed"
env_is "MAX_NUM_BATCHED_TOKENS=4096" || bad "batched-token budget not passed"
flat_has "capture_sizes\":[1,5,6,12,24,48]" || bad "benchmark capture list not passed"
flat_has "--max-cudagraph-capture-size 48" || bad "benchmark max capture is not 48"
[[ $(flat_count "--kv-cache-memory-bytes") == 1 ]] || bad "explicit KV flag is not emitted exactly once"
flat_has "--kv-cache-memory-bytes 3221225472" || bad "3 GiB KV is not 3221225472 bytes"
[[ $(flat_count "--max-cudagraph-capture-size") == 1 ]] || bad "launcher emits more than one capture scalar"
grep -Fxq "DOCKER_EXTRA -e" <<<"$OUT" || bad "event flag lacks its -e element"
grep -Fxq "DOCKER_EXTRA DSPARK_ENGINE_EVENT_FILE=/tmp/afd_dump/cwb-engine-events.jsonl" <<<"$OUT" ||
  bad "opt-in event file not passed to the container"
[[ "$OUT" != *"--enforce-eager"* ]] || bad "benchmark render unexpectedly eager"
pass "native 1M C=8 contract: context, 3 GiB KV, exact captures, opt-in event"

expect_ok "KV_CACHE_BYTES is equivalent to KV_CACHE_GIB" SPEC_TOKENS=5 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 KV_CACHE_BYTES=3221225472
flat_has "--kv-cache-memory-bytes 3221225472" || bad "KV_CACHE_BYTES not passed verbatim"

# --- refusals: context/cache/event knobs ------------------------------------
expect_fail "MAX_MODEL_LEN above native cap rejected" \
  "exceeds native max_position_embeddings 1048576" SPEC_TOKENS=5 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 MAX_MODEL_LEN=1048577
expect_fail "MAX_NUM_SEQS=0 rejected" "MAX_NUM_SEQS must be a positive integer" \
  SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=0
expect_fail "batched tokens below concurrency rejected" \
  "must be >= MAX_NUM_SEQS=8" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 \
  MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=4
expect_fail "batched tokens above context*seqs rejected" \
  "exceeds MAX_NUM_SEQS*MAX_MODEL_LEN" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 MAX_MODEL_LEN=1 MAX_NUM_BATCHED_TOKENS=4096
expect_fail "batched tokens below largest capture rejected" \
  "smaller than the largest emitted capture size 48" SPEC_TOKENS=5 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=8 \
  MAX_NUM_BATCHED_TOKENS=32
expect_fail "MAX_CAPTURE above list max rejected" \
  "must equal the largest capture size (48)" SPEC_TOKENS=5 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=8 \
  CAPTURE_SIZES=1,5,6,12,24,48 MAX_CAPTURE=64
expect_fail "KV_CACHE_GIB=0 rejected" "KV_CACHE_GIB must be a positive integer" \
  SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 KV_CACHE_GIB=0
expect_fail "KV_CACHE_BYTES non-integer rejected" \
  "KV_CACHE_BYTES must be a positive integer" SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 \
  DSPARK_EXTRA_GIB=1.0 KV_CACHE_BYTES=3g
expect_fail "conflicting KV_GIB/KV_BYTES rejected" "disagrees with KV_CACHE_BYTES" \
  SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 \
  KV_CACHE_GIB=3 KV_CACHE_BYTES=3221225473
expect_fail "C=8 list missing 48 rejected" "exact target width 48" SPEC_TOKENS=5 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 MAX_NUM_SEQS=8 \
  CAPTURE_SIZES=1,5,6,12,24
expect_fail "event file outside /tmp/afd_dump rejected" "must be under /tmp/afd_dump" \
  SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 \
  DSPARK_ENGINE_EVENT_FILE=/tmp/engine-events.jsonl

# --- refusals ----------------------------------------------------------------
expect_fail "SPEC_TOKENS=6 rejected" "SPEC_TOKENS must be 0..5 or 7" SPEC_TOKENS=6 \
  GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=2 without explicit captures rejected" "no validated capture default" \
  SPEC_TOKENS=2 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=3 list missing verify width rejected" "verify width 4" \
  SPEC_TOKENS=3 CAPTURE_SIZES=1,3 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=3 list missing 1 rejected" "must include 1" \
  SPEC_TOKENS=3 CAPTURE_SIZES=3,4 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=7 list missing verify width rejected" "verify width 8" \
  SPEC_TOKENS=7 CAPTURE_SIZES=1,7 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=7 list missing 1 rejected" "must include 1" \
  SPEC_TOKENS=7 CAPTURE_SIZES=7,8 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0
expect_fail "k=3 without GPU_MEM_UTIL rejected" "requires an explicit GPU_MEM_UTIL" \
  SPEC_TOKENS=3 DSPARK_EXTRA_GIB=1.0
expect_fail "k=3 without DSPARK_EXTRA_GIB rejected" "requires explicit DSPARK_EXTRA_GIB" \
  SPEC_TOKENS=3 GPU_MEM_UTIL=0.97
expect_fail "k=3 at .90 rejected by budget" "does not cover baseline" \
  SPEC_TOKENS=3 GPU_MEM_UTIL=0.90 DSPARK_EXTRA_GIB=1.0
expect_fail "util .98 without ACK rejected" "needs DSPARK_MEM_ACK=1" \
  SPEC_TOKENS=3 GPU_MEM_UTIL=0.98 DSPARK_EXTRA_GIB=1.0
# --- peer config validation --------------------------------------------------
printf '{ bad json\n' > "$FIX/bad.json"
printf '{"layers":40,"rows":4096,"peers":[]}\n' > "$FIX/nopeers.json"
printf '{"layers":40,"rows":4096,"peers":[{"dev":"mlx5_0","gid":3,"ip":"192.0.2.11","port":0},{"dev":"mlx5_1","gid":4,"ip":"192.0.2.12","port":47120}]}\n' > "$FIX/badport.json"
printf '{"layers":39,"rows":4096,"peers":[{"dev":"mlx5_0","gid":3,"ip":"192.0.2.11","port":47101},{"dev":"mlx5_1","gid":4,"ip":"192.0.2.12","port":47120}]}\n' > "$FIX/badlayers.json"
expect_fail "missing peer config rejected" "AFD_CONFIG_FILE not found" \
  AFD_CONFIG_FILE="$FIX/nope.json"
expect_fail "malformed peer config rejected" "not valid JSON" \
  AFD_CONFIG_FILE="$FIX/bad.json"
expect_fail "peer config without two peers rejected" "exactly two" \
  AFD_CONFIG_FILE="$FIX/nopeers.json"
expect_fail "peer config bad port rejected" "'port' must be 1..65535" \
  AFD_CONFIG_FILE="$FIX/badport.json"
expect_fail "peer config geometry mismatch rejected" "wire geometry must match" \
  AFD_CONFIG_FILE="$FIX/badlayers.json"
expect_ok "public sample accepted under DRY_RUN" \
  AFD_CONFIG_FILE="$ROOT/p7/afd_overlay/afd.example.json"
flat_has "DRY_RUN accepts it for validation only" ||
  bad "DRY_RUN public-sample note missing"
expect_fail "public sample refused for real launch" "refusing a real launch" \
  DRY_RUN=0 AFD_CONFIG_FILE="$ROOT/p7/afd_overlay/afd.example.json"
pass "peer config fail-closed cases"

# --- edge helper: DRY_RUN must never reach ssh/scp ---------------------------
edge_ok "edge helper DRY_RUN without hosts" DRY_RUN=1
flat_has "host1=<unset>" || bad "DRY_RUN host plan missing"
edge_ok "edge helper DRY_RUN with explicit hosts" DRY_RUN=1 \
  GX1_HOST=test-gx1 GX2_HOST=test-gx2
edge_ok "edge helper DRY_RUN accepts public sample" DRY_RUN=1 \
  AFD_CONFIG_FILE="$ROOT/p7/afd_overlay/afd.example.json"
flat_has "DRY_RUN accepts it for validation only" ||
  bad "edge helper DRY_RUN public-sample note missing"
edge_fail "edge helper real launch refuses public sample" "refusing public_sample" \
  DRY_RUN=0 GX1_HOST=test-gx1 GX2_HOST=test-gx2 \
  AFD_CONFIG_FILE="$ROOT/p7/afd_overlay/afd.example.json"
edge_fail "edge helper real launch requires hosts" "set GX1_HOST" DRY_RUN=0
edge_fail "edge helper rejects bad geometry" "positive integer" \
  DRY_RUN=1 AFD_LAYERS=0
edge_fail "edge helper rejects DRY_RUN=2" "DRY_RUN must be 0 or 1" DRY_RUN=2
pass "edge helper dry-run/network guard"

# A locally edited overlay must not boot: copy the package, mutate one byte.
tmp=$(mktemp -d); TMPDIRS+=("$tmp")
cp -a "$PKG/." "$tmp/"
printf '\n# tampered\n' >>"$tmp/overlay/vllm/v1/core/kv_cache_planning.py"
expect_fail "tampered retention overlay rejected" "retention overlay hash mismatch" \
  SPEC_TOKENS=3 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1.0 DSPARK_RETENTION_DIR="$tmp"

# The packaged overlay itself must still be the exact expected pair.
check_sha() {
  local file=$1 want=$2 got
  got=$(sha256sum "$file"); got=${got%% *}
  [[ "$got" == "$want" ]] || bad "hash drift: $file ($got != $want)"
}
check_sha "$PKG/src/vllm/v1/core/kv_cache_planning.py" \
  dd03a0eb2a75eebdc6b912f0cffa50bb6d3e06f0810c4d964bc9de30dc2a3360
check_sha "$PKG/src/vllm/v1/attention/backends/mla/sparse_swa.py" \
  66b246e8d2d1ea5fbd192da1c9250c2e47bc9824c774cb5e22bf093277bc1fe4
check_sha "$PKG/overlay/vllm/v1/core/kv_cache_planning.py" \
  103daabfd0df9e9a8e52938a781e119c3da6cafce08308190d5672a0631db06a
check_sha "$PKG/overlay/vllm/v1/attention/backends/mla/sparse_swa.py" \
  363268c27d32d652bca0c3b17098574cdaf029255e81dd17c8cc465e723b16f2
if bash "$PKG/package.sh" --check >/dev/null 2>&1; then
  pass "retention package self-check"
else
  bad "retention package self-check"
fi

# --- native-context indexer bound + mxfp8 dequant ----------------------------
# The launcher must mount both files at the paths the engine imports and
# preflight the image bytes there. The fixture image tree mirrors the image
# layout; a doubled `vllm/vllm` destination has no file here and cannot pass.
NPKG=$ROOT/p7/native_context_memory
NCTX_REL=vllm/v1/attention/backends/mla/indexer.py
NCTX_DEST=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py
NCTX_MXFP8_REL=vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
NCTX_MXFP8_DEST=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py
mkdir -p "$FIX/image${NCTX_DEST%/*}" "$FIX/image${NCTX_MXFP8_DEST%/*}"
cp "$NPKG/src/$NCTX_REL" "$FIX/image$NCTX_DEST"
cp "$NPKG/src/$NCTX_MXFP8_REL" "$FIX/image$NCTX_MXFP8_DEST"

expect_ok "NATIVE_CTX_MEM=1 mounts the imported paths" NATIVE_CTX_MEM=1
flat_has "$NPKG/overlay/$NCTX_REL:$NCTX_DEST:ro" ||
  bad "native-context indexer mount is not the imported path"
flat_has "$NPKG/overlay/$NCTX_MXFP8_REL:$NCTX_MXFP8_DEST:ro" ||
  bad "native-context mxfp8 mount is not the imported path"
(( $(flat_count "vllm/vllm") == 0 )) || bad "native-context mount uses a doubled vllm path"
flat_has "native-context-memory overlay on (factor=" || bad "native-context overlay not enabled"
pass "NATIVE_CTX_MEM=1 two-path mount"

expect_ok "NATIVE_CTX_MEM=0 default" NATIVE_CTX_MEM=0
(( $(flat_count "native_context_memory/overlay") == 0 )) || bad "native-context mounts while off"
pass "NATIVE_CTX_MEM=0 mounts nothing"

printf '\n# drift\n' >>"$FIX/image$NCTX_DEST"
expect_fail "native-context image destination verified" "not the pinned image source" NATIVE_CTX_MEM=1
cp "$NPKG/src/$NCTX_REL" "$FIX/image$NCTX_DEST"
rm -f "$FIX/image$NCTX_DEST"
expect_fail "native-context missing image path rejected" "not the pinned image source" NATIVE_CTX_MEM=1
cp "$NPKG/src/$NCTX_REL" "$FIX/image$NCTX_DEST"
printf '\n# drift\n' >>"$FIX/image$NCTX_MXFP8_DEST"
expect_fail "native-context mxfp8 image destination verified" "not the pinned image source" NATIVE_CTX_MEM=1
cp "$NPKG/src/$NCTX_MXFP8_REL" "$FIX/image$NCTX_MXFP8_DEST"
pass "native-context image preflight rejects drift/missing path"

if (( fail )); then
  echo "FAILED"
  exit 1
fi
echo "PASS: launcher dry-run contract (non-spec rollback, k=1/2/3/5/7, retention mounts, memory guards, overlay hash guard)"
echo "PASS: native-context overlay mount = imported path, image preflight"
echo "PASS: edge helper DRY_RUN never reaches ssh/scp"
