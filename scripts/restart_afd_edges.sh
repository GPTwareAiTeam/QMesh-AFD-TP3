#!/usr/bin/env bash
# Restart both AFD edge processes on the GX hosts, one session each.
#
# The launcher owns the engine; this helper owns only the host-side edge
# processes. It never starts a container or touches a model. Each edge serves
# exactly one client: when the client disconnects the relay exits, so a fresh
# engine needs a fresh edge.
#
# Configuration (environment wins; scripts/local.env provides defaults):
#   AFD_CONFIG_FILE  peer JSON, the same file the engine mounts. Each peer's
#                    ip/port is the management address the GX edge binds.
#   AFD_BIN_DIR      built transport artifacts; EDGE_BIN defaults to
#                    $AFD_BIN_DIR/qmesh_afd_edge.$GX_ARCH
#   AFD_EDGE_BIN     explicit local edge binary copied to both hosts
#   GX1_HOST/GX2_HOST  ssh targets (required unless DRY_RUN=1)
#   GX1_DEV/GX2_DEV  GX-side RDMA device (default rocep1s0f0)
#   GX1_GID/GX2_GID  GX-side GID index (default 3)
#   GX_ARCH          GX edge architecture (default aarch64)
#   AFD_LAYERS/AFD_ROWS  wire geometry (defaults from peer JSON)
#   GX_SHM           shared-memory region (default /dev/shm/qafd)
#   GX_SSH_OPTS      extra ssh/scp options
#   GX1_EDGE_LOG/GX2_EDGE_LOG  remote logs (default /tmp/e01.log, /tmp/e20.log)
#   DRY_RUN=1        validate config/paths and print the plan; never ssh/scp
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP=$(cd -- "$SCRIPT_DIR/.." && pwd)
if [[ "${DSV41_SKIP_LOCAL_ENV:-0}" != 1 && -f "$SCRIPT_DIR/local.env" ]]; then
  . "$SCRIPT_DIR/local.env"
fi

die() { echo "edges: $*" >&2; exit 1; }
DRY_RUN=${DRY_RUN:-0}
case "$DRY_RUN" in 0|1) ;; *) die "DRY_RUN must be 0 or 1, got '$DRY_RUN'";; esac

AFD_CONFIG_FILE=${AFD_CONFIG_FILE:-}
[[ -n "$AFD_CONFIG_FILE" ]] || die "AFD_CONFIG_FILE is not set (peer JSON)"
[[ -f "$AFD_CONFIG_FILE" ]] || die "AFD_CONFIG_FILE not found: $AFD_CONFIG_FILE"

AFD_BIN_DIR=${AFD_BIN_DIR:-$EXP/native/bin}
GX_ARCH=${GX_ARCH:-aarch64}
EDGE_BIN=${AFD_EDGE_BIN:-$AFD_BIN_DIR/qmesh_afd_edge.$GX_ARCH}
[[ -f "$EDGE_BIN" ]] ||
  die "edge binary not found: $EDGE_BIN (build native/build.sh edge or set AFD_EDGE_BIN)"

GX1_HOST=${GX1_HOST:-}
GX2_HOST=${GX2_HOST:-}
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -n "$GX1_HOST" && -n "$GX2_HOST" ]] || die "set GX1_HOST and GX2_HOST"
fi

GX1_DEV=${GX1_DEV:-rocep1s0f0}
GX2_DEV=${GX2_DEV:-rocep1s0f0}
GX1_GID=${GX1_GID:-3}
GX2_GID=${GX2_GID:-3}
GX_SHM=${GX_SHM:-/dev/shm/qafd}
GX1_EDGE_LOG=${GX1_EDGE_LOG:-/tmp/e01.log}
GX2_EDGE_LOG=${GX2_EDGE_LOG:-/tmp/e20.log}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)
if [[ -n "${GX_SSH_OPTS:-}" ]]; then
  read -r -a _extra_ssh <<< "$GX_SSH_OPTS"
  SSH_OPTS+=("${_extra_ssh[@]}")
fi

# Read the same peer JSON the engine uses instead of duplicating addresses.
PEER_DUMP=$(python3 - "$AFD_CONFIG_FILE" "$DRY_RUN" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    cfg = json.load(handle)
if cfg.get("public_sample"):
    msg = "edges: public_sample config; use a real peer file"
    if sys.argv[2] == "1":
        print(msg + " (DRY_RUN accepts it for validation only)", file=sys.stderr)
    else:
        sys.exit("edges: refusing public_sample config; use a real peer file")
peers = cfg.get("peers")
if not isinstance(peers, list) or len(peers) != 2:
    sys.exit("edges: AFD_CONFIG_FILE peers must be two objects")
for peer in peers:
    ip = peer.get("ip")
    port = peer.get("port")
    if not isinstance(ip, str) or not ip:
        sys.exit("edges: peer ip must be a non-empty string")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        sys.exit("edges: peer port must be 1..65535")
    print(f"{ip} {port}")
for key in ("layers", "rows"):
    value = cfg.get(key)
    if not isinstance(value, int) or value <= 0:
        sys.exit(f"edges: {key} must be a positive integer")
    print(value)
PY
) || die "cannot read peers from AFD_CONFIG_FILE=$AFD_CONFIG_FILE"
mapfile -t PEER_LINES <<< "$PEER_DUMP"
[[ ${#PEER_LINES[@]} -eq 4 ]] ||
  die "AFD_CONFIG_FILE=$AFD_CONFIG_FILE did not yield two peers"
IP1=${PEER_LINES[0]% *}
PORT1=${PEER_LINES[0]#* }
IP2=${PEER_LINES[1]% *}
PORT2=${PEER_LINES[1]#* }
LAYERS=${AFD_LAYERS:-${PEER_LINES[2]}}
ROWS=${AFD_ROWS:-${PEER_LINES[3]}}
[[ "$LAYERS" =~ ^[1-9][0-9]*$ ]] || die "AFD_LAYERS must be a positive integer: $LAYERS"
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] || die "AFD_ROWS must be a positive integer: $ROWS"
REMOTE_BIN=/tmp/qmesh_afd_edge.$GX_ARCH

HOSTS=("$GX1_HOST" "$GX2_HOST")
DEVS=("$GX1_DEV" "$GX2_DEV")
GIDS=("$GX1_GID" "$GX2_GID")
LOGS=("$GX1_EDGE_LOG" "$GX2_EDGE_LOG")
IPS=("$IP1" "$IP2")
PORTS=("$PORT1" "$PORT2")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "edges: DRY_RUN=1 (nothing copied, stopped, or started)" >&2
  printf 'edges: host1=%s host2=%s ip1=%s:%s ip2=%s:%s layers=%s rows=%s bin=%s\n' \
    "${GX1_HOST:-<unset>}" "${GX2_HOST:-<unset>}" \
    "$IP1" "$PORT1" "$IP2" "$PORT2" "$LAYERS" "$ROWS" "$EDGE_BIN" >&2
  exit 0
fi

reachable() {
  ssh "${SSH_OPTS[@]}" "$1" true >/dev/null 2>&1
}

ok_hosts=()
for host in "${HOSTS[@]}"; do
  if reachable "$host"; then
    ok_hosts+=("$host")
  else
    echo "edges: $host unreachable (skipping)" >&2
  fi
done
(( ${#ok_hosts[@]} > 0 )) || die "no GX host reachable"

failed=0
for i in 0 1; do
  host=${HOSTS[$i]}
  dev=${DEVS[$i]}
  gid=${GIDS[$i]}
  log=${LOGS[$i]}
  ip=${IPS[$i]}
  port=${PORTS[$i]}
  if ! reachable "$host"; then
    continue
  fi

  scp "${SSH_OPTS[@]}" "$EDGE_BIN" "$host:$REMOTE_BIN" >/dev/null ||
    die "scp $EDGE_BIN to $host failed"
  ssh "${SSH_OPTS[@]}" "$host" "chmod +x '$REMOTE_BIN' && pkill -f 'qmesh_afd_edge[.]$GX_ARCH' || true" ||
    die "could not prepare $host"

  remote_cmd="setsid nohup env CUDA_VISIBLE_DEVICES=0 $(printf %q "$REMOTE_BIN") --role server --serve --shm $(printf %q "$GX_SHM") --dev $(printf %q "$dev") --gid-idx $(printf %q "$gid") --layers $(printf %q "$LAYERS") --rows $(printf %q "$ROWS") --iters 0 --mock-us 0 --mgmt-ip $(printf %q "$ip") --mgmt-port $(printf %q "$port") > $(printf %q "$log") 2>&1 < /dev/null &"
  ssh "${SSH_OPTS[@]}" "$host" "$remote_cmd" ||
    die "could not start edge on $host"

  ready=0
  for _ in $(seq 1 100); do
    if ssh "${SSH_OPTS[@]}" "$host" "ss -ltn 2>/dev/null | grep -q ':$port '"; then
      ready=1
      break
    fi
    sleep 0.2
  done
  if (( ready == 0 )); then
    echo "edges: $host did not bind $ip:$port; tail of $log:" >&2
    ssh "${SSH_OPTS[@]}" "$host" "tail -n 5 '$log' || true" >&2 || true
    failed=1
  fi
done

(( failed == 0 )) || exit 1
echo "edges: ready on ${ok_hosts[*]}" >&2
