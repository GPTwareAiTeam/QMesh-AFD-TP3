#!/usr/bin/env bash
# Gate A transport benchmark launcher (runs on the P7 host). Two independent
# 2-rank NCCL links, each fully isolated on its own /30 rail.
#
# Every device, address and ssh target is explicit; this file has no site
# addresses or GPU UUIDs. The measured host used one GPU only, so P7_GPU must
# name GPU0 (e.g. device=0 or device=GPU-...). Values below are examples.
#
# Required per link (L = 1, 2):
#   P7_HCA_L    P7 RDMA device for the rail        (e.g. rocep142s0f0)
#   P7_IF_L     P7 netdev for the rail              (e.g. enp142s0f0np0)
#   P7_IP_L     P7 address on the rail's /30        (bootstrap + data)
#   GX_HOST_L   ssh target of the GX for the rail   (e.g. gx1)
#   GX_IF_L     GX netdev for the rail              (e.g. enp1s0f0np0)
#   GX_HCA_L    GX RDMA device for the rail         (e.g. rocep1s0f0)
#
# Required once:
#   P7_GPU      docker --gpus selector for GPU0     (e.g. device=0)
#
# Optional:
#   HERE        directory holding xfer_bench.py, and results/ (default: this dir)
#   P7_IMG, GX_IMG   images (defaults are the measured tags)
#   GX_USER     ssh user on the GX hosts (default: current user)
#   GX_GPU      docker --gpus selector on the GX (default: all)
#   PORT_L      TCP bootstrap port per link (default: 46211, 46212)
#   BNXT_PROVIDER  host bnxt rdma provider .so to mount into the P7 container
#                  (default: the distro path used for the measurement; `none`
#                  disables the mount and leaves the image's provider in place)
#   TAG, LINKS, TMO, LAYERS, ROWS, ITERS, WARM, NCCL_DEBUG, EXTRA_ENV
set -uo pipefail

HERE=${HERE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
P7_IMG=${P7_IMG:-qmesh-nccl2312-official-p7:20260814-2}
GX_IMG=${GX_IMG:-qmesh-nccl2312-official-gx10:20260814-3}
GX_USER=${GX_USER:-${USER:-$(id -un)}}
GX_GPU=${GX_GPU:-all}
TAG=${TAG:-$(date +%Y%m%d-%H%M%S)}
OUT=${OUT:-$HERE/results/$TAG}
LINKS=${LINKS:-"1 2"}
TMO=${TMO:-300}
BNXT_PROVIDER=${BNXT_PROVIDER-/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so}

: "${P7_GPU:?set P7_GPU to the GPU0 docker selector, e.g. device=0}"

die() { echo "run_xfer: $*" >&2; exit 2; }

# link -> values (no defaults: a wrong rail silently reuses another link's device)
for L in $LINKS; do
  for v in P7_HCA P7_IF P7_IP GX_HOST GX_IF GX_HCA; do
    name=${v}_$L
    [[ -n "${!name:-}" ]] || die "$name is not set"
  done
done


mkdir -p "$OUT" || die "cannot create $OUT"

env_common=(-e "LAYERS=${LAYERS:-40}" -e "ROWS=${ROWS:-1,6,48}" -e "ITERS=${ITERS:-200}" -e "WARM=${WARM:-20}"
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e "NCCL_DEBUG=${NCCL_DEBUG:-INFO}" -e NCCL_DEBUG_SUBSYS=INIT,NET -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1)
# EXTRA_ENV deliberately word-splits into additional -e flags.
# shellcheck disable=SC2206
[[ -n "${EXTRA_ENV:-}" ]] && env_common+=(${EXTRA_ENV})
flags=(--rm --network host --ipc host --shm-size 8g --device /dev/infiniband --ulimit memlock=-1:-1 --cap-add IPC_LOCK --entrypoint timeout)

for L in $LINKS; do
  n=P7_HCA_$L; p7hca=${!n}; n=P7_IF_$L; p7if=${!n}; n=P7_IP_$L; p7ip=${!n}
  n=GX_HOST_$L; gxhost=${!n}; n=GX_IF_$L; gxif=${!n}; n=GX_HCA_$L; gxhca=${!n}
  portvar=PORT_$L
  port=${!portvar:-$((46210 + L))}
  target=$GX_USER@$gxhost

  ssh -o BatchMode=yes "$target" "mkdir -p ~/afd/gateA" && scp -q "$HERE/xfer_bench.py" "$target:afd/gateA/xfer_bench.py" ||
    die "cannot copy xfer_bench.py to $target"
  # Bootstrap (TCPStore + NCCL OOB) on the link's own /30 so each link is fully isolated.
  ssh -o BatchMode=yes "$target" "docker run --name afd-xfer-gx$L ${flags[*]} --gpus $GX_GPU -v \$HOME/afd/gateA:/afd ${env_common[*]} \
    -e ROLE=gx -e LINK=$L -e MASTER_ADDR=$p7ip -e MASTER_PORT=$port \
    -e NCCL_SOCKET_IFNAME=$gxif -e GLOO_SOCKET_IFNAME=$gxif -e NCCL_IB_HCA==$gxhca:1 \
    $GX_IMG $TMO python3 /afd/xfer_bench.py" > "$OUT/gx$L.log" 2>&1 &

  # On the measured host, mounting the host rdma-core bnxt provider made
  # ibv_reg_dmabuf_mr work (EBADF probe) where the image's provider reported
  # EOPNOTSUPP. The measurement used the distro path above; `none` disables it.
  p7_mounts=(-v "$HERE:/afd")
  if [[ -n "${BNXT_PROVIDER:-}" && "${BNXT_PROVIDER:-}" != none ]]; then
    p7_mounts+=(-v "$BNXT_PROVIDER:$BNXT_PROVIDER:ro")
  fi
  docker run --name "afd-xfer-p7-$L" "${flags[@]}" --gpus "$P7_GPU" "${p7_mounts[@]}" "${env_common[@]}" \
    -e ROLE=p7 -e LINK="$L" -e MASTER_ADDR="$p7ip" -e MASTER_PORT="$port" \
    -e NCCL_SOCKET_IFNAME="$p7if" -e GLOO_SOCKET_IFNAME="$p7if" -e "NCCL_IB_HCA==$p7hca:1" \
    "$P7_IMG" "$TMO" python3 /afd/xfer_bench.py > "$OUT/p7-$L.log" 2>&1 &
done
wait
for L in $LINKS; do
  docker rm -f "afd-xfer-p7-$L" >/dev/null 2>&1
  n=GX_HOST_$L; ssh -o BatchMode=yes "$GX_USER@${!n}" docker rm -f "afd-xfer-gx$L" >/dev/null 2>&1
done
grep -h '^{' "$OUT"/p7-*.log
echo "$OUT"
