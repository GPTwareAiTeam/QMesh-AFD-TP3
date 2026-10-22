#!/usr/bin/env bash
# Consumer-chain smoke for the sm_120 mHC fused MXFP8 quant (GPU0 only, no network).
# Mounts the candidate kernel dir and the patched vLLM files over the image, so
# nothing is rebuilt or copied into the image. Engine must be stopped (exclusive GPU0).
#
# Configuration (environment wins):
#   IMAGE       image tag to run (default: vllm-moet:dsv41-213d5fa)
#   MHC_GPU     docker --gpus selector (default: device=0)
#   VLLM        vLLM package path inside the image
#   CACHE       torch extension cache on the host (default: /tmp/cache-mhc-quant)
#   VLLM_MOET_MHC_PDL   passed through (default 1)
#
#   bash run_smoke.sh --help     print this usage without touching docker
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi
ROOT=$(cd "$(dirname "$0")" && pwd)
VLLM=${VLLM:-/usr/local/lib/python3.12/dist-packages/vllm}
IMAGE=${IMAGE:-vllm-moet:dsv41-213d5fa}
CACHE=${CACHE:-/tmp/cache-mhc-quant}
mkdir -p "$CACHE"
exec docker run --rm --gpus "${MHC_GPU:-device=0}" --network none \
  --entrypoint python3 \
  -v "$ROOT/kernel":/opt/vllm-moet/dsv41_sm120/mhc_overlap:ro \
  -v "$ROOT/overlay/models/deepseek_v41/attention.py":"$VLLM"/models/deepseek_v41/attention.py:ro \
  -v "$ROOT/overlay/models/deepseek_v41/nvidia/model.py":"$VLLM"/models/deepseek_v41/nvidia/model.py:ro \
  -v "$ROOT/overlay/models/deepseek_v41/nvidia/ops/mhc.py":"$VLLM"/models/deepseek_v41/nvidia/ops/mhc.py:ro \
  -v "$ROOT":/work:ro \
  -v "$CACHE":/cache \
  -e TORCH_EXTENSIONS_DIR=/cache \
  -e VLLM_MOET_MHC_OVERLAP=1 \
  -e VLLM_MOET_MHC_OVERLAP_DIR=/opt/vllm-moet/dsv41_sm120/mhc_overlap \
  -e VLLM_MOET_MHC_PDL="${VLLM_MOET_MHC_PDL:-1}" \
  "$IMAGE" /work/smoke_mhc_quant_consumer.py "$@"
