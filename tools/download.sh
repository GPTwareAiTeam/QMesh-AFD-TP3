#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 for AFD-TP3: NVFP4 experts (LibertAI, includes DSpark mtp.*)
# plus DeepSeek's original lossless FP8 Engram tables and reference files. Pinned revisions.
#
# This is the manual download step. MODEL_ROOT selects the model root
# (default /models/dsv41). The Hugging Face CLI reuses files already present,
# so rerunning does not refetch complete shards.
set -euo pipefail
ROOT=${MODEL_ROOT:-/models/dsv41}
NVFP4_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4
NVFP4_REV=5be99cd15675589fb6693d0a8651e413bcb9a396
ORIG_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
ORIG_REV=1bae0e1088993b9b82de08f59de5be5a34e42170

usage() {
  cat <<'EOF'
Usage: [MODEL_ROOT=/models/dsv41] tools/download.sh

Fetch the pinned NVFP4 checkpoint and original Engram reference files into
MODEL_ROOT. This script takes no arguments and runs only when invoked; it does
not start any download from an import, test, or --help path.
EOF
}

case ${1:-} in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
esac
echo "[$(date -Is)] start NVFP4 ${NVFP4_REPO}@${NVFP4_REV}"
hf download "$NVFP4_REPO" --revision "$NVFP4_REV" \
  --local-dir "$ROOT/DeepSeek-V4.1-Flash-1017-NVFP4"
echo "[$(date -Is)] NVFP4 done"

echo "[$(date -Is)] start original Engram shards + reference files ${ORIG_REPO}@${ORIG_REV}"
# hf download ignores --include when positional filenames are given, and treats --include
# globs only when no filenames are passed; list exact files positionally, then globs separately.
hf download "$ORIG_REPO" \
  "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" \
  "model.safetensors.index.json" "config.json" "tokenizer.json" "tokenizer_config.json" \
  "README.md" "LICENSE" \
  --revision "$ORIG_REV" --local-dir "$ROOT/DeepSeek-V4.1-Flash-1017-orig"
hf download "$ORIG_REPO" --revision "$ORIG_REV" \
  --local-dir "$ROOT/DeepSeek-V4.1-Flash-1017-orig" --include "inference/*" "encoding/*"
echo "[$(date -Is)] original done"
echo "[$(date -Is)] ALL DONE"
