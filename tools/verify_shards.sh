#!/usr/bin/env bash
# Verify downloaded shards against pinned Hugging Face LFS sha256 lists. Read-only.
# MODEL_ROOT is the directory holding the downloaded checkpoint dirs
# (default /models/dsv41); sha256 entries name files relative to it.
set -uo pipefail
ROOT=${MODEL_ROOT:-/models/dsv41}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Usage: [MODEL_ROOT=/models/dsv41] tools/verify_shards.sh [sha256-list] [log]

  sha256-list  entries are "<sha256>  <path relative to MODEL_ROOT>"
               (default: the tracked nvfp4.sha256 next to this script)
  log          output log (default: MODEL_ROOT/verify.log)

Examples:
  tools/verify_shards.sh                       # NVFP4 shards, 48/48 expected
  tools/verify_shards.sh tools/orig.sha256     # original Engram shards, 2/2 expected
EOF
}

case ${1:-} in
  -h|--help) usage; exit 0 ;;
esac

LIST=${1:-$HERE/nvfp4.sha256}
[[ -f $LIST ]] || { echo "sha256 list not found: $LIST" >&2; exit 2; }
LIST=$(readlink -f -- "$LIST")

OUT=${2:-$ROOT/verify.log}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac

[[ -d $ROOT ]] || { echo "MODEL_ROOT is not a directory: $ROOT" >&2; exit 2; }
cd "$ROOT"
: > "$OUT" || { echo "cannot write log: $OUT" >&2; exit 2; }
xargs -P 12 -L 1 sh -c 'echo "$0  $1" | sha256sum -c - 2>&1' < "$LIST" >> "$OUT"
echo "VERIFY_DONE ok=$(grep -c ': OK$' "$OUT") failed=$(grep -vc -e ': OK$' -e VERIFY_DONE "$OUT")" >> "$OUT"
