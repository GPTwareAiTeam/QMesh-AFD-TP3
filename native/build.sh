#!/usr/bin/env bash
# Build the AFD-TP3 native transport from the sources in this directory.
#
# Offline and side-effect free: reads this directory, writes only AFD_BIN_DIR.
# It never contacts a GPU, an engine, or the network, and never writes the
# repository root bin/.
#
# Targets:
#   all    both artifacts (default)
#   edge   $AFD_BIN_DIR/qmesh_afd_edge.$(uname -m)   server/client wire engine
#   ext    $AFD_BIN_DIR/_qmesh_afd$(EXT_SUFFIX)      CPython extension (Peer/Seam)
#
# Env:
#   AFD_NATIVE_DIR  source directory (default: the directory containing this script)
#   AFD_BIN_DIR     output directory (default: $AFD_NATIVE_DIR/bin)
#   NVCC            nvcc to use (default: $CUDA_HOME/bin/nvcc, /usr/local/cuda/bin/nvcc, PATH)
#   CUDA_ARCH       gencode target (default: x86_64 -> sm_120, aarch64 -> sm_121)
#   CUDA_HOME       toolkit root used for the nvcc default and rpath
#   PYTHON          interpreter whose headers/EXT_SUFFIX build the extension (default: python3)
#
# Requires a CUDA 13.x toolkit, libibverbs development headers (rdma-core), and
# Python 3 development headers for the ext target. Both artifacts link the
# system libibverbs and libcudart; nothing is vendored or downloaded.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${AFD_NATIVE_DIR:-$SCRIPT_DIR}"
BIN="${AFD_BIN_DIR:-$SRC/bin}"
MACH="$(uname -m)"

die() { echo "build.sh: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Build the AFD-TP3 native transport from the sources in this directory.

Offline and side-effect free: reads this directory, writes only AFD_BIN_DIR.
It never contacts a GPU, an engine, or the network, and never writes the
repository root bin/.

Targets:
  all    both artifacts (default)
  edge   $AFD_BIN_DIR/qmesh_afd_edge.$(uname -m)   server/client wire engine
  ext    $AFD_BIN_DIR/_qmesh_afd$(EXT_SUFFIX)      CPython extension (Peer/Seam)

Env:
  AFD_NATIVE_DIR  source directory (default: the directory containing this script)
  AFD_BIN_DIR     output directory (default: $AFD_NATIVE_DIR/bin)
  NVCC            nvcc to use (default: $CUDA_HOME/bin/nvcc, /usr/local/cuda/bin/nvcc, PATH)
  CUDA_ARCH       gencode target (default: x86_64 -> sm_120, aarch64 -> sm_121)
  CUDA_HOME       toolkit root used for the nvcc default and rpath
  PYTHON          interpreter whose headers/EXT_SUFFIX build the extension (default: python3)

Requires a CUDA 13.x toolkit, libibverbs development headers (rdma-core), and
Python 3 development headers for the ext target. Both artifacts link the
system libibverbs and libcudart; nothing is vendored or downloaded.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
  esac
done

[[ -f "$SRC/qmesh_afd_peer.cpp" && -f "$SRC/qmesh_afd_edge.cpp" ]] \
  || die "AFD_NATIVE_DIR=$SRC does not contain the vendored AFD sources"

if [[ -z "${NVCC:-}" ]]; then
  if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    NVCC="$CUDA_HOME/bin/nvcc"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC=/usr/local/cuda/bin/nvcc
  else
    NVCC="$(command -v nvcc || true)"
  fi
fi
[[ -x "$NVCC" ]] || die "nvcc not found (set NVCC or CUDA_HOME)"

if [[ -z "${CUDA_ARCH:-}" ]]; then
  case "$MACH" in
    x86_64)  CUDA_ARCH=sm_120 ;;
    aarch64) CUDA_ARCH=sm_121 ;;
    *)       die "unsupported machine '$MACH'; set CUDA_ARCH" ;;
  esac
fi

case "$MACH" in
  x86_64)  LIBDIR_MAIN=/usr/lib/x86_64-linux-gnu ;;
  aarch64) LIBDIR_MAIN=/usr/lib/aarch64-linux-gnu ;;
  *)       LIBDIR_MAIN="/usr/lib/$MACH-linux-gnu" ;;
esac
[[ -d "$LIBDIR_MAIN" ]] || LIBDIR_MAIN=/usr/lib

PYTHON="${PYTHON:-python3}"
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
  LIBDIR_CUDA="$CUDA_HOME/lib64"
elif [[ -d /usr/local/cuda/lib64 ]]; then
  LIBDIR_CUDA=/usr/local/cuda/lib64
else
  LIBDIR_CUDA=""
fi

# Force system libibverbs; never pick up NIXL or other mangled libs from the caller.
unset LD_LIBRARY_PATH || true
export LIBRARY_PATH="$LIBDIR_MAIN${LIBRARY_PATH:+:$LIBRARY_PATH}"

RPATH=(-Xlinker "-rpath=$LIBDIR_MAIN")
[[ -n "$LIBDIR_CUDA" ]] && RPATH+=(-Xlinker "-rpath=$LIBDIR_CUDA")

build_edge() {
  local out="$BIN/qmesh_afd_edge.$MACH"
  echo "Building $out with $NVCC ($CUDA_ARCH)"
  "$NVCC" -O2 -std=c++17 -x cu -arch="$CUDA_ARCH" \
    "$SRC/qmesh_afd_edge.cpp" \
    -o "$out" \
    -libverbs -lcuda -lcudart \
    "${RPATH[@]}"
  echo "BUILD_OK: $out"
}

build_ext() {
  command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON not found"
  local py_inc ext out
  py_inc="$("$PYTHON" -c 'import sysconfig; print("-I" + sysconfig.get_path("include"))')" \
    || die "cannot query Python include path"
  ext="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')" \
    || die "cannot query Python EXT_SUFFIX"
  [[ -n "$ext" ]] || die "Python EXT_SUFFIX is empty; install Python development headers"
  out="$BIN/_qmesh_afd$ext"
  echo "Building $out with $NVCC ($CUDA_ARCH, $PYTHON)"
  # shellcheck disable=SC2086  # py_inc is intentionally one -I flag
  "$NVCC" -O3 -std=c++17 -x cu -shared -Xcompiler -fPIC -arch="$CUDA_ARCH" \
    $py_inc \
    "$SRC/qmesh_afd_peer.cpp" \
    -o "$out" \
    -libverbs -lcudart \
    "${RPATH[@]}"
  echo "BUILD_OK: $out"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(all)
for t in "${targets[@]}"; do
  case "$t" in
    all|edge|ext) ;;
    *) die "unknown target '$t' (all|edge|ext)" ;;
  esac
done

mkdir -p "$BIN"
for t in "${targets[@]}"; do
  case "$t" in
    all) build_edge; build_ext ;;
    edge) build_edge ;;
    ext) build_ext ;;
  esac
done
