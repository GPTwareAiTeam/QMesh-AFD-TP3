#!/usr/bin/env bash
# Hard-link snapshot backup (rsync --link-dest) with dated snapshots and rotation.
#
# Nothing is implicit: SOURCE and DEST must be passed explicitly. There is no
# user name, host name, mount point, or whole-home default in this file.
#
#   SOURCE=/home/alice DEST=/mnt/backup/alice ./gx-backup.sh
#   SOURCE=/srv/data DEST=/mnt/backup/data BACKUP_ETC=1 ./gx-backup.sh
#
# SOURCE           directory to snapshot (read)
# DEST             existing snapshot root on the backup target (write)
# REQUIRE_MOUNT    default 1: refuse unless DEST is a mount point, so an
#                  unmounted backup target cannot silently fill another disk.
#                  Set 0 for a backup target that is a plain directory.
# BACKUP_ETC       default 0: also snapshot /etc (world-readable files only;
#                  no ownership), as the original job did.
# KEEP             number of dated snapshots to keep (default 14)
# LABEL            per-tree directory name inside a snapshot
#                  (default: basename of SOURCE)
# EXTRA_EXCLUDES   additional rsync --exclude patterns, space-separated
#
# Each run writes $DEST/snapshots/<timestamp>/$LABEL/ (and .../etc/ when
# BACKUP_ETC=1). Unchanged files are hard-linked from the newest snapshot, so a
# dated snapshot costs only the changes. $DEST/current points at the newest one.
set -uo pipefail

case "${1:-}" in
  -h|--help)
    sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0 ;;
esac

die() { echo "gx-backup: $*" >&2; exit 1; }

: "${SOURCE:?set SOURCE to the directory to back up}"
: "${DEST:?set DEST to the existing snapshot root on the backup target}"

[[ -d "$SOURCE" ]] || die "SOURCE is not a directory: $SOURCE"
[[ -d "$DEST" ]] || die "DEST is not an existing directory: $DEST (create it first)"
SOURCE=$(cd -- "$SOURCE" && pwd -P)
DEST=$(cd -- "$DEST" && pwd -P)

case "$DEST/" in
  "$SOURCE"/*) die "DEST ($DEST) is inside SOURCE ($SOURCE); refusing to back up into the tree" ;;
esac
case "$SOURCE/" in
  "$DEST"/*) die "SOURCE ($SOURCE) is inside DEST ($DEST); refusing to snapshot a backup target" ;;
esac

if [[ "${REQUIRE_MOUNT:-1}" == 1 ]] && ! mountpoint -q "$DEST"; then
  die "DEST is not a mount point: $DEST (set REQUIRE_MOUNT=0 only for a plain directory target)"
fi

KEEP=${KEEP:-14}
LABEL=${LABEL:-$(basename -- "$SOURCE")}
DATE=$(date +%Y%m%d-%H%M%S)
SNAP=$DEST/snapshots/$DATE
[[ -e "$SNAP" ]] && die "snapshot already exists: $SNAP"

LAST=$(ls -1d "$DEST"/snapshots/*/ 2>/dev/null | sort | tail -1)
LAST=${LAST%/}
LINK=()
[[ -n "$LAST" ]] && LINK=(--link-dest="$LAST")

mkdir -p "$SNAP/$LABEL" "$DEST/logs" || die "cannot create $SNAP"

EXCL=(
  --exclude=/models/ --exclude=/.cache/ --exclude=/qmesh-runtime/pkg_cache/ --exclude=/qmesh-runtime/packages/
  --exclude='/qmesh-v2-packages-*' --exclude=/miniconda/pkgs/ --exclude='*.safetensors' --exclude='*.nsys-rep'
  --exclude=/.local/share/Trash/ --exclude='/snap/'
)
if [[ -n "${EXTRA_EXCLUDES:-}" ]]; then
  read -r -a _extra <<< "$EXTRA_EXCLUDES"
  EXCL+=("${_extra[@]}")
fi

rsync -aHAX --numeric-ids --delete "${LINK[@]}" "${EXCL[@]}" "$SOURCE/" "$SNAP/$LABEL/" \
  > "$DEST/logs/$DATE.log" 2>&1
rc_source=$?

rc_etc=0
if [[ "${BACKUP_ETC:-0}" == 1 ]]; then
  mkdir -p "$SNAP/etc"
  # /etc as the user (world-readable files; root-only secrets such as shadow/ssh
  # host keys are skipped).
  rsync -a --no-o --no-g "${LINK[@]}" /etc/ "$SNAP/etc/" >> "$DEST/logs/$DATE.log" 2>&1
  rc_etc=$?
fi

ln -sfn "snapshots/$DATE" "$DEST/current"
mapfile -t snaps < <(ls -1d "$DEST"/snapshots/*/ 2>/dev/null | sort)
if (( ${#snaps[@]} > KEEP )); then
  for old in "${snaps[@]:0:${#snaps[@]}-KEEP}"; do
    rm -rf -- "$old"
  done
fi
echo "$DATE source=$SOURCE label=$LABEL rc=$rc_source etc_rc=$rc_etc $(du -sh "$SNAP" | cut -f1)" \
  >> "$DEST/logs/summary.log"
exit $(( rc_source != 0 ? rc_source : rc_etc ))
