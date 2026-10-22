#!/usr/bin/env bash
# Put the IOMMU groups of explicitly named RoCE NIC functions into identity
# (passthrough) mode so the NIC can DMA straight to GPU BAR1 addresses handed
# out by the nvidia DMA-BUF exporter. Evidence for the need: with DMA-FQ the
# RDMA WRITEs fault in DMAR ("first-level paging entry is clear"), and the
# fault addresses are GPU BAR1.
#
# Usage:
#   sudo iommu_identity.sh MODE DEVICE [DEVICE...]
#
#   MODE     identity | DMA-FQ
#   DEVICE   PCI BDF of a NIC function. Devices must be named explicitly:
#            this script has no machine defaults and touches nothing without
#            at least one device.
#
# Example (placeholder BDFs, not this or any other machine):
#   sudo iommu_identity.sh identity 0000:3b:00.0 0000:3b:00.1
#
# Only the named NIC functions are rebound (their rails drop for a few
# seconds). GPU groups are NOT touched. The change is runtime only and reverts
# on reboot. Nothing here installs or enables a boot-time unit;
# afd-gdr-iommu.service.example is an example you must edit and install yourself.
set -euo pipefail

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}
die() { echo "iommu_identity: $*" >&2; exit 1; }

case "${1:-}" in
  -h|--help) usage 0 ;;
esac

[[ $# -ge 2 ]] || usage
MODE=$1
shift
case "$MODE" in
  identity|DMA-FQ) ;;
  *) usage ;;
esac
[[ $EUID -eq 0 ]] || die "must run as root: it writes /sys IOMMU group state"

# Map each named function to its group, and refuse before changing anything if
# a group contains a device this script does not expect.
declare -A GROUP=()
BDFS=("$@")
for BDF in "${BDFS[@]}"; do
  sys=/sys/bus/pci/devices/$BDF
  [[ -e $sys ]] || die "no such PCI device: $BDF"
  grp=$(readlink -f "$sys/iommu_group" 2>/dev/null || true)
  [[ -n "$grp" && -d "$grp" ]] || die "$BDF has no IOMMU group (is IOMMU enabled?)"
  GROUP[$BDF]=$grp
done

mapfile -t groups < <(printf '%s\n' "${GROUP[@]}" | sort -u)
for grp in "${groups[@]}"; do
  for p in "$grp"/devices/*; do
    d=$(basename "$p")
    drv=none
    [[ -e "$p/driver" ]] && drv=$(basename "$(readlink -f "$p/driver")")
    case "$drv" in
      bnxt_en|none) ;;
      *) die "unexpected driver $drv for $d in group $(basename "$grp"); abort" ;;
    esac
  done
done

set_group() {
  local grp=$1 devs=() drivers=()
  for p in "$grp"/devices/*; do
    local d; d=$(basename "$p"); devs+=("$d")
    if [[ -e "$p/driver" ]]; then
      drivers+=("$(basename "$(readlink -f "$p/driver")")")
    else
      drivers+=(none)
    fi
  done
  echo "group $(basename "$grp"): ${devs[*]} drivers ${drivers[*]} type $(cat "$grp/type") -> $MODE"
  for i in "${!devs[@]}"; do
    [[ "${drivers[$i]}" != none ]] &&
      echo "${devs[$i]}" > "/sys/bus/pci/drivers/${drivers[$i]}/unbind"
  done
  echo "$MODE" > "$grp/type"
  for i in "${!devs[@]}"; do
    [[ "${drivers[$i]}" != none ]] &&
      echo "${devs[$i]}" > "/sys/bus/pci/drivers/${drivers[$i]}/bind"
  done
  echo "  now $(cat "$grp/type")"
}

for grp in "${groups[@]}"; do
  set_group "$grp"
done

sleep 5   # NetworkManager re-applies the static rail addresses when the netdevs return
for BDF in "${BDFS[@]}"; do
  for net in "/sys/bus/pci/devices/$BDF"/net/*; do
    [[ -e $net ]] && ip -br addr show "$(basename "$net")"
  done
done
rdma link show
