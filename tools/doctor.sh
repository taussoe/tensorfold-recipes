#!/usr/bin/env bash
# Check both DGX Sparks before the first recipe: GPU, Docker, memory, disk, SSH, the 200 Gb/s link and RDMA.
# Run on Spark 1:   ./tools/doctor.sh
# Every check prints ✓ or ✗ with what to do; nothing is changed.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"
RECIPE_NAME=doctor
load_cluster
fails=0
bad() { printf '%s  ✗%s %s\n' "$C_R" "$C_0" "$*"; fails=$((fails + 1)); }
good() { printf '%s  ✓%s %s\n' "$C_G" "$C_0" "$*"; }
note() { printf '%s  !%s %s\n' "$C_Y" "$C_0" "$*"; }

check_node() {
  local n="$1" own_ip="$2" peer_ip="$3" out
  echo; echo "Spark $n"
  if [ "$n" = 2 ] && ! run_on 2 true 2>/dev/null; then
    bad "ssh $SPARK2_SSH failed. Set up keys: ssh-copy-id $SPARK2_SSH (docs/01-hardware-setup.md#ssh)"; return
  fi
  out=$(run_on "$n" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1) \
    && good "GPU: $out" || bad "nvidia-smi failed: $out"
  run_on "$n" docker info >/dev/null 2>&1 && good "docker runs without sudo" \
    || bad "docker needs sudo or is not running: sudo usermod -aG docker \$USER, then log in again"
  run_on "$n" docker info 2>/dev/null | grep -qi nvidia && good "NVIDIA container runtime present" \
    || bad "no NVIDIA container runtime in docker info (DGX OS ships it; reinstall nvidia-container-toolkit)"
  out=$(run_on "$n" awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
  [ "${out:-0}" -ge 100 ] && good "memory available: ${out} GB of 128 (unified CPU+GPU; read it with free -h, not nvidia-smi)" \
    || note "only ${out} GB available: stop other workloads before serving a large model"
  out=$(run_on "$n" bash -c "mkdir -p '$HF_CACHE' && df -BG --output=avail '$HF_CACHE' | tail -1 | tr -dc 0-9")
  if [ "${out:-0}" -ge 400 ]; then good "disk free at HF_CACHE: ${out} GB"
  else note "disk free at HF_CACHE: ${out} GB (GLM halves 91-182, Flash Next 113, 27B 20 GB)"; fi
  out=$(run_on "$n" docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
  [ -z "$out" ] && good "no containers running" || note "running containers: $out"
  # The link
  out=$(run_on "$n" ip -o -4 addr show dev "$NCCL_SOCKET_IFNAME" 2>&1)
  echo "$out" | grep -q " $own_ip/" && good "$NCCL_SOCKET_IFNAME has $own_ip" \
    || bad "$NCCL_SOCKET_IFNAME does not have $own_ip ($(echo "$out" | awk '{print $4}')). docs/01-hardware-setup.md#the-link"
  out=$(run_on "$n" cat "/sys/class/net/$NCCL_SOCKET_IFNAME/mtu" 2>/dev/null)
  [ "${out:-0}" = 9000 ] && good "MTU 9000" || note "MTU ${out:-?} on the link (9000 is the usual choice for RoCE; see the hardware doc)"
  out=$(run_on "$n" cat "/sys/class/net/$NCCL_SOCKET_IFNAME/speed" 2>/dev/null)
  [ "${out:-0}" -ge 100000 ] && good "link speed ${out} Mb/s" || bad "link speed ${out:-unknown} Mb/s: check the QSFP cable"
  run_on "$n" ping -c 3 -W 1 -I "$NCCL_SOCKET_IFNAME" "$peer_ip" >/dev/null 2>&1 && good "ping $peer_ip over the link" \
    || bad "cannot ping $peer_ip over $NCCL_SOCKET_IFNAME"
  if run_on "$n" command -v ibdev2netdev >/dev/null 2>&1; then
    out=$(run_on "$n" ibdev2netdev 2>/dev/null)
    local h
    for h in $(echo "$NCCL_IB_HCA" | tr ',' ' '); do
      echo "$out" | grep -q "^$h .*(Up)" && good "RDMA adapter $h is Up" \
        || bad "RDMA adapter $h is not Up. ibdev2netdev says:$(echo; echo "$out" | sed 's/^/        /')"
    done
  else note "ibdev2netdev not found (it comes with DGX OS / MLNX OFED): cannot check NCCL_IB_HCA"; fi
  [ "$(run_on "$n" ls /dev/infiniband 2>/dev/null | wc -l)" -gt 0 ] && good "/dev/infiniband present" || bad "/dev/infiniband missing"
}

echo "TensorFold recipes doctor — config/cluster.env: link $SPARK1_LINK_IP <-> $SPARK2_LINK_IP on $NCCL_SOCKET_IFNAME"
check_node 1 "$SPARK1_LINK_IP" "$SPARK2_LINK_IP"
check_node 2 "$SPARK2_LINK_IP" "$SPARK1_LINK_IP"
if [ "$WEIGHTS_SYNC" = rsync ]; then
  echo; echo "Weights sync"
  u=$(ssh -T -G "$SPARK2_SSH" | awk '/^user /{print $2}')
  ssh -T -o BatchMode=yes -o ConnectTimeout=5 "$u@$SPARK2_LINK_IP" true 2>/dev/null \
    && good "ssh $u@$SPARK2_LINK_IP works (rsync goes over the 200 Gb/s link)" \
    || bad "ssh $u@$SPARK2_LINK_IP failed: rsync needs SSH over the link address too (or set WEIGHTS_SYNC=pull)"
fi
echo
if [ "$fails" = 0 ]; then ok "all checks passed"; else die "$fails check(s) failed"; fi
