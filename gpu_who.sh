#!/bin/bash
# Which GPUs are free, which hold my jobs, which belong to other users.
echo "now $(date '+%m-%d %H:%M')"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "  gpu %s: used %5d MiB  free %5d MiB  util %3d%%\n", $1, $2, $3, $4}'
echo "--- processes (excluding the MPS server):"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader > /tmp/_gpu_uuids_$$.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -v '30 MiB' | while IFS=, read -r u pid mem; do
  u=$(echo "$u" | tr -d ' '); pid=$(echo "$pid" | tr -d ' ')
  idx=$(grep "$u" /tmp/_gpu_uuids_$$.txt | cut -d, -f1)
  info=$(ps -o user=,etime=,args= -p "$pid" 2>/dev/null | cut -c1-110)
  user=$(echo "$info" | awk '{print $1}')
  tag="other"; [ "$user" = "$USER" ] && tag="MINE "
  echo "  gpu $idx: $mem $tag $info"
done
rm -f /tmp/_gpu_uuids_$$.txt
echo "--- my waiters/chains: $(pgrep -af 'rerun_guard_clean|bench_resume|fit_4ep|gate_sweep|bench_pipeline|eval_guard_easy|eval_gated' | grep -oE 'bash [a-z_0-9]+\.sh' | sort | uniq -c | tr '\n' ';')"
