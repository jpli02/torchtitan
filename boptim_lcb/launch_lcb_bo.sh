#!/bin/bash
# 1) per-token correctness dumps for r1 and ep4 on GPU 0 (the earlier attempt OOMed on a card with 25 GB free)
# 2) then the LiveCodeBench router search (gp_hedge over per-step exit thresholds) on the GPUs in LCB_BO_GPUS
#   LCB_BO_GPUS=0,3 bash boptim_lcb/launch_lcb_bo.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
if [ ! -s "$TMP/tokdump_ep4.jsonl" ]; then
  GPUS="${DUMP_GPUS:-0}" MINFREE="${MINFREE:-36000}" bash run_when_gpu_free.sh
  echo "token dumps done $(date '+%m-%d %H:%M')"
fi
cd /home/jli199/boptim-agent || exit 1
echo "BO start $(date '+%m-%d %H:%M') gpus ${LCB_BO_GPUS:-0,3}"
LCB_BO_GPUS="${LCB_BO_GPUS:-0,3}" ./.venv/bin/python -u run_experiment.py --config configs/ouro_lcb_router_gp_local.yaml --log-prefix lcb_router_gp > "$TMP/lcb_bo_gp.log" 2>&1
echo "BO done $(date '+%m-%d %H:%M')"
grep -c lcb-router "$TMP/lcb_bo_gp.log"
tail -n 5 "$TMP/lcb_bo_gp.log"
