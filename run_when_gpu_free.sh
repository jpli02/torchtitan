#!/bin/bash
# Wait for a card with >= MINFREE MiB free among GPUS, then run the per-token dumps for r1 and ep4 on it.
#   GPUS="3 1 2 4 5" MINFREE=40000 bash run_when_gpu_free.sh
set -u
GPUS=${GPUS:-"3 1 2 4 5"}; MINFREE=${MINFREE:-40000}
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
cd "$WT" || exit 1
END=$(( $(date +%s) + 86400 ))
pick=""
while [ "$(date +%s)" -lt "$END" ]; do
  for g in $GPUS; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "${free:-0}" -ge "$MINFREE" ]; then pick=$g; break; fi
  done
  [ -n "$pick" ] && break
  sleep 300
done
[ -n "$pick" ] || { echo "no free GPU within 24 h"; exit 1; }
echo "token dumps on GPU $pick $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=$pick PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u fit_eval_dump.py /home/jli199/boptim_scratch/ouro_tvary_r1_clean "$TMP/tvary_r1_rendered.jsonl" "$TMP/tokdump_r1.jsonl" new > "$TMP/tokdump_r1.log" 2>&1
CUDA_VISIBLE_DEVICES=$pick PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u fit_eval_dump.py /home/jli199/boptim_scratch/ouro_tvary_r1_4ep_ep4_clean "$TMP/tvary_r1_rendered.jsonl" "$TMP/tokdump_ep4.jsonl" new > "$TMP/tokdump_ep4.log" 2>&1
tail -n 1 "$TMP/tokdump_r1.log" "$TMP/tokdump_ep4.log"
