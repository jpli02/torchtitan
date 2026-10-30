#!/bin/bash
# Oracle per-token depth bound + gate comparison on the K=4 generations of each benchmark.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
GPU=${GPU:-2}
cd "$WT" || exit 1
for t in crux gsm8k math; do
  echo "=== ORACLE $t $(date '+%m-%d %H:%M') ===" | tee -a "$TMP/oracle_depth.txt"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" oracle_depth.py /home/jli199/boptim_scratch/ouro_base_kvshare \
    "$TMP/depth_${t}_k4.jsonl" --task "$t" --out "$TMP/oracle_${t}.json" > "$TMP/oracle_${t}.log" 2>&1
  grep -vE 'Warning|warn|deprecated|^ORACLE|gens,' "$TMP/oracle_${t}.log" | tail -10 | tee -a "$TMP/oracle_depth.txt"
done
echo "=== ORACLE all done $(date '+%m-%d %H:%M') ===" | tee -a "$TMP/oracle_depth.txt"
