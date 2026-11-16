#!/bin/bash
# Learning-rate debug: teacher-forced validation loss of checkpoints on the held-out groups of lrval_rendered.jsonl
# (lr_debug_valset.py). Runs PAR checkpoints at a time on one card (each needs about 10 GB).
#   GPU=1 PAR=4 bash lr_debug_fit.sh name=dir [name=dir ...]      results: <job tmp>/lrval_<name>.json
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
GPU=${GPU:-1}
PAR=${PAR:-4}
cd "$WT" || exit 1
n=0
for pair in "$@"; do
  name=${pair%%=*}; dir=${pair#*=}
  if [ -s "$TMP/lrval_${name}.json" ]; then echo "lrval $name: already done"; continue; fi
  if ! ls "$dir"/model*.safetensors > /dev/null 2>&1; then echo "lrval $name: no weights in $dir, skipped"; continue; fi
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u fit_eval.py "$dir" "$TMP/lrval_rendered.jsonl" \
    "$TMP/lrval_${name}.json" > "$TMP/lrval_${name}.log" 2>&1 &
  n=$((n + 1))
  if [ $((n % PAR)) -eq 0 ]; then wait; fi
done
wait
echo "lrval batch done $(date '+%m-%d %H:%M')"
"$PY" -I lr_debug_fit_report.py
