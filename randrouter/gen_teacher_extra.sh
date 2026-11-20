#!/bin/bash
# Extra single-stream teacher-generation processes on cards that are idle but mostly full (another user's inference server
# holds the memory and computes nothing): one batch-1 process per card, memory-capped so it can never grow into the
# neighbour, working from the END of the problem list towards the main process (gen_teacher.py on GPU 0).
#   bash randrouter/gen_teacher_extra.sh 8 9          (cards; shard i/k over the listed cards)
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
R=/home/jli199/boptim_scratch/randrouter
PY=/home/jli199/boptim-agent/.venv/bin/python
cards=("$@")
k=${#cards[@]}
also=$R/teacher_lcb_v1v4.jsonl
for g in "${cards[@]}"; do also=$also,$R/teacher_lcb_v1v4.g$g.jsonl; done
cd "$WT" || exit 1
i=0
for g in "${cards[@]}"; do
  CUDA_VISIBLE_DEVICES=$g PYTORCH_ALLOC_CONF=expandable_segments:True setsid nohup "$PY" -u randrouter/gen_teacher.py \
    --batch 1 --reverse --shard "$i/$k" --mem_gb "${MEM_GB:-9.3}" --out "$R/teacher_lcb_v1v4.g$g.jsonl" --also "$also" \
    > "$R/gen_teacher.g$g.log" 2>&1 < /dev/null &
  i=$((i + 1))
done
echo "started $k extra generation processes on cards ${cards[*]}"
