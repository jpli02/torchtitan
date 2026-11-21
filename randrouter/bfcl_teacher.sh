#!/bin/bash
# BFCL teacher data for gate training: full-depth answers of the released model to 600 live items (randrouter/bfcl_teacher.py),
# one shard on each of cards 8 and 9, then the conversion and the cached states for the router training (nas_feats.py).
set -u
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
PY=/home/jli199/boptim-agent/.venv/bin/python
OUT=/home/jli199/boptim_scratch/randrouter/bfcl_teacher
export PYTORCH_ALLOC_CONF=expandable_segments:True
$PY -I randrouter/bfcl_teacher.py prep --n 600
for k in 0 1; do
  card=$((8 + k))
  n=$(wc -l < $OUT/live_short_$k.jsonl)
  CUDA_VISIBLE_DEVICES=$card $PY depth_eval.py /home/jli199/boptim_scratch/ouro_base_kvshare $OUT/live_short_$k.jsonl --task bfcl --steps 4 \
    --n "$n" --seed_offset 0 --batch 1 --max_new 256 --assistant_prefix "[" --dump_ids --out $OUT/gen_live_short_$k.jsonl > $OUT/gen_live_short_$k.log 2>&1 &
done
wait
$PY -I randrouter/bfcl_teacher.py convert
CUDA_VISIBLE_DEVICES=9 $PY randrouter/nas_feats.py --teacher $OUT/teacher_bfcl_live.jsonl --split train --out /home/jli199/boptim_scratch/randrouter/nas_feats_bfcl_live.pt --mem_gb 8.3 > $OUT/nas_feats.log 2>&1
tail -n 5 $OUT/nas_feats.log
echo "BFCL TEACHER done $(date '+%m-%d %H:%M')"
