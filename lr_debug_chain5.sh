#!/bin/bash
# Learning-rate debug on GPU 1, part 5 (after part 4 has finished): REPEAT Terminal-Bench test evaluations, a second set of
# 22 trials each under new run ids (a run id is never reused: left-over containers of an id are picked up again), so that
# the comparison of the half-step checkpoint with r1 and v3 rests on 44 trials per checkpoint instead of 22:
#   1. halfway between v3 and r1 (first set: 11/22)   2. take-turns r1 (first set: 9/22)   3. v3 (first set: 8/22)
# Start detached:  setsid nohup bash lr_debug_chain5.sh >> $TMP/lr_debug_chain5.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
CANDS=${CANDS:-"lerp_r1_a050_rep2=$S/ouro_lerp_r1_a050_clean r1_rep2=$S/ouro_tvary_r1_clean v3_rep2=$S/ouro_tb80sft26b_v3_clean"}
cd "$WT" || exit 1
END=$(( $(date +%s) + 86400 ))
until grep -q 'LR-DEBUG part 4 done\|LR-DEBUG part 4 gave up' "$TMP/lr_debug_chain4.log" 2>/dev/null; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG part 5 gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 60
done
while pgrep -u "$(id -un)" -f 'eval_gated.sh|ouro_openai_server.py|tb run --dataset-path|torchtitan.train' > /dev/null \
   || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -lt 44000 ]; do sleep 60; done
export OURO_SEQ_CAP=12000
port=8191
for pair in $CANDS; do
  tag=${pair%%=*}; dir=${pair#*=}
  ls "$dir"/model*.safetensors > /dev/null 2>&1 || { echo "no weights in $dir, skipped"; continue; }
  left=$(docker ps -a --format '{{.Names}}' | grep -- "-of-2-${tag}_eval\$" | tr '\n' ' ')
  [ -n "$left" ] && { echo "left-over containers for ${tag}_eval ($left): this run id is skipped"; continue; }
  echo "TB test evaluation of $tag starts $(date '+%m-%d %H:%M')"
  GPU=1 GATE=0 CKPT="$dir" NAME=${tag}_eval PORT=$port ATT=2 MINFREE=${MINFREE:-38000} TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${tag}_eval.log" 2>&1
  python3 -I tb_test_summary.py "${tag}_eval"
  echo "LR-DEBUG TB evaluation of $tag done $(date '+%m-%d %H:%M')"
  port=$((port + 1))
done
echo "LR-DEBUG part 5 done $(date '+%m-%d %H:%M')"
