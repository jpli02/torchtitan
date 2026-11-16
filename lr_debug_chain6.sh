#!/bin/bash
# Learning-rate debug on GPU 1, part 6 (replaces the rest of part 5). The first Terminal-Bench evaluation of the 5e-6 recipe
# checkpoint died after 2 trials: one trial (sanitize-git-repo) printed 4.5 GB to the terminal recording and filled the 8 GB
# /tmp, so the harness could not create the next trial directory. From here on run directories go to boptim_scratch
# (18 TB free). After the repeat evaluation of the half-step checkpoint that is running now (on /tmp, 4.6 GB free again):
#   1. the 5e-6 recipe checkpoint (new run id)   2. repeat of take-turns r1   3. repeat of v3
# Start detached:  setsid nohup bash lr_debug_chain6.sh >> $TMP/lr_debug_chain6.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
OUT=$S/tb_runs
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
CANDS=${CANDS:-"lr5e-6_cos804_b=$S/ouro_lrsweep_lr5e-6_cos804_clean r1_rep2=$S/ouro_tvary_r1_clean v3_rep2=$S/ouro_tb80sft26b_v3_clean"}
cd "$WT" || exit 1
mkdir -p "$OUT"
END=$(( $(date +%s) + 86400 ))
while pgrep -u "$(id -un)" -f 'eval_gated.sh|ouro_openai_server.py|tb run --dataset-path|torchtitan.train' > /dev/null \
   || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -lt 44000 ]; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG part 6 gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 60
done
python3 -I tb_test_summary.py lerp_r1_a050_rep2_eval
echo "LR-DEBUG TB evaluation of lerp_r1_a050_rep2 done $(date '+%m-%d %H:%M')"
export OURO_SEQ_CAP=12000
port=8201
for pair in $CANDS; do
  tag=${pair%%=*}; dir=${pair#*=}
  ls "$dir"/model*.safetensors > /dev/null 2>&1 || { echo "no weights in $dir, skipped"; continue; }
  left=$(docker ps -a --format '{{.Names}}' | grep -- "-of-2-${tag}_eval\$" | tr '\n' ' ')
  [ -n "$left" ] && { echo "left-over containers for ${tag}_eval ($left): this run id is skipped"; continue; }
  echo "TB test evaluation of $tag starts $(date '+%m-%d %H:%M') (run directory $OUT/${tag}_eval)"
  GPU=1 GATE=0 CKPT="$dir" NAME=${tag}_eval PORT=$port ATT=2 MINFREE=${MINFREE:-38000} OUTROOT=$OUT TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${tag}_eval.log" 2>&1
  python3 -I tb_test_summary.py "${tag}_eval" "$OUT"
  echo "LR-DEBUG TB evaluation of $tag done $(date '+%m-%d %H:%M')"
  port=$((port + 1))
done
echo "LR-DEBUG part 6 done $(date '+%m-%d %H:%M')"
