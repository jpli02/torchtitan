#!/bin/bash
# Learning-rate debug on GPU 1, part 7 (replaces part 6, whose three evaluations failed at once: the run directories were on
# /home/jli199/boptim_scratch, an NFS mount the Docker daemon cannot create bind-mount paths on, "mkdir ...: permission
# denied", so every trial ended as unknown_agent_error before the agent started). Run directories now go to the LOCAL
# scratch disk /srv/scratch/jli199/tb_runs (where the 75-task evaluations ran; 68 GB free), under new run ids:
#   1. the 5e-6 recipe checkpoint   2. repeat of take-turns r1   3. repeat of v3
# GPU 1 must be free of other users' jobs first (the server needs the whole card for 12k-token contexts): waits up to
# WAIT_MIN minutes for that and stops with a message otherwise.
# Start detached:  setsid nohup bash lr_debug_chain7.sh >> $TMP/lr_debug_chain7.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
OUT=/srv/scratch/jli199/tb_runs
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
CANDS=${CANDS:-"lr5e-6_cos804_c=$S/ouro_lrsweep_lr5e-6_cos804_clean r1_rep2b=$S/ouro_tvary_r1_clean v3_rep2b=$S/ouro_tb80sft26b_v3_clean"}
cd "$WT" || exit 1
free1() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1; }
export OURO_SEQ_CAP=12000
port=8211
for pair in $CANDS; do
  tag=${pair%%=*}; dir=${pair#*=}
  ls "$dir"/model*.safetensors > /dev/null 2>&1 || { echo "no weights in $dir, skipped"; continue; }
  END=$(( $(date +%s) + 60 * ${WAIT_MIN:-45} ))
  while pgrep -u "$(id -un)" -f 'eval_gated.sh|ouro_openai_server.py|tb run --dataset-path|torchtitan.train' > /dev/null || [ "$(free1)" -lt 47500 ]; do
    if [ "$(date +%s)" -gt "$END" ]; then
      echo "LR-DEBUG part 7 stopped before $tag: gpu 1 has only $(free1) MiB free (another user's job?) $(date '+%m-%d %H:%M')"
      exit 1
    fi
    sleep 30
  done
  left=$(docker ps -a --format '{{.Names}}' | grep -- "-of-2-${tag}_eval\$" | tr '\n' ' ')
  [ -n "$left" ] && { echo "left-over containers for ${tag}_eval ($left): this run id is skipped"; continue; }
  echo "TB test evaluation of $tag starts $(date '+%m-%d %H:%M') (run directory $OUT/${tag}_eval)"
  GPU=1 GATE=0 CKPT="$dir" NAME=${tag}_eval PORT=$port ATT=2 MINFREE=${MINFREE:-44000} OUTROOT=$OUT TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${tag}_eval.log" 2>&1
  python3 -I tb_test_summary.py "${tag}_eval" "$OUT"
  echo "server out-of-memory turns: $(grep -a -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/${tag}_eval_server.log 2>/dev/null)"
  echo "LR-DEBUG TB evaluation of $tag done $(date '+%m-%d %H:%M')"
  port=$((port + 1))
done
echo "LR-DEBUG part 7 done $(date '+%m-%d %H:%M')"
