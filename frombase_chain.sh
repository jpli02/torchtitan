#!/bin/bash
# Start the from-base run (train_frombase.sh) on GPU 1 as soon as the Terminal-Bench evaluation running there has ended and
# the card is free of other users' jobs (the run needs 36 GB for 41 hours, and the evaluation after it the whole card).
# Waits up to WAIT_H hours for a free card, then stops with a message instead of squeezing in beside someone else's job.
# Start detached:  setsid nohup bash frombase_chain.sh >> $TMP/frombase_chain.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
END=$(( $(date +%s) + 3600 * ${WAIT_H:-12} ))
while pgrep -u "$(id -un)" -f 'eval_gated.sh|ouro_openai_server.py|tb run --dataset-path|torchtitan.train' > /dev/null \
   || [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -lt 47000 ]; do
  if [ "$(date +%s)" -gt "$END" ]; then
    echo "FROM-BASE chain stopped: gpu 1 not free ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1) MiB) $(date '+%m-%d %H:%M')"
    exit 1
  fi
  sleep 60
done
python3 -I tb_test_summary.py lr5e-6_cos804_c_eval /srv/scratch/jli199/tb_runs
echo "FROM-BASE chain: gpu 1 free, training starts $(date '+%m-%d %H:%M')"
GPU=1 bash train_frombase.sh
echo "FROM-BASE chain done $(date '+%m-%d %H:%M')"
