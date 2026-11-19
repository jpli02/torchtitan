#!/bin/bash
# Terminal-Bench TEST evaluation of the from-base checkpoint as soon as a card is free. The run's own launcher gave up at
# 06:29 on 11-19 ("gpu 1 has 36775MiB free, need 44000"): another user's job took GPU 1 the minute the training ended.
# The server needs a whole card for 12k-token contexts (a half-filled card gives out-of-memory turns and a score that
# cannot be compared), so this waits, polling every 20 s, for ANY of the cards in CARDS to have 47 GB free and starts there.
# Start detached:  setsid nohup bash frombase_eval_when_free.sh >> $TMP/frombase_eval_when_free.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
TAG=${TAG:-frombase_v3r1_p1}
CKPT=${CKPT:-$S/ouro_${TAG}_clean}
NAME=${NAME:-${TAG}_b_eval}
CARDS=${CARDS:-"1 0 3 4 5"}
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
cd "$WT" || exit 1
ls "$CKPT"/model*.safetensors > /dev/null 2>&1 || { echo "no weights in $CKPT"; exit 1; }
END=$(( $(date +%s) + 3600 * ${WAIT_H:-48} ))
gpu=""
while [ -z "$gpu" ]; do
  for g in $CARDS; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | head -n 1)
    if [ "${free:-0}" -ge 47000 ]; then gpu=$g; break; fi
  done
  [ -n "$gpu" ] && break
  [ "$(date +%s)" -gt "$END" ] && { echo "EVAL-WHEN-FREE gave up: no free card $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 20
done
echo "card $gpu is free; Terminal-Bench test evaluation $NAME starts $(date '+%m-%d %H:%M')"
export OURO_SEQ_CAP=12000
GPU=$gpu GATE=0 CKPT="$CKPT" NAME=$NAME PORT=8231 ATT=2 MINFREE=44000 OUTROOT=/srv/scratch/jli199/tb_runs TASKS="$ALL11" \
  bash eval_gated.sh > "$TMP/${NAME}.log" 2>&1
python3 -I tb_test_summary.py "$NAME" /srv/scratch/jli199/tb_runs
echo "server out-of-memory turns: $(grep -a -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/${NAME}_server.log 2>/dev/null)"
tail -n 2 "$TMP/${NAME}.log" | cut -c1-160
echo "EVAL-WHEN-FREE $NAME done $(date '+%m-%d %H:%M')"
