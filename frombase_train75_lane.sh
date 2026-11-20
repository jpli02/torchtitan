#!/bin/bash
# Training-set pass rate of the from-base checkpoint: the 75 variant tasks the take-turns r1 data came from, one attempt
# each, 1,800 s, the protocol of the per-checkpoint table (epoch_train75_chain2.sh), so that the from-base run gets the
# same two numbers as every other row of that table (training set: r1 31 %, epoch 4 44 %, epoch 8 61 %).
# Waits for a card with CARD_MIB free (polling every 30 s), then runs for about 25 hours; run directory on /srv/scratch.
# Start detached:  setsid nohup bash frombase_train75_lane.sh >> $TMP/frombase_train75_lane.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
POOL=$S/tb_variants/pool
OUT=/srv/scratch/jli199/tb_runs
CKPT=${CKPT:-$S/ouro_frombase_v3r1_p1_clean}
NAME=${NAME:-train75_frombase_1800}
CARDS=${CARDS:-"1 2 3 4 5"}
cd "$WT" || exit 1
TASKS75=$(cat "$TMP/train_tasks_r1.txt")
[ -n "$TASKS75" ] && ls "$CKPT"/model*.safetensors > /dev/null 2>&1 && [ -d "$POOL" ] || { echo "TRAIN75 lane: missing task list, weights or task pool"; exit 1; }
left=$(docker ps -a --format '{{.Names}}' | grep -- "-1-of-1-${NAME}\$" | head -n 3 | tr '\n' ' ')
[ -n "$left" ] && { echo "TRAIN75 lane: left-over containers of $NAME ($left): choose another run name"; exit 1; }
END=$(( $(date +%s) + 3600 * ${WAIT_H:-72} ))
gpu=""
while [ -z "$gpu" ]; do
  for g in $CARDS; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | head -n 1)
    if [ "${free:-0}" -ge "${CARD_MIB:-44500}" ]; then gpu=$g; break; fi
  done
  [ -n "$gpu" ] && break
  [ "$(date +%s)" -gt "$END" ] && { echo "TRAIN75 lane gave up: no free card $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 30
done
echo "card $gpu is free; $NAME starts $(date '+%m-%d %H:%M')"
export OURO_SEQ_CAP=12000
GPU=$gpu GATE=0 MAXTOK=2048 TBTIMEOUT=200000 CKPT="$CKPT" NAME=$NAME PORT=8241 ATT=1 MINFREE=${MINFREE:-42000} \
  DATASET=$POOL OUTROOT=$OUT AGENT_TIMEOUT=1800 TASKS="$TASKS75" bash eval_gated.sh > "$TMP/$NAME.log" 2>&1
python3 -I tb_test_summary.py "$NAME" "$OUT" | head -n 1
echo "server out-of-memory turns: $(grep -a -c 'OOM at step' /home/jli199/terminal_bench_eval/logs/${NAME}_server.log 2>/dev/null)"
tail -n 2 "$TMP/$NAME.log" | cut -c1-160
echo "TRAIN75 lane $NAME done $(date '+%m-%d %H:%M')"
