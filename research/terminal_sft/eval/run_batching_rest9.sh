#!/bin/bash
# Finish the 9 tasks the batching5k 80-task run never reached before /tmp filled
# and it was killed at 71/80.
#
# Settings are copied EXACTLY from eval80_parallel.sh so these results merge
# with the existing e80_batching80_s0/s1 dirs into one comparable 80-task
# number: stock terminus-2 (UNGATED -- this is the baseline the verification
# gate gets measured against), 1 attempt, temp 0.7, 420s, no ngram guard.
#
# One of the 9 is vim-terminal-task, which continue10k solved -- so the single
# task most likely to move the count is in this remainder.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
LOGDIR=/home/jli199/terminal_bench_eval/logs
CLEAN=/home/jli199/boptim_scratch/ouro_batching_clean
OUT=$TMP/batching_rest9.txt
: "${GPU:?GPU must be set explicitly}"
PORT=${PORT:-8090}
NAME=batching80_s9
: > "$OUT"

free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | tr -d ' ' | awk -F, -v g="$GPU" '$1==g {print $2}')
if [ -z "$free" ] || [ "$free" -lt "${MINFREE:-15000}" ]; then
  echo "ABORT - gpu $GPU has ${free:-?}MiB free" | tee -a "$OUT"; exit 1
fi
echo "[rest9] gpu=$GPU free=${free}MiB port=$PORT $(date '+%m-%d %H:%M')" | tee -a "$OUT"

slog=$LOGDIR/batching_rest9_server.log
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /home/jli199/torchtitan/.venv/bin/python \
  /home/jli199/torchtitan/scripts/ouro_openai_server.py \
  --hf_dir "$CLEAN" --port $PORT --model_name $NAME \
  --no_repeat_ngram_size 0 --dtype bfloat16 --attn_impl sdpa \
  --force_temperature 0.7 > "$slog" 2>&1 &
spid=$!
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "VOID - server never came up" | tee -a "$OUT"; kill -9 $spid 2>/dev/null; exit 1
fi
echo "server up" | tee -a "$OUT"

rm -rf /tmp/tb_runs/e80_batching80_s9
cd /home/jli199/terminal_bench_eval || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
timeout 21600 .venv/bin/tb run \
  --dataset-path /home/jli199/terminal_bench_eval/tb_tasks \
  --agent terminus-2 --model "openai/$NAME" \
  --agent-kwarg api_base="http://127.0.0.1:$PORT/v1" \
  -t decommissioning-service-with-sensitive-data \
  -t extract-safely \
  -t git-workflow-hack \
  -t nginx-request-logging \
  -t openssl-selfsigned-cert \
  -t path-tracing-reverse \
  -t simple-sheets-put \
  -t tmux-advanced-workflow \
  -t vim-terminal-task \
  --n-attempts 1 --n-concurrent 1 --global-agent-timeout-sec 420 \
  --output-path /tmp/tb_runs --run-id "e80_batching80_s9" --no-livestream \
  >> "$LOGDIR/batching_rest9.tb.log" 2>&1

err=$(grep -c "500 Internal Server Error" "$slog" 2>/dev/null)
tr=$(ls /tmp/tb_runs/e80_batching80_s9/*/*/results.json 2>/dev/null | wc -l)
kill -9 $spid 2>/dev/null
echo "trials=$tr http500=$err" | tee -a "$OUT"
echo "--- done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
