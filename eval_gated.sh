#!/bin/bash
# Eval a checkpoint through the task_complete verification gate
# (verify_agent:TerminusVerify) instead of stock terminus-2.
#
# A/B against the ungated numbers on the same 12 tasks and same settings:
#   continue10k 5/24, batching5k 4/24.
#
# Env:
#   CKPT   serving dir (sharded/-free)      NAME  model/run name
#   PORT   server port                      GPU   card index (never defaulted)
#   TASKS  space-separated task ids, or empty for the standard 12
#   ATT    attempts per task (default 2)
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
LOGDIR=/home/jli199/terminal_bench_eval/logs
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
: "${GPU:?GPU must be set explicitly}"
: "${CKPT:?CKPT must be set}"
: "${NAME:?NAME must be set}"
PORT=${PORT:-8060}
ATT=${ATT:-2}
OUT=$TMP/${NAME}_results.txt
: > "$OUT"

DEFAULT_TASKS="hello-world fix-permissions extract-safely csv-to-parquet \
processing-pipeline fix-git heterogeneous-dates chess-best-move \
fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
TASKS=${TASKS:-$DEFAULT_TASKS}
TARGS=""
for t in $TASKS; do TARGS="$TARGS -t $t"; done

# pre-flight: the named GPU must actually be free. A silent default once put six
# servers on one card and voided two 80-task runs.
free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
       | tr -d ' ' | awk -F, -v g="$GPU" '$1==g {print $2}')
MINFREE=${MINFREE:-20000}
if [ -z "$free" ] || [ "$free" -lt "$MINFREE" ]; then
  echo "ABORT - gpu $GPU has ${free:-?}MiB free, need $MINFREE" | tee -a "$OUT"; exit 1
fi

# THR = early-exit threshold for the Ouro router. Below 1.0 the server exits the
# UT loop early per token; at/above 1.0 it runs full recurrence (the baseline).
# Empty THR means "don't pass the flag at all", which is also full recurrence.
THRARG=""
[ -n "${THR:-}" ] && THRARG="--early_exit_threshold $THR"
echo "[$NAME] gpu=$GPU free=${free}MiB port=$PORT attempts=$ATT thr=${THR:-none}" \
  | tee -a "$OUT"

slog=$LOGDIR/${NAME}_server.log
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /home/jli199/torchtitan/.venv/bin/python \
  /home/jli199/torchtitan/scripts/ouro_openai_server.py \
  --hf_dir "$CKPT" --port $PORT --model_name $NAME \
  --no_repeat_ngram_size 0 --dtype bfloat16 --attn_impl sdpa \
  $THRARG --force_temperature 0.7 > "$slog" 2>&1 &
spid=$!
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "$NAME: VOID - server never came up" | tee -a "$OUT"
  kill -9 $spid 2>/dev/null; exit 1
fi
echo "server up $(date '+%m-%d %H:%M')" | tee -a "$OUT"

rm -rf /tmp/tb_runs/$NAME
cd /home/jli199/terminal_bench_eval || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
# GATE=0 runs stock terminus-2, which is what every 12-task baseline used
# (continue10k 5/24, batching5k 4/24). Use it whenever the number has to be
# comparable to those; the gate is a separate, additive intervention.
if [ "${GATE:-1}" = "0" ]; then
  AGENT_ARGS="--agent terminus-2"
else
  AGENT_ARGS="--agent-import-path verify_agent:TerminusVerify"
fi
echo "agent: $AGENT_ARGS" | tee -a "$OUT"

PYTHONPATH="$WT" timeout 43200 .venv/bin/tb run \
  --dataset-path /home/jli199/terminal_bench_eval/tb_tasks \
  $AGENT_ARGS \
  --model "openai/$NAME" \
  --agent-kwarg api_base="http://127.0.0.1:$PORT/v1" \
  $TARGS \
  --n-attempts $ATT --n-concurrent 1 --global-agent-timeout-sec 420 \
  --output-path /tmp/tb_runs --run-id "$NAME" --no-livestream \
  >> "$LOGDIR/${NAME}.tb.log" 2>&1

err=$(grep -c "500 Internal Server Error" "$slog" 2>/dev/null)
ok=$(grep -c '"POST /v1/chat/completions HTTP/1.1" 200' "$slog" 2>/dev/null)
tr=$(ls /tmp/tb_runs/$NAME/*/*/results.json 2>/dev/null | wc -l)
kill -9 $spid 2>/dev/null
echo "$NAME: trials=$tr http200=$ok http500=$err" | tee -a "$OUT"
echo "--- done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
