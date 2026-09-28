#!/bin/bash
# After the batching-dense SFT finishes: (1) re-probe P(batch) -- the 2-minute
# measurement that says whether the intervention did the thing it was designed
# to do -- then (2) eval on the SAME 12 tasks with the SAME settings as
# continue10k / repro11k, so the comparison is apples-to-apples.
#
# The probe gates INTERPRETATION, not spending: the eval runs either way,
# because "P(batch) moved but pass@1 didn't" is itself the informative result,
# and it is the one that would tell us batching was never the binding constraint.
#
# Read the outcome against the measured noise floor: continue10k scored 4/80 and
# its same-recipe reproduction scored 3/80, sharing only 2 solved tasks. On 12
# tasks a swing of +-1 is not signal.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
LOGDIR=/home/jli199/terminal_bench_eval/logs
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
DUMP=/home/jli199/boptim_scratch/ouro_batching_sft
CLEAN=/home/jli199/boptim_scratch/ouro_batching_clean
OUT=$TMP/batching12_results.txt
TRAIN_PID=${TRAIN_PID:-820960}
: > "$OUT"

# 1. wait for training to exit (cap 12h)
for _ in $(seq 1 720); do
  kill -0 "$TRAIN_PID" 2>/dev/null || break
  sleep 60
done
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "ABORT - training still alive after 12h" | tee -a "$OUT"; exit 1
fi
echo "training exited $(date '+%m-%d %H:%M')" | tee -a "$OUT"

# 2. newest checkpoint. Weights are re-inited from c10k so the step counter
#    restarts at 1; the final checkpoint is step-5000, not step-15000.
SRC=$(ls -d "$DUMP"/checkpoint/step-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$SRC" ]; then
  echo "ABORT - no checkpoint under $DUMP/checkpoint" | tee -a "$OUT"; exit 1
fi
echo "checkpoint: $SRC" | tee -a "$OUT"

# 3. stage HF companion files, then a sharded/-free serving copy. That subdir
#    hangs torchtitan's HF loader (67 min at 100% CPU, zero steps).
cd "$WT" || exit 1
/home/jli199/torchtitan/.venv/bin/python scripts/push_ouro_sft_to_hub.py \
  --checkpoint "$SRC" --base_model /home/jli199/torchtitan/assets/hf/Ouro-1.4B-Thinking \
  --repo_id jaslee/tmp --dry_run >/dev/null 2>&1
rm -rf "$CLEAN"; mkdir -p "$CLEAN"
for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
         tokenizer_config.json special_tokens_map.json vocab.json merges.txt \
         model.safetensors.index.json model-00001-of-00001.safetensors; do
  [ -e "$SRC/$f" ] && ln -s "$SRC/$f" "$CLEAN/$f"
done
echo "staged $CLEAN" | tee -a "$OUT"

# 4. a genuinely free GPU. Never default one: a silent GPU:-4 once put six
#    servers on one card and voided two 80-task runs.
gpu=""
for _ in $(seq 1 120); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=30000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
if [ -z "$gpu" ]; then
  echo "ABORT - no GPU with 30GB free after 2h" | tee -a "$OUT"; exit 1
fi
echo "gpu=$gpu" | tee -a "$OUT"

# 5. THE PROBE. Baseline on continue10k was P(start another command)=0.00027
#    against ~0.195 implied by the data.
echo "=== probe P(batch) on batching5k ===" | tee -a "$OUT"
CUDA_VISIBLE_DEVICES=$gpu OURO_PROBE_DIR="$CLEAN" \
  /home/jli199/torchtitan/.venv/bin/python probe_batching2.py 2>&1 \
  | grep -E "P\(|VERDICT|reference" | tee -a "$OUT"

# 6. 12-task eval, settings identical to the c10k / repro11k runs
name=batching5k
port=8044
slog=$LOGDIR/batching12_server.log
CUDA_VISIBLE_DEVICES=$gpu setsid nohup /home/jli199/torchtitan/.venv/bin/python \
  /home/jli199/torchtitan/scripts/ouro_openai_server.py \
  --hf_dir "$CLEAN" --port $port --model_name $name \
  --no_repeat_ngram_size 0 --dtype bfloat16 --attn_impl sdpa \
  --force_temperature 0.7 > "$slog" 2>&1 &
spid=$!
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  echo "$name: VOID - server never came up" | tee -a "$OUT"
  kill -9 $spid 2>/dev/null; exit 1
fi
echo "server up, eval starting $(date '+%m-%d %H:%M')" | tee -a "$OUT"

rm -rf /tmp/tb_runs/clean_$name
cd /home/jli199/terminal_bench_eval || exit 1
set -a; . ~/.boptim_keys.env 2>/dev/null || true; set +a
timeout 43200 .venv/bin/tb run \
  --dataset-path /home/jli199/terminal_bench_eval/tb_tasks \
  --agent terminus-2 --model "openai/$name" \
  --agent-kwarg api_base="http://127.0.0.1:$port/v1" \
  -t hello-world -t fix-permissions -t extract-safely \
  -t csv-to-parquet -t processing-pipeline \
  -t fix-git -t heterogeneous-dates -t chess-best-move \
  -t fibonacci-server -t crack-7z-hash.easy \
  -t cron-broken-network -t sanitize-git-repo \
  --n-attempts 2 --n-concurrent 1 --global-agent-timeout-sec 420 \
  --output-path /tmp/tb_runs --run-id "clean_$name" --no-livestream \
  >> "$LOGDIR/batching12.tb.log" 2>&1

# 7. contamination accounting. LiteLLM retries 3x, so raw 500 counts overstate
#    damage; the real criterion is per-task "0 usable responses".
err=$(grep -c "500 Internal Server Error" "$slog" 2>/dev/null)
ok=$(grep -c '"POST /v1/chat/completions HTTP/1.1" 200' "$slog" 2>/dev/null)
tr=$(ls /tmp/tb_runs/clean_$name/*/*/results.json 2>/dev/null | wc -l)
kill -9 $spid 2>/dev/null
echo "$name: trials=$tr http200=$ok http500=$err" | tee -a "$OUT"
echo "results: /tmp/tb_runs/clean_$name" | tee -a "$OUT"
echo "--- done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
