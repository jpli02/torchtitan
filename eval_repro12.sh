#!/bin/bash
# Eval repro_c10k on the SAME 12 tasks continue10k was scored on, with the same
# settings (2 attempts, temp 0.7, no ngram guard, 420s), so the comparison is
# apples-to-apples. Single model, own GPU, verified headroom.
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
LOGDIR=/home/jli199/terminal_bench_eval/logs
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
SRC=/home/jli199/boptim_scratch/ouro_repro_c10k/checkpoint/step-11000
CLEAN=/home/jli199/boptim_scratch/ouro_repro_clean
OUT=$TMP/repro12_results.txt
: > "$OUT"

# Stage HF companion files, then a sharded/-free serving copy (that subdir
# hangs torchtitan's HF loader).
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

gpu=""
for _ in $(seq 1 120); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, '$2+0>=30000 {print $1; exit}')
  [ -n "$gpu" ] && break
  sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no free GPU after 2h" | tee -a "$OUT"; exit 1; }

name=repro11k
port=8040
slog=$LOGDIR/repro12_server.log
echo "[$name] gpu=$gpu" | tee -a "$OUT"

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
  kill -9 $spid 2>/dev/null
  exit 1
fi

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
  >> "$LOGDIR/repro12.tb.log" 2>&1

err=$(grep -c "500 Internal Server Error" "$slog" 2>/dev/null)
ok=$(grep -c '"POST /v1/chat/completions HTTP/1.1" 200' "$slog" 2>/dev/null)
tr=$(ls /tmp/tb_runs/clean_$name/*/*/results.json 2>/dev/null | wc -l)
kill -9 $spid 2>/dev/null
echo "$name: trials=$tr http200=$ok http500=$err" | tee -a "$OUT"
echo "--- done ---" | tee -a "$OUT"
