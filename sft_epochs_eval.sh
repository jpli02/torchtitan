#!/bin/bash
# Epoch-wise SFT + eval: train Ouro-2.6B-Thinking (HF base) on one JSONL for
# EPOCHS epochs, save a checkpoint at every epoch boundary, stage each one as an
# HF serving dir, and evaluate each on the 12-task Terminal-Bench subset
# (stock terminus-2, 2 attempts = the n/24 pass@1 protocol) as soon as it lands.
#
# An "epoch" is total_tokens / (seq_len + 1) optimizer steps (batch 1, packed
# sequences): the loader re-loops the file when exhausted. SPE is computed by
# tokenising the file once; override with SPE=... to skip that.
#
#   TAG=ds_flash_ep JSONL=.../deepseek_flash_generated.jsonl EPOCHS=4 bash sft_epochs_eval.sh
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/torchtitan/.venv/bin/python
TAG=${TAG:-ds_flash_ep}
JSONL=${JSONL:-/home/jli199/boptim_scratch/gen_out/deepseek_flash_generated.jsonl}
EPOCHS=${EPOCHS:-4}
SEQLEN=${SEQLEN:-4096}
MINFREE=${MINFREE:-38000}
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
DUMP=/home/jli199/boptim_scratch/ouro_${TAG}
OUT=$TMP/${TAG}_chain.txt
TRAINLOG=/home/jli199/terminal_bench_eval/logs/${TAG}_train.log
: > "$OUT"
cd "$WT" || exit 1
[ -s "$JSONL" ] || { echo "ABORT - no data at $JSONL" | tee -a "$OUT"; exit 1; }

if [ -z "${SPE:-}" ]; then
  SPE=$("$PY" - "$JSONL" "$BASE" "$SEQLEN" <<'PY2' 2>/dev/null | tail -1
import json, sys, math
from transformers import AutoTokenizer
jsonl, base, seqlen = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
tot = 0
for l in open(jsonl):
    m = json.loads(l)["messages"]
    if len(m) > 64: m = m[:1] + m[-63:]          # loader keeps head + tail
    tot += len(tok(tok.apply_chat_template(m, tokenize=False), add_special_tokens=False).input_ids) + 1
print(math.ceil(tot / (seqlen + 1)))
PY2
)
fi
[ -n "$SPE" ] && [ "$SPE" -gt 0 ] || { echo "ABORT - could not compute steps/epoch" | tee -a "$OUT"; exit 1; }
STEPS=$((SPE * EPOCHS))
echo "[$TAG] SFT from Ouro-2.6B-Thinking HF base on $(wc -l < "$JSONL") rows: $EPOCHS epochs x $SPE steps = $STEPS steps, seq $SEQLEN $(date '+%m-%d %H:%M')" | tee -a "$OUT"

gpu="${GPU:-}"
[ -z "$gpu" ] && for _ in $(seq 1 1440); do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, -v m="$MINFREE" '$2+0>=m {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
  [ -n "$gpu" ] && break; sleep 60
done
[ -n "$gpu" ] || { echo "ABORT - no GPU with ${MINFREE}MiB" | tee -a "$OUT"; exit 1; }
echo "[$TAG] train gpu $gpu $(date '+%m-%d %H:%M')" | tee -a "$OUT"

export OURO_SFT_MIX=tb80sft OURO_SFT_LOCAL_JSONL=$JSONL
unset OURO_INIT_FROM OURO_INIT_DCP OURO_SFT_REQUIRE_COMPLETE OURO_SFT_MIN_CMDS
export WANDB_MODE=disabled PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$DUMP"
CUDA_VISIBLE_DEVICES=$gpu /home/jli199/torchtitan/.venv/bin/torchrun \
  --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 --role rank --tee 1 --log-dir "$DUMP/torchrun_logs" -m torchtitan.train \
  --module ouro --config ouro_2_6b_thinking_terminal_sft \
  --parallelism.data_parallel_shard_degree 1 \
  --dump_folder "$DUMP" --checkpoint.folder "$DUMP/checkpoint" \
  --training.steps "$STEPS" --training.seq_len "$SEQLEN" \
  --checkpoint.interval "$SPE" --checkpoint.keep_latest_k 0 --metrics.log_freq 50 \
  > "$TRAINLOG" 2>&1 &
TRAIN_PID=$!
# torchrun keeps the titan logger output in its own per-rank log, not in the redirected stdout
TITANLOG=""; for _ in $(seq 1 120); do TITANLOG=$(ls "$DUMP"/torchrun_logs/*/attempt_0/0/stdout.log 2>/dev/null | head -1); [ -n "$TITANLOG" ] && break; sleep 5; done
[ -n "$TITANLOG" ] || { echo "ABORT - no torchrun log under $DUMP/torchrun_logs" | tee -a "$OUT"; exit 1; }
echo "[$TAG] titan log $TITANLOG" | tee -a "$OUT"

stage() {  # stage <step-dir> <clean-dir>: HF serving dir. Interval saves are DCP (model+optimizer),
           # so convert them; only the final save is already model-only safetensors.
  local src=$1 clean=$2
  rm -rf "$clean"; mkdir -p "$clean"
  if ls "$src"/model*.safetensors >/dev/null 2>&1; then
    cp "$src"/model*.safetensors "$clean"/
    [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  else
    local cg
    cg=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' \
         | awk -F, '$2+0>=20000 {print $2+0, $1}' | sort -rn | head -1 | awk '{print $2}')
    CUDA_VISIBLE_DEVICES="${cg:-0}" PYTHONPATH="$WT" "$PY" scripts/export_dcp_to_hf.py \
      --module ouro --config ouro_2_6b_thinking_terminal_sft \
      --checkpoint "$src" --hf_template "$BASE" --out "$clean" --dtype bf16 \
      > "$TMP/${TAG}_export_$(basename "$src").log" 2>&1
    rm -rf "$clean/sharded"
  fi
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json \
           tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  ls "$clean"/model*.safetensors >/dev/null 2>&1 || { echo "ABORT - staging $clean produced no weights" | tee -a "$OUT"; exit 1; }
}

for k in $(seq 1 "$EPOCHS"); do
  step=$((SPE * k)); src="$DUMP/checkpoint/step-$step"
  # wait for the k-th "Finished saving" line (checkpoint k complete) or training death
  while :; do
    saved=$(grep -cE "Finished saving the checkpoint|Saving a model only checkpoint" "$TITANLOG" 2>/dev/null); saved=${saved:-0}
    { [ "$saved" -ge "$k" ] && [ -d "$src" ]; } && break
    ls "$src"/model*.safetensors >/dev/null 2>&1 && break
    kill -0 "$TRAIN_PID" 2>/dev/null || { echo "ABORT - training died before epoch $k ($(grep -m1 -E 'Error|OutOfMemory' "$TITANLOG" "$TRAINLOG" | cut -c1-120))" | tee -a "$OUT"; exit 1; }
    sleep 60
  done
  sleep 30
  loss=$(grep -oE "step: *$step .*loss: *[0-9.]+" "$TITANLOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "loss: *[0-9.]+")
  clean="${DUMP}_ep${k}_clean"; stage "$src" "$clean"
  echo "[$TAG] epoch $k (step $step, $loss) staged $clean; eval starting $(date '+%m-%d %H:%M')" | tee -a "$OUT"
  CKPT="$clean" BASENAME="${TAG}_ep${k}" GATE=0 MINFREE=28000 bash eval_parallel.sh >> "$OUT" 2>&1
  echo "[$TAG] epoch $k eval done $(date '+%m-%d %H:%M')" | tee -a "$OUT"
done
wait "$TRAIN_PID"
echo "=== $TAG summary (12 tasks x 2 attempts, stock terminus-2) ===" | tee -a "$OUT"
for k in $(seq 1 "$EPOCHS"); do grep -E "^${TAG}_ep${k}: " "$OUT" | tail -1; done | tee -a "$OUT"
echo "--- $TAG done $(date '+%m-%d %H:%M') ---" | tee -a "$OUT"
