#!/bin/bash
# HumanEval EvalPlus repro mapping shards onto an explicit list of GPUs (robust to
# a shared machine where only some GPUs are free).
# Usage: run_he_gpus.sh <tag> <evalplus|chat|raw> <max_gen_toks> <gpu_csv> [extra args...]
#   gpu_csv e.g. "0,2,3,5,6,7"  -> num_shards = count, shard i on that GPU.
set -euo pipefail
cd /home/jli199/torchtitan

TAG=${1:-run}; MODE=${2:-evalplus}; MAXTOK=${3:-1024}; GPUS=${4:-0,1,2,3,4,5,6,7}
shift $(( $# < 4 ? $# : 4 )) || true
EXTRA="$*"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
PY=.venv-eval/bin/python
OUT=outputs/he_${TAG}; mkdir -p "$OUT"

MODE_ARG=""
case "$MODE" in
  evalplus) MODE_ARG="--evalplus_prompt" ;;
  chat)     MODE_ARG="--chat" ;;
  raw)      MODE_ARG="" ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[driver] tag=$TAG mode=$MODE maxtok=$MAXTOK gpus=$GPUS nshards=$NGPU extra='$EXTRA'"
for idx in "${!GPU_ARR[@]}"; do
  g=${GPU_ARR[$idx]}
  CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $PY -u scripts/evaluate_humaneval_evalplus.py \
    --module ouro --config ouro_1_4b --hf_checkpoint ./assets/hf/Ouro-1.4B \
    --max_gen_toks "$MAXTOK" --early_exit_threshold 1.0 \
    --num_shards "$NGPU" --shard_index "$idx" \
    --output_path "$OUT/shard${idx}.jsonl" \
    $MODE_ARG $EXTRA > "$OUT/shard${idx}.log" 2>&1 &
done
wait
echo "[driver] generation done"

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
N=$(wc -l < "$SAMPLES")
echo "[driver] combined $N samples"
if [ "$N" -ne 164 ]; then
  echo "[driver] WARNING: expected 164 samples, got $N (a shard may have failed)"
fi

$PY -m evalplus.sanitize --samples "$SAMPLES" >/dev/null 2>&1 || $PY -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
echo "[driver] scoring base HumanEval..."
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$OUT/score.txt"
echo "[driver] DONE -> $OUT/score.txt"
