#!/bin/bash
# Local (non-slurm) HumanEval EvalPlus repro for Ouro-1.4B across N GPUs.
# Usage: run_he_local.sh <tag> <evalplus_prompt|chat|raw> <max_gen_toks> <num_gpus> [extra args...]
set -euo pipefail
cd /home/jli199/torchtitan

TAG=${1:-run}
MODE=${2:-evalplus}
MAXTOK=${3:-768}
NGPU=${4:-10}
shift $(( $# < 4 ? $# : 4 )) || true
EXTRA="$*"

PY=.venv-eval/bin/python
OUT=outputs/he_${TAG}
mkdir -p "$OUT"

MODE_ARG=""
case "$MODE" in
  evalplus) MODE_ARG="--evalplus_prompt" ;;
  chat)     MODE_ARG="--chat" ;;
  raw)      MODE_ARG="" ;;
  *) echo "bad mode $MODE"; exit 1 ;;
esac

echo "[driver] tag=$TAG mode=$MODE maxtok=$MAXTOK ngpu=$NGPU extra='$EXTRA'"
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i TOKENIZERS_PARALLELISM=false $PY -u scripts/evaluate_humaneval_evalplus.py \
    --module ouro --config ouro_1_4b \
    --hf_checkpoint ./assets/hf/Ouro-1.4B \
    --max_gen_toks "$MAXTOK" --early_exit_threshold 1.0 \
    --num_shards "$NGPU" --shard_index "$i" \
    --output_path "$OUT/shard${i}.jsonl" \
    $MODE_ARG $EXTRA > "$OUT/shard${i}.log" 2>&1 &
done
wait
echo "[driver] generation done"

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "[driver] combined $(wc -l < "$SAMPLES") samples"

$PY -m evalplus.sanitize --samples "$SAMPLES" >/dev/null 2>&1 || $PY -m evalplus.sanitize --samples "$SAMPLES"
SANITIZED="${SAMPLES%.jsonl}-sanitized.jsonl"
echo "[driver] scoring base HumanEval..."
$PY -m evalplus.evaluate --dataset humaneval --samples "$SANITIZED" --base_only | tee "$OUT/score.txt"
echo "[driver] DONE -> $OUT/score.txt"
