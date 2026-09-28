#!/bin/bash
# Local fp32 + EvalPlus-canonical-prompt HumanEval repro for Ouro-1.4B.
# Full 164 problems, sharded across 4 free GPUs (0,1,2,4). Mirrors
# run_ouro_he_fp32_full.slurm but for this non-slurm box.
set -euo pipefail
cd /home/jli199/torchtitan

PY=.venv-eval/bin/python
HF=./assets/hf/Ouro-1.4B
GPUS=(0 1 2 4)
NSHARD=${#GPUS[@]}
OUT=outputs/he_fp32_local
mkdir -p "$OUT"

export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

echo "[driver] fp32 + evalplus canonical prompt | shards=$NSHARD gpus=${GPUS[*]}"
for si in $(seq 0 $((NSHARD - 1))); do
  gpu=${GPUS[$si]}
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/evaluate_humaneval_evalplus.py \
    --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
    --max_gen_toks 1024 --early_exit_threshold 1.0 \
    --num_shards "$NSHARD" --shard_index "$si" \
    --evalplus_prompt --fp32 \
    --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1 &
done
wait
echo "[driver] generation done"

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "[driver] combined $(wc -l < "$SAMPLES") samples"

$PY -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
echo "[driver] scoring base HumanEval..."
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$OUT/evalplus_score.txt"
echo "[driver] DONE -> $OUT/evalplus_score.txt"
