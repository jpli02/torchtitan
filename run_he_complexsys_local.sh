#!/bin/bash
# Local bf16 complexsys HumanEval repro for Ouro-1.4B: EvalPlus-canonical protocol
# (R=4, prefill ON) + the complex expert-programmer system prompt. Mirrors
# run_ouro_he_complexsys_full.slurm. Sharded across 3 free GPUs (0,1,2).
set -euo pipefail
cd /home/jli199/torchtitan

PY=.venv-eval/bin/python
HF=./assets/hf/Ouro-1.4B
GPUS=(0 1 2)
NSHARD=${#GPUS[@]}
OUT=outputs/he_complexsys_local
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

# The winning complex system prompt (identical to run_ouro_he_complexsys_full.slurm).
SYS="You are an expert Python programmer and careful problem solver. Given a \
function signature and its docstring, implement the function so that it is \
correct and self-contained. Read the docstring closely, including every example, \
and make sure your implementation reproduces those examples exactly. Reason \
through the algorithm step by step before writing code, and handle edge cases \
carefully: empty inputs, zero, negative numbers, single-element or empty \
collections, duplicates, and boundary conditions. Prefer clear, correct logic \
over cleverness, and return only valid Python code."

echo "[driver] bf16 complexsys | shards=$NSHARD gpus=${GPUS[*]}"
for si in $(seq 0 $((NSHARD - 1))); do
  gpu=${GPUS[$si]}
  CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/evaluate_humaneval_evalplus.py \
    --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
    --max_gen_toks 1024 --early_exit_threshold 1.0 \
    --num_shards "$NSHARD" --shard_index "$si" \
    --evalplus_prompt --system_prompt "$SYS" \
    --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1 &
done
wait
echo "[driver] generation done"

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "[driver] combined $(wc -l < "$SAMPLES") samples"
$PY -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$OUT/evalplus_score.txt"
echo "[driver] DONE -> $OUT/evalplus_score.txt"
