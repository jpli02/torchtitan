#!/bin/bash
#SBATCH --job-name=ouro_he_cplx
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160g
#SBATCH --time=05:00:00
#SBATCH --output=logs/ouro_he_cplx_%j.out
#SBATCH --error=logs/ouro_he_cplx_%j.err

# FULL 164-problem HumanEval run of the winning prompt config: EvalPlus-canonical
# protocol (R=4) + the complex system prompt (prefill ON), which rescued 9/52 of
# the fails in the targeted sweep (job 19858794). This full run nets those rescues
# against any REGRESSIONS on the 112 previously-passing problems -> the real
# pass@1 vs the 0.683 default-prompt baseline. Scored with evalplus.evaluate
# (same scorer as the 0.683 runs), base tests.
#
# Complex prompt makes generations longer -> 6h walltime (vs ~4h default).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
OUT=outputs/he_cplx_full_${SLURM_JOB_ID}; mkdir -p "$OUT"

# The winning complex system prompt (identical to the sweep's complexsys arm).
SYS="You are an expert Python programmer and careful problem solver. Given a \
function signature and its docstring, implement the function so that it is \
correct and self-contained. Read the docstring closely, including every example, \
and make sure your implementation reproduces those examples exactly. Reason \
through the algorithm step by step before writing code, and handle edge cases \
carefully: empty inputs, zero, negative numbers, single-element or empty \
collections, duplicates, and boundary conditions. Prefer clear, correct logic \
over cleverness, and return only valid Python code."
echo "System prompt:"; echo "$SYS"; echo

# 1) Generate all 164 problems, one disjoint shard per GPU (4 GPUs -> ~41/shard;
#    the complex prompt is ~1.6x slower/problem, so 4-way keeps it well under wall).
for si in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --num_shards 4 --shard_index "$si" \
        --evalplus_prompt --system_prompt "$SYS" \
        --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1 &
done
wait

# 2) Combine, sanitize, score (base HumanEval) with EvalPlus.
SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "Combined $(wc -l < "$SAMPLES") samples -> $SAMPLES"
python -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
python -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$OUT/evalplus_score.txt"

echo "=================================================="
echo "complexsys FULL 164 base pass@1 above."
echo "Baseline (default prompt) = 0.683 | paper Table 7 = 0.744"
echo "Artifacts under: $OUT"
