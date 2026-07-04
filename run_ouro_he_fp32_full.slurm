#!/bin/bash
#SBATCH --job-name=ouro_he_fp32
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160g
#SBATCH --time=05:00:00
#SBATCH --output=logs/ouro_he_fp32_%j.out
#SBATCH --error=logs/ouro_he_fp32_%j.err

# FULL 164 HumanEval in FLOAT32 with the DEFAULT EvalPlus-canonical prompt (no
# system-prompt override) -- isolates whether bf16 precision (matmul accumulation
# across the deep R=4 UT stack) costs pass@1 vs the paper. Directly comparable to
# the bf16 default baseline 0.683 (same prompt, only precision changes).
# fp32 is ~2x slower -> 4 GPUs (num_shards=4, ~41/shard). Scored with EvalPlus.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
OUT=outputs/he_fp32_full_${SLURM_JOB_ID}; mkdir -p "$OUT"

# 1) Generate all 164 in fp32, default canonical prompt, one shard per GPU.
for si in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --num_shards 4 --shard_index "$si" \
        --evalplus_prompt --fp32 \
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
echo "fp32 default-prompt FULL 164 base pass@1 above."
echo "bf16 default baseline = 0.683 | complexsys bf16 = 0.695 | paper = 0.744"
echo "Artifacts under: $OUT"
