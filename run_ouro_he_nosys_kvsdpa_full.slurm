#!/bin/bash
#SBATCH --job-name=ouro_he_kvsdpa
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:30:00
#SBATCH --output=logs/ouro_he_kv_%j.out
#SBATCH --error=logs/ouro_he_kv_%j.err

# FULL 164 HumanEval, best lever (--no_system) via the KV-cache HF path (~5.5x
# faster than the no-cache decoder; validated argmax-equivalent within bf16 noise
# in job 19974730). R=4, EvalPlus-canonical prompt with the system turn removed
# entirely. Expected ~0.707 (+-1-2 problems of bf16 cache noise). ~40 min.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
OUT=outputs/he_nosys_kvsdpa_${SLURM_JOB_ID}; mkdir -p "$OUT"

# 1) Generate all 164 via KV cache, no-system prompt, one shard per GPU.
for si in 0 1; do
    CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --num_shards 2 --shard_index "$si" \
        --evalplus_prompt --no_system --kv_cache --kv_attn_impl sdpa \
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
echo "no-system + KV-cache FULL 164 base pass@1 above."
echo "Refs: default bf16 0.683 | no-system (no-cache) 0.707 | complexsys 0.695 | paper 0.744"
echo "Artifacts under: $OUT"
