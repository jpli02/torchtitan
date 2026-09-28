#!/bin/bash
#SBATCH --job-name=ouro_he_kv_np
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=03:00:00
#SBATCH --output=logs/ouro_he_kv_np_%j.out
#SBATCH --error=logs/ouro_he_kv_np_%j.err

# Full 164 HumanEval EvalPlus base score for the no-system/no-prefill prompt
# that implied 116/164 = 0.7073 in the no-cache failing-set sweep.
# Uses the raw pre-SFT HF checkpoint with KV cache and threshold=1.0.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
OUT=outputs/he_nosys_noprefill_kv_${SLURM_JOB_ID}; mkdir -p "$OUT"

for si in 0 1; do
    CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --num_shards 2 --shard_index "$si" \
        --evalplus_prompt --no_system --no_prefill --kv_cache \
        --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1 &
done
wait

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "Combined $(wc -l < "$SAMPLES") samples -> $SAMPLES"
python -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
python -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$OUT/evalplus_score.txt"

echo "=================================================="
echo "no-system + no-prefill + KV-cache FULL 164 base pass@1 above."
echo "Target reference: no-cache failing-set inference = 116/164 = 0.7073."
echo "Artifacts under: $OUT"
