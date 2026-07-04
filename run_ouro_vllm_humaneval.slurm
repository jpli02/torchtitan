#!/bin/bash
#SBATCH --job-name=ouro_vllm_he
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=01:00:00
#SBATCH --output=logs/ouro_vllm_he_%j.out
#SBATCH --error=logs/ouro_vllm_he_%j.err

# Paper-exact + fast HumanEval via vLLM + EvalPlus's official codegen pipeline
# (KV cache + batching). This also serves as the feasibility gate: if vLLM can't
# load the Ouro (LoopLM) architecture, codegen fails fast on model load.
# vLLM runs full total_ut_steps=4 (no adaptive exit) -- exactly the R=4 reference.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/hdd/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"

source /work/hdd/bdjz/jli37/venvs/vllm/bin/activate

# Env for vLLM: fixes the libcudart.so.13 import failure (cu13 libs onto
# LD_LIBRARY_PATH) plus HF/vLLM knobs. Must come after venv activate.
# cwd is already $SLURM_SUBMIT_DIR (cd above); $0 points at the spooled copy so
# don't use dirname "$0" here.
set -a; source ./.env.vllm; set +a

MODEL=${MODEL:-./assets/hf/Ouro-1.4B}
ROOT=outputs/vllm_evalplus_${SLURM_JOB_ID}; mkdir -p "$ROOT"

echo "=== vLLM codegen (greedy, EvalPlus canonical prompt) ==="
python -m evalplus.codegen \
    --model "$MODEL" --dataset humaneval --greedy \
    --backend vllm --tp 1 --trust_remote_code --dtype bfloat16 \
    --root "$ROOT"

SAMPLES=$(ls "$ROOT"/humaneval/*.jsonl 2>/dev/null | grep -v sanitized | head -1)
echo "Samples: $SAMPLES"
python -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"

echo "=== EvalPlus base HumanEval (vs paper 0.744 / custom-gen canonical 0.683) ==="
python -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$ROOT/score.txt"
echo "done -> $ROOT"
