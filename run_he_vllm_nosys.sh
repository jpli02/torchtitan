#!/bin/bash
# vLLM + EvalPlus HumanEval for Ouro-1.4B with the system prompt REMOVED
# (model dir uses a chat template whose default-system else-branch is stripped).
# Mirrors run_he_vllm_local.sh exactly except MODEL + GPU are parameters.
# Usage: run_he_vllm_nosys.sh <gpu> <model_dir> <tag>
set -euo pipefail
cd /home/jli199/torchtitan

PY=.venv-vllm/bin/python
GPU=${1:-6,7}                     # comma list; tp = number of GPUs
MODEL=${2:-./assets/hf/Ouro-1.4B-nosys}
TAG=${3:-vllm_nosys}
ROOT=outputs/${TAG}; mkdir -p "$ROOT"
TP=$(awk -F',' '{print NF}' <<< "$GPU")   # tensor-parallel size = #GPUs

export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/software/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== vLLM codegen (greedy, EvalPlus prompt, fp16, NO system prompt) GPUs $GPU tp=$TP model $MODEL ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -m evalplus.codegen \
    --model "$MODEL" --dataset humaneval --greedy \
    --backend vllm --tp "$TP" --trust_remote_code --dtype float16 \
    --root "$ROOT"

SAMPLES=$(ls "$ROOT"/humaneval/*.jsonl 2>/dev/null | grep -v sanitized | head -1)
echo "Samples: $SAMPLES"
$PY -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"

echo "=== EvalPlus base HumanEval (baseline WITH system = 0.683 / paper 0.744) ==="
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$ROOT/score.txt"
echo "done -> $ROOT"
