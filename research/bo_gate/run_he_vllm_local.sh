#!/bin/bash
# vLLM native Ouro HumanEval via EvalPlus's official codegen pipeline (KV cache +
# batching). EvalPlus default/canonical prompt, greedy, fp16, full 164 problems.
# Uses vLLM 0.24.0's in-tree OuroForCausalLM (full R=4, no adaptive early-exit).
set -euo pipefail
cd /home/jli199/torchtitan

PY=.venv-vllm/bin/python
MODEL=./assets/hf/Ouro-1.4B
GPU=${1:-9}
ROOT=outputs/vllm_evalplus_local; mkdir -p "$ROOT"

export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Greedy decode doesn't need flashinfer's JIT-compiled sampler kernel; disabling
# it avoids a runtime nvcc build (which failed: no CUDA toolkit on PATH).
export VLLM_USE_FLASHINFER_SAMPLER=0
# Provide a CUDA toolkit (nvcc) as a safety net for any other JIT path.
export CUDA_HOME=/software/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "=== vLLM codegen (greedy, EvalPlus default prompt, fp16) on GPU $GPU ==="
CUDA_VISIBLE_DEVICES=$GPU $PY -m evalplus.codegen \
    --model "$MODEL" --dataset humaneval --greedy \
    --backend vllm --tp 1 --trust_remote_code --dtype float16 \
    --root "$ROOT"

SAMPLES=$(ls "$ROOT"/humaneval/*.jsonl 2>/dev/null | grep -v sanitized | head -1)
echo "Samples: $SAMPLES"
$PY -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"

echo "=== EvalPlus base HumanEval (vs custom-gen canonical 0.683 / paper 0.744) ==="
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAN" --base_only | tee "$ROOT/score.txt"
echo "done -> $ROOT"
