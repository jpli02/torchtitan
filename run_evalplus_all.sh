#!/bin/bash
# Greedy vLLM + EvalPlus eval of Ouro-1.4B (NO system prompt) on both benchmarks,
# reporting base AND plus: HumanEval / HumanEval+ and MBPP / MBPP+.
# Uses the official evalplus.codegen path (faithful prompt handling per dataset).
# Usage: run_evalplus_all.sh <gpu> [model_dir] [tag]
set -euo pipefail
cd /home/jli199/torchtitan

PY=.venv-vllm/bin/python
GPU=${1:-3}                                   # single physical GPU (PCI-bus order)
MODEL=${2:-./assets/hf/Ouro-1.4B-nosys}       # system-prompt-free chat template
TAG=${3:-evalplus_all_nosys}
ROOT=outputs/${TAG}; mkdir -p "$ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID           # make CUDA index match nvidia-smi
export CUDA_VISIBLE_DEVICES=$GPU
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/software/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

for DS in humaneval mbpp; do
  echo "==================== $DS (greedy, no-system, fp16) ===================="
  $PY -m evalplus.codegen \
      --model "$MODEL" --dataset "$DS" --greedy \
      --backend vllm --tp 1 --trust_remote_code --dtype float16 \
      --root "$ROOT"
  SAMPLES=$(ls "$ROOT"/"$DS"/*.jsonl 2>/dev/null | grep -v sanitized | head -1)
  echo "Samples: $SAMPLES"
  $PY -m evalplus.sanitize --samples "$SAMPLES"
  SAN="${SAMPLES%.jsonl}-sanitized.jsonl"
  # No --base_only  => prints both base ($DS) and plus ($DS+) pass@1.
  $PY -m evalplus.evaluate --dataset "$DS" --samples "$SAN" | tee "$ROOT/score_${DS}.txt"
done

echo "===================================================================="
echo "SUMMARY (Ouro-1.4B, no system prompt, greedy, R=4):"
for DS in humaneval mbpp; do
  echo "--- $DS ---"; grep -A1 -iE "\(base tests\)|\+ \(base|base \+ extra" "$ROOT/score_${DS}.txt" 2>/dev/null | grep -iE "pass@1|tests" || cat "$ROOT/score_${DS}.txt" | grep -iE "pass@1"
done
echo "Refs: paper HumanEval 74.4 / MBPP 61.0 (Table 7, if applicable)"
