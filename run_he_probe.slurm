#!/bin/bash
#SBATCH --job-name=he_probe
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=00:30:00
#SBATCH --output=logs/he_probe_%j.out
#SBATCH --error=logs/he_probe_%j.err

# Probe: which prompt protocol makes the RAW HF base Ouro-1.4B behave on HumanEval?
# Runs the standalone evaluator on the first LIMIT problems under several settings,
# all at full R=4 recurrence, and prints pass@1 for each.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/scratch/bcjw/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export ALLOW_CODE_EXECUTION=1 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false

HF=./assets/hf/Ouro-1.4B
LIMIT=${LIMIT:-15}
OUT=outputs/he_probe_${SLURM_JOB_ID}; mkdir -p "$OUT"

run () {  # name  extra-args...
    local name=$1; shift
    python -u scripts/evaluate_humaneval.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 --limit "$LIMIT" \
        --output_path "$OUT/${name}.json" "$@" > "$OUT/${name}.log" 2>&1 || true
    python - "$OUT/${name}.json" "$name" <<'PY'
import json,sys
try:
    s=json.load(open(sys.argv[1]))["ouro_eval_summary"]
    print(f"  {sys.argv[2]:18s} pass@1={s['pass@1']:.3f}  n={s['n_problems']}")
except Exception as e:
    print(f"  {sys.argv[2]:18s} FAILED ({e})")
PY
}

echo "Probing $LIMIT problems, raw HF base, R=4:"
run raw_nobos                       # raw completion, no BOS  (current base slurm)
run raw_bos        --add_bos        # raw completion, +BOS
run chat           --chat           # ChatML instruction wrapper (forces +BOS)
echo "done -> $OUT"
