#!/bin/bash
#SBATCH --job-name=he_bosprobe
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=01:00:00
#SBATCH --output=logs/he_bosprobe_%j.out
#SBATCH --error=logs/he_bosprobe_%j.err

# A/B: does forcing a leading BOS before the ChatML template hurt chat HumanEval?
# The released Ouro-1.4B chat template starts directly with <|im_start|> and its
# tokenizer is add_bos_token=False, so native chat usage adds no BOS -- now the
# default. Two conditions on the first LIMIT problems, full R=4, lm-eval
# humaneval scorer, one per GPU in parallel:
#   released  : --chat                   (new default: released template, no BOS)
#   force_bos : --chat --chat_force_bos  (legacy: forced leading BOS)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export ALLOW_CODE_EXECUTION=1 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false

HF=./assets/hf/Ouro-1.4B
LIMIT=${LIMIT:-20}
OUT=outputs/he_bosprobe_${SLURM_JOB_ID}; mkdir -p "$OUT"

run () {  # gpu name extra-args...
    local gpu=$1 name=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/evaluate_humaneval.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 --limit "$LIMIT" \
        --output_path "$OUT/${name}.json" "$@" > "$OUT/${name}.log" 2>&1
}

echo "Chat BOS A/B on $LIMIT problems, R=4:"
run 0 released  --chat &
run 1 force_bos --chat --chat_force_bos &
wait

python - "$OUT" <<'PY'
import json, sys, pathlib
O = pathlib.Path(sys.argv[1])
for name in ("released", "force_bos"):
    try:
        s = json.load(open(O / f"{name}.json"))["ouro_eval_summary"]
        print(f"  {name:12s} pass@1={s['pass@1']:.3f}  n={s['n_problems']}")
    except Exception as e:
        print(f"  {name:12s} FAILED ({e})")
PY
echo "done -> $OUT"
