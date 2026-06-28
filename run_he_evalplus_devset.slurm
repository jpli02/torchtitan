#!/bin/bash
#SBATCH --job-name=he_ep_dev
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --output=logs/he_ep_dev_%j.out
#SBATCH --error=logs/he_ep_dev_%j.err

# Fast stratified-subset read of the EvalPlus *canonical-prompt* protocol before
# committing to the ~5h full run. ~33 evenly-spaced HumanEval problems
# (task_ids[0::10] on GPU0, task_ids[5::10] on GPU1 -- the generator sorts task
# ids, so this is a fixed spread across the set), full R=4, --evalplus_prompt.
# EvalPlus's evaluate asserts all 164 are present, so we sanitize the subset and
# score base pass@1 ourselves (exec sanitized solution + base test, per-problem
# timeout) -- the same scorer that reproduced EvalPlus's base_status exactly.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
OUT=outputs/he_ep_dev_${SLURM_JOB_ID}; mkdir -p "$OUT"

gen () {  # gpu shard_index
    local gpu=$1 si=$2
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --num_shards 10 --shard_index "$si" --evalplus_prompt \
        --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1
}
echo "Generating stratified subset (canonical EvalPlus prompt, R=4)..."
gen 0 0 &
gen 1 5 &
wait

SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
python -m evalplus.sanitize --samples "$SAMPLES"
SAN="${SAMPLES%.jsonl}-sanitized.jsonl"

python - "$SAN" <<'PY'
import json, sys, signal
from evalplus.data import get_human_eval_plus
prob = get_human_eval_plus()
san = {json.loads(l)["task_id"]: json.loads(l)["solution"]
       for l in open(sys.argv[1]) if l.strip()}
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
passed = 0
for tid, sol in san.items():
    p = prob[tid]
    src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
    signal.alarm(10)
    try:
        exec(compile(src, "<s>", "exec"), {}); passed += 1
    except Exception:
        pass
    finally:
        signal.alarm(0)
n = len(san)
print("=" * 50)
print(f"DEVSET base HumanEval pass@1 = {passed/n:.3f}  ({passed}/{n})  [canonical prompt]")
print("Reference: paper 0.744 | our-wrapper full 0.610")
PY
echo "done -> $OUT"
