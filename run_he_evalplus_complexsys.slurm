#!/bin/bash
#SBATCH --job-name=he_ep_cplx
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=03:30:00
#SBATCH --output=logs/he_ep_cplx_%j.out
#SBATCH --error=logs/he_ep_cplx_%j.err

# Test a DETAILED/complex system prompt (targets the observed failure modes:
# edge cases like 0/empty/negative inputs, reading the docstring examples) vs the
# simple "You are an expert Python programmer." (codesys, which rescued 5/52).
# Two arms over the 52 failing problems of the 0.683 run, at both prefill settings
# so the complex-prompt effect is isolated:
#   complexsys          : complex system prompt, EvalPlus prefill ON  (vs codesys 5/52)
#   complexsys_noprefill : complex system prompt, prefill OFF          (vs noprefill 7/52)
# Each arm splits the 52 over both GPUs (num_shards=2); arms run sequentially.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
FAILS=outputs/failing_task_ids_19821485.txt   # 52 task_ids from the 0.683 run
OUT=outputs/he_ep_cplx_${SLURM_JOB_ID}; mkdir -p "$OUT"
NFAIL=$(grep -c . "$FAILS")

# The complex system prompt under test.
SYS="You are an expert Python programmer and careful problem solver. Given a \
function signature and its docstring, implement the function so that it is \
correct and self-contained. Read the docstring closely, including every example, \
and make sure your implementation reproduces those examples exactly. Reason \
through the algorithm step by step before writing code, and handle edge cases \
carefully: empty inputs, zero, negative numbers, single-element or empty \
collections, duplicates, and boundary conditions. Prefer clear, correct logic \
over cleverness, and return only valid Python code."
echo "Targeting $NFAIL problems. System prompt:"; echo "$SYS"

# run_arm <name> [extra generator args...]   (splits the failure set over 2 GPUs)
run_arm () {
    local name=$1; shift
    local dir="$OUT/$name"; mkdir -p "$dir"
    for si in 0 1; do
        CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
            --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
            --max_gen_toks 1024 --early_exit_threshold 1.0 \
            --task_ids_file "$FAILS" --num_shards 2 --shard_index "$si" \
            --evalplus_prompt "$@" \
            --output_path "$dir/shard${si}.jsonl" > "$dir/shard${si}.log" 2>&1 &
    done
    wait
    cat "$dir"/shard*.jsonl > "$dir/samples.jsonl"
    python -m evalplus.sanitize --samples "$dir/samples.jsonl" >> "$dir/sanitize.log" 2>&1
    python - "$dir/samples-sanitized.jsonl" "$name" "$NFAIL" > "$dir/score.txt" 2>&1 <<'PY'
import json, sys, signal
from evalplus.data import get_human_eval_plus
prob = get_human_eval_plus()
san = {json.loads(l)["task_id"]: json.loads(l)["solution"]
       for l in open(sys.argv[1]) if l.strip()}
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
rescued = []
for tid, sol in san.items():
    p = prob[tid]
    src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
    signal.alarm(10)
    try:
        exec(compile(src, "<s>", "exec"), {}); rescued.append(tid)
    except Exception:
        pass
    finally:
        signal.alarm(0)
nfail = int(sys.argv[3])
print(f"{sys.argv[2]}\trescued {len(rescued)}/{nfail}\t{sorted(rescued, key=lambda x:int(x.split('/')[1]))}")
PY
    cat "$dir/score.txt"
}

echo "=== complex-system-prompt sweep (R=4, $NFAIL problems/arm) ==="
run_arm complexsys           --system_prompt "$SYS"
run_arm complexsys_noprefill --system_prompt "$SYS" --no_prefill

echo
echo "==================== COMPLEX-SYS SUMMARY ===================="
echo "(single-lever ref: noprefill 7/52, codesys[simple] 5/52, nosys 3/52)"
cat "$OUT"/*/score.txt
echo "------------------------------------------------------------"
echo "Beats codesys(5)/noprefill(7)? -> promote to a full 164 run. Artifacts: $OUT"
