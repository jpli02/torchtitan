#!/bin/bash
#SBATCH --job-name=he_ep_combo
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=03:30:00
#SBATCH --output=logs/he_ep_combo_%j.out
#SBATCH --error=logs/he_ep_combo_%j.err

# Combo prompt sweep on the EvalPlus-canonical protocol (R=4), targeted at ONLY
# the 52 problems that failed the 0.683 full run. The single-lever sweep (job
# 19851761) gave: noprefill 7/52, codesys 5/52, nosys 3/52, with largely disjoint
# rescue sets (union = 10). This tests whether the strongest lever (noprefill)
# STACKS with a system-prompt change:
#   codesys_noprefill : "You are an expert Python programmer." + drop prefill
#   nosys_noprefill   : suppress default system prompt        + drop prefill
# Each arm splits the 52 over both GPUs (num_shards=2); arms run sequentially.
# Rescues only (no regression coverage) -> promote the winner to a full 164 run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
FAILS=outputs/failing_task_ids_19821485.txt   # 52 task_ids from the 0.683 run
OUT=outputs/he_ep_combo_${SLURM_JOB_ID}; mkdir -p "$OUT"
NFAIL=$(grep -c . "$FAILS")
echo "Targeting $NFAIL failing problems from $FAILS"

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

echo "=== combo prompt sweep (R=4, $NFAIL problems/arm) ==="
run_arm codesys_noprefill --system_prompt "You are an expert Python programmer." --no_prefill
run_arm nosys_noprefill   --system_prompt "" --no_prefill

echo
echo "==================== COMBO SUMMARY ===================="
echo "(single-lever ref: noprefill 7/52, codesys 5/52, nosys 3/52; union=10)"
cat "$OUT"/*/score.txt
echo "------------------------------------------------------"
echo "Any arm rescuing >7 -> levers stack; promote to a full 164 run to net out regressions. Artifacts: $OUT"
