#!/bin/bash
#SBATCH --job-name=he_ep_soft
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --output=logs/he_ep_soft_%j.out
#SBATCH --error=logs/he_ep_soft_%j.err

# Targeted test of a SOFTENED system prompt on the 16 problems where complexsys
# swung things (9 rescues + 7 regressions). The complex prompt's "reason step by
# step / handle every edge case" pushed the 1.4B to OVER-ENGINEER, breaking 7
# previously-passing problems (e.g. HE/73 degenerated into an if n==2..n==34
# ladder; HE/138 over-constrained to distinct evens). The softened prompt keeps
# edge-case awareness but drops step-by-step and explicitly asks for the SIMPLEST
# direct code. Goal: keep the 9 rescues AND recover the 7 regressions.
# Prefill ON (complexsys's setting). 16 problems split over 2 GPUs (~40 min).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1 ALLOW_CODE_EXECUTION=1

HF=./assets/hf/Ouro-1.4B
IDS=outputs/softcheck_task_ids.txt          # 16 targeted problems
OUT=outputs/he_ep_soft_${SLURM_JOB_ID}; mkdir -p "$OUT"

# Softened prompt: edge-case aware, but simple/direct and anti-over-engineering.
SYS="You are an expert Python programmer. Given a function signature and its \
docstring, write a correct, self-contained implementation. Make sure it matches \
the examples in the docstring, and watch for simple edge cases such as empty \
inputs, zero, and negative numbers. Write the simplest, most direct code that \
works -- do not over-engineer, enumerate special cases, or add unnecessary \
branches. Return only valid Python code."
echo "Softened system prompt:"; echo "$SYS"; echo

for si in 0 1; do
    CUDA_VISIBLE_DEVICES=$si python -u scripts/evaluate_humaneval_evalplus.py \
        --module ouro --config ouro_1_4b --hf_checkpoint "$HF" \
        --max_gen_toks 1024 --early_exit_threshold 1.0 \
        --task_ids_file "$IDS" --num_shards 2 --shard_index "$si" \
        --evalplus_prompt --system_prompt "$SYS" \
        --output_path "$OUT/shard${si}.jsonl" > "$OUT/shard${si}.log" 2>&1 &
done
wait
cat "$OUT"/shard*.jsonl > "$OUT/samples.jsonl"
python -m evalplus.sanitize --samples "$OUT/samples.jsonl" >> "$OUT/sanitize.log" 2>&1

python - "$OUT/samples-sanitized.jsonl" > "$OUT/score.txt" 2>&1 <<'PY'
import json, sys, signal
from evalplus.data import get_human_eval_plus
prob = get_human_eval_plus()
rescues = set(l.strip() for l in open("outputs/softcheck_rescues.txt") if l.strip())
regress = set(l.strip() for l in open("outputs/softcheck_regress.txt") if l.strip())
san = {json.loads(l)["task_id"]: json.loads(l)["solution"]
       for l in open(sys.argv[1]) if l.strip()}
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
passed = set()
for tid, sol in san.items():
    p = prob[tid]
    src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
    signal.alarm(10)
    try:
        exec(compile(src, "<s>", "exec"), {}); passed.add(tid)
    except Exception:
        pass
    finally:
        signal.alarm(0)
kept = sorted(rescues & passed, key=lambda x:int(x.split('/')[1]))
lost = sorted(rescues - passed, key=lambda x:int(x.split('/')[1]))
recovered = sorted(regress & passed, key=lambda x:int(x.split('/')[1]))
still = sorted(regress - passed, key=lambda x:int(x.split('/')[1]))
print(f"KEPT rescues     {len(kept)}/{len(rescues)}: {kept}")
print(f"LOST rescues     {len(lost)}/{len(rescues)}: {lost}")
print(f"RECOVERED regress {len(recovered)}/{len(regress)}: {recovered}")
print(f"STILL-broken reg  {len(still)}/{len(regress)}: {still}")
# vs complexsys on these 16: kept=9, recovered=0. Net on the 16 = kept+recovered.
print(f"\nsoftened net-pass on the 16 = {len(kept)+len(recovered)}/16 "
      f"(complexsys was 9/16, default was 7/16)")
PY

echo "==================== SOFTCHECK SUMMARY ===================="
cat "$OUT/score.txt"
echo "----------------------------------------------------------"
echo "Want kept>=8 AND recovered>=4 -> softened prompt beats complexsys; run full 164. Artifacts: $OUT"
