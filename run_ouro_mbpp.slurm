#!/bin/bash
#SBATCH --job-name=ouro_mbpp
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=12:00:00
#SBATCH --output=logs/ouro_mbpp_%j.out
#SBATCH --error=logs/ouro_mbpp_%j.err

# Full MBPP (all 500 test problems) on the released Ouro-1.4B, chat protocol,
# full R=4 recurrence, scored with lm-eval's `mbpp` task (pass_at_1 = candidate +
# test). Sharded across 2 GPUs (problems[shard::2]); shard summaries aggregated
# exactly. NUM_FEWSHOT=0 (zero-shot instruct) by default; set 3 for the standard
# base-model MBPP protocol.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
export TMPDIR=${TMPDIR:-/work/nvme/bdjz/jli37/tmp}; mkdir -p "$TMPDIR"
source .venv/bin/activate
# mbpp's metric self-test runs at task import and needs this set first.
export ALLOW_CODE_EXECUTION=1 HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false

MODULE=${MODULE:-ouro}
CONFIG=${CONFIG:-ouro_1_4b}
HF_CHECKPOINT=${HF_CHECKPOINT:-./assets/hf/Ouro-1.4B}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-512}        # MBPP solutions are short
NUM_SHARDS=${NUM_SHARDS:-2}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
EARLY_EXIT_THRESHOLD=${EARLY_EXIT_THRESHOLD:-1.0}   # 1.0 = full R=4
LIMIT=${LIMIT:-}                          # empty => all 500 test problems

OUT=outputs/mbpp_${SLURM_JOB_ID}; mkdir -p "$OUT"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/evaluate_mbpp.py \
        --module "$MODULE" --config "$CONFIG" --hf_checkpoint "$HF_CHECKPOINT" \
        --chat --num_fewshot "$NUM_FEWSHOT" \
        --max_gen_toks "$MAX_GEN_TOKS" --early_exit_threshold "$EARLY_EXIT_THRESHOLD" \
        --num_shards "$NUM_SHARDS" --shard_index "$i" \
        --output_path "$OUT/shard${i}.json" \
        $LIMIT_ARG > "$OUT/shard${i}.log" 2>&1 &
done
wait

# Aggregate shards exactly: pass@1 = sum(correct)/sum(problems).
python - "$OUT" "$NUM_SHARDS" <<'PY'
import json, sys, pathlib
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
tot_p = 0; tot_c = 0.0; lsum = 0.0; lcnt = 0
for i in range(n):
    s = json.load(open(out / f"shard{i}.json"))["ouro_eval_summary"]
    p1, np_ = float(s["pass@1"]), int(s["n_problems"])
    tot_p += np_; tot_c += p1 * np_
    if s.get("loop_sum") is not None and s.get("loop_count"):
        lsum += float(s["loop_sum"]); lcnt += int(s["loop_count"])
    print(f"  shard{i}: pass@1={p1:.4f}  n={np_}  eval_time_s={s.get('eval_time_s'):.1f}")
summary = {
    "pass@1": tot_c / tot_p if tot_p else 0.0,
    "n_problems": tot_p,
    "avg_loops": (lsum / lcnt) if lcnt else None,
    "num_shards": n, "task": "mbpp", "aggregated": True,
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("=" * 50)
print(f"FULL MBPP Ouro-1.4B (chat): pass@1 = {summary['pass@1']:.4f} "
      f"over {tot_p} problems  (avg_loops={summary['avg_loops']})")
print(f"Saved: {out / 'summary.json'}")
PY
