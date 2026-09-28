#!/bin/bash
#SBATCH --job-name=ouro_he_full
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --output=logs/ouro_he_full_%j.out
#SBATCH --error=logs/ouro_he_full_%j.err

# Full HumanEval (all 164 problems) on the RAW pretrained Ouro-1.4B (pre-SFT),
# loaded directly from HF safetensors, with adaptive early exit ON
# (early_exit_threshold=0.9, matching the BO eval) so this is the apples-to-apples
# pre-SFT baseline for both pass@1 and avg_loops. Sharded across 2 GPUs: each GPU
# scores problems[shard::2], then pass@1 / avg_loops are aggregated exactly.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

export TMPDIR=${TMPDIR:-/scratch/bcjw/jli37/tmp}
mkdir -p "$TMPDIR"

source .venv/bin/activate
python -c "import lm_eval" 2>/dev/null || pip install --quiet 'lm_eval[code]'

export ALLOW_CODE_EXECUTION=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

MODULE=${MODULE:-ouro}
CONFIG=${CONFIG:-ouro_1_4b}
HF_CHECKPOINT=${HF_CHECKPOINT:-./assets/hf/Ouro-1.4B}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-512}
NUM_SHARDS=${NUM_SHARDS:-2}
# Adaptive early exit ON (matches BO eval). Set to 1.0 for full UT recursion.
EARLY_EXIT_THRESHOLD=${EARLY_EXIT_THRESHOLD:-0.9}
# LIMIT empty => all 164 problems. Set e.g. LIMIT=20 for a quick check.
LIMIT=${LIMIT:-}

OUT=outputs/humaneval_hf_full_${SLURM_JOB_ID}
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# Launch one eval process per GPU, each on a disjoint problem shard.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/evaluate_humaneval.py \
        --module "$MODULE" \
        --config "$CONFIG" \
        --hf_checkpoint "$HF_CHECKPOINT" \
        --max_gen_toks "$MAX_GEN_TOKS" \
        --early_exit_threshold "$EARLY_EXIT_THRESHOLD" \
        --chat \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$i" \
        --output_path "$OUT/shard${i}.json" \
        $LIMIT_ARG \
        > "$OUT/shard${i}.log" 2>&1 &
done
wait

# Aggregate shard summaries exactly: pass@1 = sum(correct)/sum(problems),
# avg_loops = sum(loop_sum)/sum(loop_count).
python - "$OUT" "$NUM_SHARDS" <<'PY'
import json, sys, pathlib
out, n = pathlib.Path(sys.argv[1]), int(sys.argv[2])
tot_p = 0; tot_c = 0.0; lsum = 0.0; lcnt = 0
for i in range(n):
    s = json.load(open(out / f"shard{i}.json"))["ouro_eval_summary"]
    p1, np_ = s["pass@1"], int(s["n_problems"])
    tot_p += np_; tot_c += p1 * np_
    if s.get("loop_sum") is not None and s.get("loop_count"):
        lsum += float(s["loop_sum"]); lcnt += int(s["loop_count"])
    print(f"  shard{i}: pass@1={p1:.4f}  n={np_}  eval_time_s={s.get('eval_time_s'):.1f}")
summary = {
    "pass@1": tot_c / tot_p if tot_p else 0.0,
    "n_problems": tot_p,
    "avg_loops": (lsum / lcnt) if lcnt else None,
    "num_shards": n,
    "model": "Ouro-1.4B (pre-SFT, HF safetensors)",
    "aggregated": True,
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("=" * 50)
print(f"FULL HumanEval pre-SFT Ouro-1.4B: pass@1 = {summary['pass@1']:.4f} "
      f"over {tot_p} problems  (avg_loops={summary['avg_loops']})")
print(f"Saved: {out / 'summary.json'}")
PY
