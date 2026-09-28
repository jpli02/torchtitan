#!/bin/bash
#SBATCH --job-name=ouro_he_base
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --output=logs/ouro_he_base_%j.out
#SBATCH --error=logs/ouro_he_base_%j.err

# Full HumanEval (all 164 problems) on the RAW pretrained Ouro-1.4B (pre-SFT),
# evaluated the way the paper does: RAW CODE COMPLETION (no chat template), full
# R=4 recurrence (early_exit_threshold=1.0). This is the apples-to-apples repro of
# the paper's Table 7 base-model number (74.40 via EvalPlus). The base model is NOT
# instruction-tuned, so --chat (ChatML "reply with only the body") is the wrong
# protocol and depresses pass@1 / makes every completion echo the docstring back.
# Reserve --chat for the SFT'd checkpoint only.
#
# Note: this scores with lm-eval-harness's `humaneval` task (no sanitization), which
# runs a few points below the paper's EvalPlus number. For the exact paper metric use
# run_ouro_humaneval_evalplus.slurm.
#
# Sharded across 2 GPUs: each GPU scores problems[shard::2], then aggregated exactly.

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
# Match the lm-eval humaneval task budget (the chat run capped at 512 and the
# docstring re-emission ate into it); raw completion is short, 1024 is ample.
MAX_GEN_TOKS=${MAX_GEN_TOKS:-1024}
NUM_SHARDS=${NUM_SHARDS:-2}
# Full UT recursion = R=4, matching the paper's reported base-model setting.
EARLY_EXIT_THRESHOLD=${EARLY_EXIT_THRESHOLD:-1.0}
# LIMIT empty => all 164 problems. Set e.g. LIMIT=20 for a quick check.
LIMIT=${LIMIT:-}

OUT=outputs/humaneval_base_${SLURM_JOB_ID}
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# Launch one eval process per GPU, each on a disjoint problem shard.
# NO --chat: raw signature+docstring completion (base-model protocol).
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/evaluate_humaneval.py \
        --module "$MODULE" \
        --config "$CONFIG" \
        --hf_checkpoint "$HF_CHECKPOINT" \
        --max_gen_toks "$MAX_GEN_TOKS" \
        --early_exit_threshold "$EARLY_EXIT_THRESHOLD" \
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
    "protocol": "raw completion (no chat), R=4, lm-eval humaneval",
    "aggregated": True,
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("=" * 50)
print(f"FULL HumanEval base Ouro-1.4B (raw completion): pass@1 = {summary['pass@1']:.4f} "
      f"over {tot_p} problems  (avg_loops={summary['avg_loops']})")
print(f"Saved: {out / 'summary.json'}")
PY
