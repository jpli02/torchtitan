#!/bin/bash
#SBATCH --job-name=ouro_he_chat
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --output=logs/ouro_he_chat_%j.out
#SBATCH --error=logs/ouro_he_chat_%j.err

# Full HumanEval (all 164 problems) on the RAW pretrained Ouro-1.4B (pre-SFT),
# evaluated with the ChatML instruction wrapper (--chat, which forces +BOS),
# at full R=4 recurrence (early_exit_threshold=1.0).
#
# Rationale: the he_probe job (19738480) showed that on the raw HF base model,
# raw code completion (with or without BOS) scores pass@1=0.000 on the first 15
# problems, while the --chat protocol recovers it to 0.667. This is the full-set
# counterpart of that probe, to get a proper 164-problem chat number.
#
# Note: scored with lm-eval-harness's `humaneval` task (no sanitization). For the
# exact paper EvalPlus metric use run_ouro_humaneval_evalplus.slurm.
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
# 1024 matched the probe's chat run and is ample headroom.
MAX_GEN_TOKS=${MAX_GEN_TOKS:-1024}
NUM_SHARDS=${NUM_SHARDS:-2}
# Full UT recursion = R=4, matching the paper's reported base-model setting.
EARLY_EXIT_THRESHOLD=${EARLY_EXIT_THRESHOLD:-1.0}
# LIMIT empty => all 164 problems. Set e.g. LIMIT=20 for a quick check.
LIMIT=${LIMIT:-}

OUT=outputs/humaneval_chat_${SLURM_JOB_ID}
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# Launch one eval process per GPU, each on a disjoint problem shard.
# --chat: ChatML instruction wrapper (forces +BOS), the protocol the probe found works.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/evaluate_humaneval.py \
        --module "$MODULE" \
        --config "$CONFIG" \
        --hf_checkpoint "$HF_CHECKPOINT" \
        --chat \
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
    "protocol": "chat (ChatML wrapper, +BOS), R=4, lm-eval humaneval",
    "aggregated": True,
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("=" * 50)
print(f"FULL HumanEval chat Ouro-1.4B (ChatML wrapper): pass@1 = {summary['pass@1']:.4f} "
      f"over {tot_p} problems  (avg_loops={summary['avg_loops']})")
print(f"Saved: {out / 'summary.json'}")
PY
