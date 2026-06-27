#!/bin/bash
#SBATCH --job-name=ouro_he_ep
#SBATCH --account=bdjz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=logs/ouro_he_ep_%j.out
#SBATCH --error=logs/ouro_he_ep_%j.err

# Paper-exact HumanEval repro for the base (pre-SFT) Ouro-1.4B: RAW code completion,
# full R=4 recurrence, scored with EvalPlus (HumanEval base + HumanEval+). This is the
# framework the paper used for Table 7 (1.4B base = 74.40 / HumanEval+ 67.40).
#
# Generation is sharded across 2 GPUs (each writes problems[shard::2]); the shards are
# concatenated and scored once with evalplus.sanitize + evalplus.evaluate.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

export TMPDIR=${TMPDIR:-/scratch/bcjw/jli37/tmp}
mkdir -p "$TMPDIR"

source .venv/bin/activate
python -c "import evalplus" 2>/dev/null || pip install --quiet evalplus

export TOKENIZERS_PARALLELISM=false

MODULE=${MODULE:-ouro}
CONFIG=${CONFIG:-ouro_1_4b}
HF_CHECKPOINT=${HF_CHECKPOINT:-./assets/hf/Ouro-1.4B}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-1024}
NUM_SHARDS=${NUM_SHARDS:-4}   # one disjoint problem shard per GPU (data parallel)
EARLY_EXIT_THRESHOLD=${EARLY_EXIT_THRESHOLD:-1.0}   # 1.0 = full R=4 recurrence
LIMIT=${LIMIT:-}
# CHAT=1 => ChatML protocol (required for the released instruct-style Ouro-1.4B,
# which echoes the prompt under raw completion). Empty => raw completion (paper base).
CHAT=${CHAT:-}

OUT=outputs/humaneval_evalplus_${SLURM_JOB_ID}
mkdir -p "$OUT"

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
CHAT_ARG=""
[ -n "$CHAT" ] && CHAT_ARG="--chat"

# 1) Generate raw completions, one disjoint shard per GPU.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/evaluate_humaneval_evalplus.py \
        --module "$MODULE" \
        --config "$CONFIG" \
        --hf_checkpoint "$HF_CHECKPOINT" \
        --max_gen_toks "$MAX_GEN_TOKS" \
        --early_exit_threshold "$EARLY_EXIT_THRESHOLD" \
        --num_shards "$NUM_SHARDS" \
        --shard_index "$i" \
        --output_path "$OUT/shard${i}.jsonl" \
        $LIMIT_ARG $CHAT_ARG \
        > "$OUT/shard${i}.log" 2>&1 &
done
wait

# 2) Concatenate shards into one samples file.
SAMPLES="$OUT/samples.jsonl"
cat "$OUT"/shard*.jsonl > "$SAMPLES"
echo "Combined $(wc -l < "$SAMPLES") samples -> $SAMPLES"

# 3) EvalPlus sanitize (extracts the target function from each completion) then score.
#    sanitize writes <SAMPLES>-sanitized.jsonl
python -m evalplus.sanitize --samples "$SAMPLES"
SANITIZED="${SAMPLES%.jsonl}-sanitized.jsonl"

# --base_only: compute only the base HumanEval tests (skip HumanEval+ extra tests).
# evaluate prints "humaneval (base tests) pass@1" and writes
# <SANITIZED%.jsonl>_eval_results.json next to the samples.
python -m evalplus.evaluate --dataset humaneval --samples "$SANITIZED" --base_only | tee "$OUT/evalplus_score.txt"

echo "=================================================="
echo "EvalPlus results saved under: $OUT"
echo "  samples:   $SAMPLES"
echo "  sanitized: $SANITIZED"
echo "  score:     $OUT/evalplus_score.txt"
echo "Paper reference (Table 7): Ouro-1.4B base = 74.40 (HumanEval base)"
