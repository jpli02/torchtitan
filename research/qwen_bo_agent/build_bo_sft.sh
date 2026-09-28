#!/bin/bash
# Assemble the Qwen BO-agent SFT set from raw trajectories.
#   reasoning rows  : CIFAR (gpt-5-mini's own run, 15) + gp10r_* = gp_hedge's
#                     points with Qwen3-8B-written analysis/plan ($WT/research/qwen_bo_agent/rationalize.py)
#   points-only rows: the gp10 trajectories that were not rationalized,
#                     --no-explanation JSON contract
# Rows are the optimizer's exact prompt (system + user) -> JSON reply.
#
# The teacher's OWN trajectories (qwen3-8b_*) are deliberately excluded: its
# point policy is worse than random search on fourier2d (normalised regret
# 0.33 vs random 0.27 on 18 held-out seeds); gp_hedge with its default 10
# initial points is not (0.24, beats random 13/20). gp10 is the policy to
# imitate; the n_initial=3 variant (gp_*) also lost to random (0.30).
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
RAW=${RAW:-/home/jli199/boptim_scratch/bo_traj}
OUTDIR=${OUTDIR:-/home/jli199/boptim_scratch/bo_sft}
PY=/home/jli199/torchtitan/.venv/bin/python
mkdir -p "$OUTDIR"
cd "$WT" || exit 1

"$PY" $WT/research/qwen_bo_agent/bo_replay_format.py ${BOPTIM_REPO:-$WT/research/boptim-agent}/data/gpt_cifar_3d_seed0000.json \
  "$RAW"/gp10r_fourier2d/*.json "$RAW"/gp10r_rosenbrock/*.json "$RAW"/gp10r_sumsq/*.json \
  --out "$OUTDIR/reasoning.jsonl"

# fourier2d seeds 80-199, rosenbrock/sumsq seeds 15-29 (the rationalized ones
# are 0-79 / 0-14), so no trajectory appears under both contracts.
"$PY" $WT/research/qwen_bo_agent/bo_replay_format.py "$RAW"/gp10_fourier2d/*seed00[89]?.json "$RAW"/gp10_fourier2d/*seed01??.json \
  "$RAW"/gp10_rosenbrock/*seed001[5-9].json "$RAW"/gp10_rosenbrock/*seed002?.json \
  "$RAW"/gp10_sumsq/*seed001[5-9].json "$RAW"/gp10_sumsq/*seed002?.json \
  --no-explanation --out "$OUTDIR/points_only.jsonl"

# shuffle with a fixed seed so the stream order is reproducible
cat "$OUTDIR/reasoning.jsonl" "$OUTDIR/points_only.jsonl" | shuf --random-source=<(yes 42) > "$OUTDIR/train.jsonl"
echo "train.jsonl rows: $(wc -l < "$OUTDIR/train.jsonl")  (reasoning $(wc -l < "$OUTDIR/reasoning.jsonl"), points-only $(wc -l < "$OUTDIR/points_only.jsonl"))"
"$PY" $WT/research/qwen_bo_agent/check_qwen_template.py "$OUTDIR/train.jsonl" 2>&1 | grep -v Warning | head -2
