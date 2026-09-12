#!/bin/bash
# Assemble the Qwen BO-agent SFT set from raw trajectories:
#   reasoning rows  : CIFAR (gpt-5-mini, 15) + local Qwen3-14B teacher runs
#   points-only rows: gp_hedge runs (free), --no-explanation JSON contract
# Rows are the optimizer's exact prompt (system + user) -> JSON reply.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
RAW=${RAW:-/home/jli199/boptim_scratch/bo_traj}
OUTDIR=${OUTDIR:-/home/jli199/boptim_scratch/bo_sft}
PY=/home/jli199/torchtitan/.venv/bin/python
mkdir -p "$OUTDIR"
cd "$WT" || exit 1

"$PY" bo_replay_format.py /home/jli199/boptim-agent/data/gpt_cifar_3d_seed0000.json \
  "$RAW"/qwen3-*_fourier2d/*.json "$RAW"/qwen3-*_rosenbrock/*.json "$RAW"/qwen3-*_sumsq/*.json \
  --out "$OUTDIR/reasoning.jsonl"
# gp10_* = gp_hedge with its default 10 random initial points. The n_initial=3
# variant (gp_*) lost to random search on fourier2d (regret 0.30 vs 0.27 on
# 20 held-out seeds); gp10 beats random (0.24, 13/20), so imitate that one.
"$PY" bo_replay_format.py "$RAW"/gp10_fourier2d/*.json "$RAW"/gp10_rosenbrock/*.json "$RAW"/gp10_sumsq/*.json \
  --no-explanation --out "$OUTDIR/points_only.jsonl"

# shuffle with a fixed seed so the stream order is reproducible
cat "$OUTDIR/reasoning.jsonl" "$OUTDIR/points_only.jsonl" | shuf --random-source=<(yes 42) > "$OUTDIR/train.jsonl"
echo "train.jsonl rows: $(wc -l < "$OUTDIR/train.jsonl")  (reasoning $(wc -l < "$OUTDIR/reasoning.jsonl"), points-only $(wc -l < "$OUTDIR/points_only.jsonl"))"
"$PY" check_qwen_template.py "$OUTDIR/train.jsonl" 2>&1 | grep -v Warning | head -2
