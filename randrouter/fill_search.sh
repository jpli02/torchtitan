#!/bin/bash
# Threshold search on the proxy in the REAL objective's units (ouro_tf_calib: 1; randrouter/proxy_value.py), for the released
# model with (a) the calibrated cache fill and (b) the decoder's plain copy fill (the space of the earlier real searches).
# Question (user, 2026-11-20: keep the original setting, how can the LiveCodeBench result be improved): does the same search,
# the same objective and the same baseline find a better router once the exits are served with the calibrated fill?
#   GPU=8 WAIT_PID=<pid> bash randrouter/fill_search.sh            results: bo_tf/{fill24_calib,base_copy_calib}/evaluations.jsonl
# Gaussian process, 10 random points then 35 model-based ones, no warm start; about 30 s per evaluation on an idle card.
set -u
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
BO=/home/jli199/boptim-agent
ROOT=/home/jli199/boptim_scratch/randrouter/bo_tf
GPU=${GPU:-8}
ITERS=${ITERS:-45}
MEM=${MEM_GB:-8.5}
SYSTEMS=${SYSTEMS:-fill24_calib base_copy_calib}
FILL=/home/jli199/boptim_scratch/randrouter/fill_maps_lcb_n24.pt
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ -n "${WAIT_PID:-}" ]; then until ! kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi
"$BO/.venv/bin/python" "$WT/randrouter/setup_tf_objective.py" > /dev/null 2>&1
cd "$BO" || exit 1
for name in $SYSTEMS; do
  W=$ROOT/$name
  if [ "$(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l)" -ge "$ITERS" ]; then continue; fi
  rm -rf "$W"; mkdir -p "$W/runs"
  CFG=configs/ouro_tf_${name}.yaml
  {
    echo "# threshold search on the proxy in the real objective's units (randrouter/fill_search.sh): system $name"
    echo "replicates: 1"
    echo "wandb:"
    echo "  enabled: false"
    echo "experiments:"
    echo "  - optim: gp_hedge"
    echo "    objective:"
    echo "      type: ouro_tf_router"
    echo "      ouro_tf_n: 99"
    echo "      ouro_tf_calib: 1"
    echo "      ouro_tf_mem_gb: $MEM"
    echo "      ouro_tf_work_dir: $W"
    if [ "$name" = fill24_calib ]; then echo "      ouro_tf_fill_maps: $FILL"; fi
    echo "    batch_size: 1"
    echo "    max_iter: $ITERS"
    echo "    resume: false"
    echo "    resume_dir: $W/runs"
    echo "    optim_seed: 1"
    echo "    objective_seed: 0"
    echo "    n_initial: 10"
  } > "$CFG"
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -n 1)" -ge 8800 ]; do sleep 30; done
  CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python -u run_experiment.py --config "$CFG" --log-prefix "tf_${name}" > "$W/bo.log" 2>&1
  echo "$(date '+%m-%d %H:%M') $name: $(cat "$W/evaluations.jsonl" 2>/dev/null | wc -l) evaluations"
done
echo "FILL SEARCH done $(date '+%m-%d %H:%M')"
