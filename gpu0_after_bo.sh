#!/bin/bash
# GPU 0 after the pooled LiveCodeBench BO finishes there: (1) the new-benchmark probes (about 4 h), then (2) epoch 8 on all 75
# training tasks. Cards 2 to 5 belong to another user's holder (24 GB left at best, and their job grows without notice: the probe
# chain lost 7 of 9 points there), so GPU 0 (39 GB next to an idle server) and GPU 1 are the only cards we can rely on.
#   bash gpu0_after_bo.sh
set -u
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
END=$(( $(date +%s) + 172800 ))
until grep -q 'Completed 1 experiment' "$TMP/lcb_bo_pooled_cal32.log" 2>/dev/null; do
  [ "$(date +%s)" -gt "$END" ] && { echo "BO did not finish within 48 h"; exit 1; }
  sleep 60
done
echo "BO finished; probes on GPU 0 start $(date '+%m-%d %H:%M')" | tee -a "$TMP/gpu0_chain.log"
sleep 30
# the six MMLU-Redux points that failed on the borrowed card, then BFCL if its tooling is in place, then the thinking probes
GPU=0 N=100 TB=6 MB=16 MMLU_POINTS="k2 q0.5 q0.3 q0.1 steps0.3-0.5-0.5 steps0.1-0.5-0.5" LCB_POINTS="k4" OTHER_POINTS="" bash more_bench_chain.sh > "$TMP/gpu0_probes_a.log" 2>&1
[ -f bfcl_points.sh ] && GPU=0 bash bfcl_points.sh > "$TMP/gpu0_probes_bfcl.log" 2>&1
GPU=0 N=100 TB=6 MMLU_POINTS="" LCB_POINTS="q0.3" OTHER_POINTS="k4" bash more_bench_chain.sh > "$TMP/gpu0_probes_b.log" 2>&1
echo "probes done; epoch 8 on GPU 0 starts $(date '+%m-%d %H:%M')" | tee -a "$TMP/gpu0_chain.log" "$TMP/train_tvary_r1_8ep.txt"
GPU=0 MINFREE=36000 EPOCHS_LIST="8" bash epoch_train75_chain2.sh
