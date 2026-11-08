#!/bin/bash
# Restart a Dory-suite part on a card we hold, without letting the card go idle (another user's holder grabs idle cards within
# seconds): kill the running dory/ifb/bench processes on this GPU's run and immediately relaunch.
#   GPU=1 PART=ifb BATCH_FULL=6 bash restart_dory_gpu.sh
set -u
: "${GPU:?}"; : "${PART:?}"
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
cd "$WT" || exit 1
if [ "$PART" = ifb ]; then PAT='ifbench.jsonl|bash ifb_points.sh'; else PAT='gpqa_diamond.jsonl|bash bench_pipeline.sh|feat_gpqa_think'; fi
# kill shells first (so nothing launches a next step), then the python processes; only processes whose command line names this part
for p in $(pgrep -f 'bash dory_suite_gpu.sh'); do
  if tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -q "PART=$PART"; then kill -TERM "$p" 2>/dev/null && echo "TERM dory $p"; fi
done
pkill -TERM -f "$PAT" && echo "TERM $PAT"
sleep 2
pkill -KILL -f "$PAT" 2>/dev/null
[ "$PART" = ifb ] && mv "$TMP/bench_ifbench_think.txt" "$TMP/bench_ifbench_think_restarted_$(date +%H%M).txt" 2>/dev/null
[ "$PART" = gpqa ] && mv "$TMP/bench_gpqa_think.txt" "$TMP/bench_gpqa_think_restarted_$(date +%H%M).txt" 2>/dev/null
echo "relaunch $PART on GPU $GPU $(date '+%m-%d %H:%M')"
GPU=$GPU PART=$PART BATCH_FULL=${BATCH_FULL:-6} bash dory_suite_gpu.sh
