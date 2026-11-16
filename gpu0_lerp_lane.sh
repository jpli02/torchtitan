#!/bin/bash
# Learning-rate debug on GPU 0: weight-space line search between v3 and the fine-tuned checkpoints (lr_debug_lerp_eval.py),
# in a gap of the BFCL validation queue (lower priority since the user made the LR debug first priority).
#   1. wait until the BFCL evaluation now running has written its result (a clean router boundary), then stop that queue;
#   2. two line searches at a time (12 GB each, next to another user's 10 to 14 GB on this card);
#   3. restart the BFCL queue (boptim_lcb/bo3_followup4.sh skips what is cached).
# Start detached:  setsid nohup bash gpu0_lerp_lane.sh >> $TMP/gpu0_lerp_lane.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
S=/home/jli199/boptim_scratch
V3=$S/ouro_tb80sft26b_v3_clean
ALPHAS=${ALPHAS:-0,0.25,0.5,0.75,1,1.5}
cd "$WT" || exit 1
N0=$(grep -c '^LAB ' "$TMP/lab_eval_live.log")
END=$(( $(date +%s) + 7200 ))
until [ "$(grep -c '^LAB ' "$TMP/lab_eval_live.log")" -gt "$N0" ] || ! pgrep -u "$(id -un)" -f 'boptim_lcb/lab_eval.py' > /dev/null; do
  [ "$(date +%s)" -gt "$END" ] && break
  sleep 10
done
echo "BFCL queue paused at a router boundary $(date '+%m-%d %H:%M') ($(grep -c '^LAB ' "$TMP/lab_eval_live.log") live results)"
python3 -I kill_by_pattern.py "bo3_followup4.sh" | tail -n 1
python3 -I kill_by_pattern.py "boptim_lcb/lab_eval.py" | tail -n 1
python3 -I kill_by_pattern.py "bfcl_stage/bfcl_" | tail -n 1
sleep 20
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -n 1)
echo "gpu 0 free ${free} MiB"
if [ "${free:-0}" -ge 28000 ]; then
  run() {  # run TAG B_DIR
    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True "$PY" -u lr_debug_lerp_eval.py "$V3" "$2" "$ALPHAS" \
      "$TMP/lrval_small_rendered.jsonl" "$1" > "$TMP/lerp_$1.log" 2>&1
  }
  run r1 $S/ouro_tvary_r1_clean &
  run ep4 $S/ouro_tvary_r1_4ep_ep4_clean &
  wait
  run ep8 $S/ouro_tvary_r1_8ep_ep8_clean &
  run ep1 $S/ouro_tvary_r1_4ep_ep1_clean &
  wait
  grep -h -A 8 '^LERP ' "$TMP"/lerp_r1.log "$TMP"/lerp_ep1.log "$TMP"/lerp_ep4.log "$TMP"/lerp_ep8.log | cut -c1-200
else
  echo "gpu 0 too full for the line search; skipped"
fi
echo "LERP lane done $(date '+%m-%d %H:%M'); restarting the BFCL queue"
setsid nohup bash boptim_lcb/bo3_followup4.sh >> "$TMP/bo3_followup.log" 2>&1 < /dev/null &
