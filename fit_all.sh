#!/bin/bash
# Wait for the rendered token dumps, then run the teacher-forced fit evals:
#   GPU 2: v3 on r1+r2 rows (fit BEFORE continuation), then r1+r2 ckpt on its rows
#   GPU 5: take-turns r1 ckpt on its rows, then teacher-cont r1 ckpt on its rows
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/boptim-agent/.venv/bin/python
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
S=/home/jli199/boptim_scratch
cd "$WT" || exit 1
END=$(( $(date +%s) + 7200 ))
n=0
while [ "$(date +%s)" -lt "$END" ]; do
  n=$(grep -l '^wrote ' "$TMP/render_tcont.log" "$TMP/render_r1.log" "$TMP/render_r12.log" 2>/dev/null | wc -l)
  [ "$n" -eq 3 ] && break
  sleep 60
done
[ "$n" -eq 3 ] || { echo "ABORT: render incomplete after 2h"; exit 1; }
echo "renders ready $(date '+%m-%d %H:%M'); starting fit evals"
(
  CUDA_VISIBLE_DEVICES=2 "$PY" -u fit_eval.py "$S/ouro_tb80sft26b_v3_clean" "$TMP/tvary_r12_rendered.jsonl" "$TMP/fit_v3_on_r12.json" > "$TMP/fit_v3_on_r12.log" 2>&1
  CUDA_VISIBLE_DEVICES=2 "$PY" -u fit_eval.py "$S/ouro_tvary_r12_clean" "$TMP/tvary_r12_rendered.jsonl" "$TMP/fit_r12.json" > "$TMP/fit_r12.log" 2>&1
) &
(
  CUDA_VISIBLE_DEVICES=5 "$PY" -u fit_eval.py "$S/ouro_tvary_r1_clean" "$TMP/tvary_r1_rendered.jsonl" "$TMP/fit_r1.json" > "$TMP/fit_r1.log" 2>&1
  CUDA_VISIBLE_DEVICES=5 "$PY" -u fit_eval.py "$S/ouro_tcont_r1_clean" "$TMP/tcont_r1_rendered.jsonl" "$TMP/fit_tcont.json" > "$TMP/fit_tcont.log" 2>&1
) &
wait
echo "FIT_ALL done $(date '+%m-%d %H:%M')"
for f in v3_on_r12 r12 r1 tcont; do
  echo "--- $f"
  grep -vE 'Warning|warn|deprecated' "$TMP/fit_$f.log" | tail -3
done
