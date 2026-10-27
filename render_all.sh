#!/bin/bash
# Render the three mixed SFT files to token/label dumps, in parallel (CPU only).
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
PY=/home/jli199/torchtitan/.venv/bin/python
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
V3=/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl
cd "$WT" || exit 1
"$PY" -u render_sft_rows.py "$V3" "$TMP/tcont_r1_mixed.jsonl" "$TMP/tcont_r1_rendered.jsonl" 720 > "$TMP/render_tcont.log" 2>&1 &
"$PY" -u render_sft_rows.py "$V3" "$TMP/tvary_r1_mixed.jsonl" "$TMP/tvary_r1_rendered.jsonl" 896 > "$TMP/render_r1.log" 2>&1 &
"$PY" -u render_sft_rows.py "$V3" "$TMP/tvary_r12_mixed.jsonl" "$TMP/tvary_r12_rendered.jsonl" 1568 > "$TMP/render_r12.log" 2>&1 &
wait
echo "RENDER_ALL done $(date '+%m-%d %H:%M')"
for f in tcont r1 r12; do echo "--- $f"; grep -vE 'Warning|warn' "$TMP/render_$f.log" | tail -4; done
