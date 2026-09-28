#!/bin/bash
# Per-skill 2.6B sweep. Each skill waits for a GENUINELY free card (>=40GB) so
# it runs exclusive at seq 4096 -- sharing a card with the growing co-tenant
# OOMs (sqlite validation stepped at 25.8GB then the neighbour pushed it over).
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
QUEUE=/home/jli199/.claude/jobs/c0d2da0a/tmp/skill_queue.tsv
LOG=/home/jli199/.claude/jobs/c0d2da0a/tmp/run_all_skills_26b.log
STAGGER=${STAGGER:-200}
CAP=${CAP:-8}
: > "$LOG"
cd "$WT" || exit 1
while IFS=$'\t' read -r skill ntest tasks; do
  capped=$(echo "$tasks" | tr ' ' '\n' | head -n "$CAP" | tr '\n' ' ')
  echo "launch26 $skill (eval capped $CAP): $capped" | tee -a "$LOG"
  SKILL="$skill" TASKS="$capped" STEPS=1000 ATT=2 SEQLEN=4096 MINFREE_TRAIN=40000 \
    nohup bash $WT/research/terminal_sft/train/run_skill_26b.sh > /home/jli199/.claude/jobs/c0d2da0a/tmp/skill26_${skill}_launch.log 2>&1 &
  sleep "$STAGGER"
done < "$QUEUE"
echo "all skill26 launched $(date '+%m-%d %H:%M')" | tee -a "$LOG"
wait
echo "all skill26 finished $(date '+%m-%d %H:%M')" | tee -a "$LOG"
