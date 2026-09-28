#!/bin/bash
# Sequential retry of the 16 skill26 runs that OOM'd (no checkpoint) when the
# co-tenant grew on a shared card. One skill at a time, up to 3 tries each,
# each waiting for a >=40GB-free card; sequential maximises the chance of a
# stable exclusive window vs the parallel orchestrator.
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
QUEUE=/home/jli199/.claude/jobs/c0d2da0a/tmp/skill_queue.tsv
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
LOG=$TMP/rerun_skill26_seq.log
: > "$LOG"
cd "$WT" || exit 1
ABORTED="cron-svc crypto data-files fasttext games git huggingface jupyter python-env pytorch qemu sci sqlite textproc tmux vim"
while IFS=$'\t' read -r skill ntest tasks; do
  case " $ABORTED " in *" $skill "*) : ;; *) continue ;; esac
  capped=$(echo "$tasks" | tr ' ' '\n' | head -n 8 | tr '\n' ' ')
  for try in 1 2 3; do
    echo "rerun26 $skill try $try $(date '+%m-%d %H:%M')" | tee -a "$LOG"
    SKILL="$skill" TASKS="$capped" STEPS=1000 ATT=2 SEQLEN=4096 MINFREE_TRAIN=40000 \
      bash $WT/research/terminal_sft/train/run_skill_26b.sh >> "$TMP/skill26_${skill}_rerun.log" 2>&1
    if grep -q "pass@k on skill" "$TMP/skill26_${skill}.txt" 2>/dev/null; then
      echo "  $skill OK on try $try" | tee -a "$LOG"; break
    fi
    echo "  $skill failed try $try (likely OOM), cleaning + sleep 300" | tee -a "$LOG"
    pkill -9 -f "ouro_skill26_${skill}" 2>/dev/null; sleep 300
  done
done < "$QUEUE"
echo "rerun_skill26_seq done $(date '+%m-%d %H:%M')" | tee -a "$LOG"
