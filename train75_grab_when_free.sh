#!/bin/bash
# Run the given continuation epochs on all 75 training tasks on the first card that becomes entirely free (another user takes any
# card that idles for seconds, so this polls every 30 s and starts the chain the moment a card qualifies).
#   EPOCHS_LIST="7 8" bash train75_grab_when_free.sh
set -u
: "${EPOCHS_LIST:?}"
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
MINFREE=${MINFREE:-38000}   # a 12k-context serving run peaks near 30 to 40 GB; a half-held card (24 GB) runs out of memory at the cap
CANDS=${CANDS:-"2 3 4 5"}   # card 0 is left to the BO (its shards fit next to the idle server there)
END=$(( $(date +%s) + 345600 ))
cd /home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft || exit 1
while [ "$(date +%s)" -lt "$END" ]; do
  for g in $CANDS; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "${free:-0}" -ge "$MINFREE" ] && [ "${util:-100}" -le 5 ]; then
      echo "card $g free: running epochs [$EPOCHS_LIST] on it $(date '+%m-%d %H:%M')" | tee -a "$TMP/train75_grab.log"
      if GPU=$g EPOCHS_LIST="$EPOCHS_LIST" bash epoch_train75_chain2.sh; then
        exit 0
      fi
      echo "card $g: the chain could not start (card taken again); keep looking $(date '+%m-%d %H:%M')" | tee -a "$TMP/train75_grab.log"
      sleep 120
      break
    fi
  done
  sleep 30
done
echo "no free card within 96 h for epochs [$EPOCHS_LIST]" | tee -a "$TMP/train75_grab.log"
