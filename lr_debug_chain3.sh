#!/bin/bash
# Learning-rate debug on GPU 1, part 3 (replaces the rest of lr_debug_chain2.sh, whose 2e-5 constant arm is postponed):
#   after the 5e-6 constant-LR run has ended, Terminal-Bench TEST evaluations (11 tasks x 2 attempts, the protocol of the
#   epoch 5-8 evaluations) of the candidates the held-out loss points to, in this order:
#     1. the 5e-6 run's own final checkpoint (step 1072), if its last validation is still good
#        (test-task loss within 3 % of v3's and unseen-run loss under 0.56; otherwise it is skipped here);
#     2. the checkpoint halfway between v3 and take-turns r1 (lr_debug_lerp_build.py).
# Start detached:  setsid nohup bash lr_debug_chain3.sh >> $TMP/lr_debug_chain3.log 2>&1 < /dev/null &
WT=/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft
TMP=/home/jli199/.claude/jobs/c0d2da0a/tmp
PY=/home/jli199/boptim-agent/.venv/bin/python
S=/home/jli199/boptim_scratch
BASE=/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking
ALL11="hello-world fix-permissions extract-safely csv-to-parquet processing-pipeline heterogeneous-dates chess-best-move fibonacci-server crack-7z-hash.easy cron-broken-network sanitize-git-repo"
cd "$WT" || exit 1
END=$(( $(date +%s) + 28800 ))
until grep -q 'lr sweep lr5e-6_const exited' "$TMP/lr_debug_chain2.log" 2>/dev/null \
   && [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | head -n 1)" -ge 44000 ]; do
  [ "$(date +%s)" -gt "$END" ] && { echo "LR-DEBUG part 3 gave up waiting $(date '+%m-%d %H:%M')"; exit 1; }
  sleep 30
done
"$PY" -I lr_debug_valcurve.py lr5e-6_const
echo "LR-DEBUG 5e-6 curve done $(date '+%m-%d %H:%M')"

CANDS=""
src=$S/ouro_lrsweep_lr5e-6_const/checkpoint/step-1072
clean=$S/ouro_lrsweep_lr5e-6_const_clean
verdict=$("$PY" -I -c "
import json
rows = [json.loads(l) for l in open('$TMP/valprobe_lr5e-6_const.jsonl') if l.strip()]
last = max(r['step'] for r in rows)
get = lambda s, g: next(r['ce_final'] for r in rows if r['step'] == s and r['group'] == g)
ok = last >= 1072 and get(last, 'oracle_test') <= 1.03 * get(0, 'oracle_test') and get(last, 'r2_unseen') < 0.56
print('ok' if ok else 'skip', 'step', last, 'test-task loss %.3f (start %.3f)' % (get(last, 'oracle_test'), get(0, 'oracle_test')), 'unseen-run loss %.3f' % get(last, 'r2_unseen'))
")
echo "5e-6 final checkpoint: $verdict"
if [ "${verdict%% *}" = "ok" ] && ls "$src"/model*.safetensors > /dev/null 2>&1; then
  case "$clean" in /home/jli199/boptim_scratch/ouro_lrsweep_*_clean) rm -rf "$clean" ;; esac
  mkdir -p "$clean"
  cp "$src"/model*.safetensors "$clean"/
  [ -f "$src/model.safetensors.index.json" ] && cp "$src/model.safetensors.index.json" "$clean"/
  for f in config.json configuration_ouro.py modeling_ouro.py tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do cp "$BASE/$f" "$clean/$f"; done
  CANDS="lr5e-6_s1072=$clean"
fi
CANDS="$CANDS lerp_r1_a050=$S/ouro_lerp_r1_a050_clean"
echo "TB candidates: $CANDS"

export OURO_SEQ_CAP=12000
port=8181
for pair in $CANDS; do
  tag=${pair%%=*}; dir=${pair#*=}
  ls "$dir"/model*.safetensors > /dev/null 2>&1 || { echo "no weights in $dir, skipped"; continue; }
  left=$(docker ps -a --format '{{.Names}}' | grep -- "-of-2-${tag}_eval\$" | tr '\n' ' ')
  [ -n "$left" ] && echo "WARNING: left-over containers for ${tag}_eval, their trials would reuse them: $left"
  echo "TB test evaluation of $tag starts $(date '+%m-%d %H:%M')"
  GPU=1 GATE=0 CKPT="$dir" NAME=${tag}_eval PORT=$port ATT=2 MINFREE=${MINFREE:-38000} TASKS="$ALL11" \
    bash eval_gated.sh > "$TMP/${tag}_eval.log" 2>&1
  python3 -I tb_test_summary.py "${tag}_eval"
  echo "LR-DEBUG TB evaluation of $tag done $(date '+%m-%d %H:%M')"
  port=$((port + 1))
done
echo "LR-DEBUG part 3 done $(date '+%m-%d %H:%M')"
