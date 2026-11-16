"""Learning-rate debug: one rendered validation file for teacher-forced loss per checkpoint (fit_eval.py format).

Groups (the 'group' field):
  oracle_test    the 12 oracle trajectories of the TEST tasks (never trained on by any non-oracle checkpoint)
  r2_unseen      round-2 take-turns trajectories on the 75 TRAINING tasks that are not in the r1 training mix (same tasks, new runs)
  replay_unseen  rows of v3's own training data that the r1 mix did NOT replay (forgetting of the earlier training)
  train_new      r1 trajectories the 4- and 8-epoch runs trained on (fit)
  train_replay   v3 rows the r1 mix replayed (fit)

  python lr_debug_valset.py            # writes <job tmp>/lrval_rendered.jsonl
"""
import hashlib
import json
import random

T = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
N = {"r2_unseen": 30, "replay_unseen": 20, "train_new": 12, "train_replay": 12}


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def h(r):
    return hashlib.md5(json.dumps(r["ids"]).encode()).hexdigest()


r1, r12, oracle = load(f"{T}/tvary_r1_rendered.jsonl"), load(f"{T}/tvary_r12_rendered.jsonl"), load(f"{T}/oracle12_rendered.jsonl")
seen = {h(r) for r in r1}
rng = random.Random(11)
pools = {"r2_unseen": [r for r in r12 if h(r) not in seen and r["group"] == "new"],
         "replay_unseen": [r for r in r12 if h(r) not in seen and r["group"] == "replay"],
         "train_new": [r for r in r1 if r["group"] == "new"],
         "train_replay": [r for r in r1 if r["group"] == "replay"]}
out = [dict(r, group="oracle_test") for r in oracle]
for g, pool in pools.items():
    pick = rng.sample(pool, min(N[g], len(pool)))
    out += [dict(r, group=g) for r in pick]
with open(f"{T}/lrval_rendered.jsonl", "w") as f:
    for k, r in enumerate(out):
        f.write(json.dumps(dict(r, i=k)) + "\n")
for g in ["oracle_test"] + list(pools):
    rs = [r for r in out if r["group"] == g]
    print(f"{g:14s} rows {len(rs):3d}  tokens {sum(len(r['ids']) for r in rs):8,d}  supervised {sum(sum(1 for x in r['labels'] if x != -100) for r in rs):8,d}")
print("total tokens", sum(len(r["ids"]) for r in out))
