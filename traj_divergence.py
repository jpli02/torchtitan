"""Where do r1's evaluation trajectories depart from the training trajectories of the same task?

For each task in a run dir: the training row(s) of that task (first user message matched to the pool task.yaml
instruction, as in underfit_pertask.py) give a sequence of assistant turns -> keystrokes; the evaluation episodes give
another. Reports the first turn whose keystrokes differ (after whitespace normalisation), the share of training
commands that appear anywhere in the evaluation trajectory (recall), lengths, and the outcome.
Usage: python traj_divergence.py RUN [MIXED.jsonl]"""
import collections
import glob
import json
import os
import re
import statistics
import sys

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
POOL = "/home/jli199/boptim_scratch/tb_variants/pool"
RUNS = "/srv/scratch/jli199/tb_runs"
run = sys.argv[1]
mixed = sys.argv[2] if len(sys.argv) > 2 else f"{TMP}/tvary_r1_mixed.jsonl"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def keystrokes(text):
    """Keystrokes of the commands in an assistant turn (JSON action), normalised; None if no action parses."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        return None
    for cand in (text[i:j + 1],):
        try:
            d = json.loads(cand)
        except Exception:
            try:
                d = json.loads(re.sub(r",\s*}", "}", cand))
            except Exception:
                return None
        cmds = d.get("commands") or []
        ks = " && ".join(norm(c.get("keystrokes", "")) for c in cmds if isinstance(c, dict))
        return ks if ks else ("<task_complete>" if d.get("task_complete") else "")
    return None


instr = {}
for ty in glob.glob(f"{POOL}/*/task.yaml"):
    txt = open(ty, errors="replace").read()
    m = re.search(r"instruction:\s*\|?-?\s*\n?(.*?)(?:\n[a-z_]+:|\Z)", txt, re.S)
    if m:
        instr[os.path.basename(os.path.dirname(ty))] = norm(m.group(1))[:200]

train = collections.defaultdict(list)
for line in open(mixed):
    if not line.strip():
        continue
    row = json.loads(line)
    first = norm(next((m["content"] for m in row["messages"] if m["role"] == "user"), ""))
    best, best_len = None, 0
    for t, key in instr.items():
        if key and key in first and len(key) > best_len:
            best, best_len = t, len(key)
    if best:
        train[best].append([keystrokes(m["content"]) for m in row["messages"] if m["role"] == "assistant"])

rows = []
for tdir in sorted(glob.glob(f"{RUNS}/{run}/*/*/")):
    try:
        res = json.load(open(f"{tdir}/results.json"))
    except Exception:
        continue
    task = res["task_id"]
    if task not in train:
        continue
    eps = sorted(glob.glob(f"{tdir}/agent-logs/episode-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    ev = []
    for e in eps:
        try:
            ev.append(keystrokes(open(f"{e}/response.txt", errors="replace").read()))
        except OSError:
            pass
    outcome = "OK" if res.get("is_resolved") else ("timeout" if res.get("failure_mode") == "agent_timeout" else "fail")
    best = None
    for tr in train[task]:  # compare against the training row that matches longest
        div = 0
        while div < min(len(tr), len(ev)) and tr[div] is not None and tr[div] == ev[div]:
            div += 1
        tr_set = {k for k in tr if k}
        recall = (sum(1 for k in tr_set if k in set(x for x in ev if x)) / len(tr_set)) if tr_set else 0.0
        cand = (div, recall, len(tr), len(ev))
        if best is None or cand > best:
            best = cand
    rows.append((outcome, task, best))

print(f"{run}: {len(rows)} trials with a training trajectory")
print(f"{'task':44s} {'outcome':8s} {'same-prefix turns':>17} {'train turns':>11} {'eval turns':>10} {'cmd recall':>10}")
for outcome, task, (div, recall, ntr, nev) in sorted(rows, key=lambda r: (r[0], r[1])):
    print(f"{task[:44]:44s} {outcome:8s} {div:17d} {ntr:11d} {nev:10d} {recall:10.2f}")
for outcome in ("OK", "fail", "timeout"):
    sel = [b for o, _, b in rows if o == outcome]
    if sel:
        print(f"{outcome:8s} n={len(sel):2d}  mean same-prefix turns {statistics.mean(b[0] for b in sel):.2f}  "
              f"first-turn identical {sum(b[0] >= 1 for b in sel)}/{len(sel)}  mean command recall {statistics.mean(b[1] for b in sel):.2f}  "
              f"train turns {statistics.mean(b[2] for b in sel):.1f} eval turns {statistics.mean(b[3] for b in sel):.1f}")
