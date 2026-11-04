"""Per-task fit of the r1 (and ep4) checkpoints on their training rows, joined with the training-task evaluations.

Rows have no task id: each new row's first user message embeds the task instruction, which is matched against the
pool tasks' task.yaml instructions. Fit comes from fit_eval outputs (per-row supervised token accuracy / CE under the
4,096-token training windows); outcomes from the Terminal-Bench run dirs.
Usage: python underfit_pertask.py"""
import collections
import glob
import hashlib
import json
import os
import re
import statistics

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
POOL = "/home/jli199/boptim_scratch/tb_variants/pool"
RUNS = "/srv/scratch/jli199/tb_runs"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


# task instructions
instr = {}
for ty in glob.glob(f"{POOL}/*/task.yaml"):
    txt = open(ty, errors="replace").read()
    m = re.search(r"instruction:\s*\|?-?\s*\n?(.*?)(?:\n[a-z_]+:|\Z)", txt, re.S)
    if m:
        instr[os.path.basename(os.path.dirname(ty))] = norm(m.group(1))[:400]

rows = [json.loads(l) for l in open(f"{TMP}/tvary_r1_mixed.jsonl") if l.strip()]


def task_of(row):
    first = norm(next((m["content"] for m in row["messages"] if m["role"] == "user"), ""))
    best, best_len = None, 0
    for t, ins in instr.items():
        key = ins[:200]
        if key and key in first and len(key) > best_len:
            best, best_len = t, len(key)
    return best


fit = {}
for name, path in [("r1", f"{TMP}/fit_r1.json"), ("ep4", f"{TMP}/fit_r1_4ep_ep4.json"), ("v3", f"{TMP}/fit_v3_on_r12.json")]:
    if os.path.exists(path):
        fit[name] = {r["i"]: r for r in json.load(open(path))["rows"]}

# outcomes per task
def outcomes(run):
    out = {}
    for f in glob.glob(f"{RUNS}/{run}/*/*/results.json"):
        d = json.load(open(f))
        out[d["task_id"]] = "OK" if d.get("is_resolved") else ("timeout" if d.get("failure_mode") == "agent_timeout" else "fail")
    return out


oc = {run: outcomes(run) for run in ("train22_r1_1800", "train75_r1", "train75_ep4", "train75_v3")}

per_task = collections.defaultdict(list)
unmapped = 0
for i, row in enumerate(rows):
    if "r1" in fit and fit["r1"].get(i, {}).get("group") != "new":
        continue
    t = task_of(row)
    if t is None:
        unmapped += 1
        continue
    per_task[t].append(i)
print(f"new rows mapped to {len(per_task)} tasks; unmapped {unmapped}")

print(f"\n{'task':44s} {'rows':>4} {'r1 acc':>7} {'r1 CE':>6} {'ep4 acc':>7} {'tok/row':>7}  r1@1800  r1@900  ep4@900  v3@900")
lines = []
for t, idx in per_task.items():
    acc_r1 = statistics.mean(fit["r1"][i]["acc"] for i in idx) if "r1" in fit else float("nan")
    ce_r1 = statistics.mean(fit["r1"][i]["ce"] for i in idx) if "r1" in fit else float("nan")
    acc_ep4 = statistics.mean(fit["ep4"][i]["acc"] for i in idx) if "ep4" in fit and all(i in fit["ep4"] for i in idx) else float("nan")
    toks = statistics.mean(fit["r1"][i]["tokens"] for i in idx) if "r1" in fit else 0
    o = [oc[r].get(t, "-") for r in ("train22_r1_1800", "train75_r1", "train75_ep4", "train75_v3")]
    lines.append((o[0], t, len(idx), acc_r1, ce_r1, acc_ep4, toks, o))
for o0, t, n, a, c, a4, toks, o in sorted(lines, key=lambda x: (x[0], x[1])):
    print(f"{t[:44]:44s} {n:4d} {a:7.3f} {c:6.3f} {a4:7.3f} {toks:7.0f}  {o[0]:7s}  {o[1]:6s}  {o[2]:7s}  {o[3]:6s}")

# summary: fit of tasks r1 solved vs not at 1,800 s (only tasks that were evaluated there)
for label, run in [("r1 @1800 s", "train22_r1_1800"), ("ep4 @900 s", "train75_ep4"), ("r1 @900 s", "train75_r1")]:
    groups = collections.defaultdict(list)
    for o0, t, n, a, c, a4, toks, o in lines:
        res = oc[run].get(t)
        if res:
            groups[res].append((a, a4, toks, n))
    if groups:
        print(f"\n{label}:")
        for res, vals in sorted(groups.items()):
            print(f"  {res:8s} tasks={len(vals):2d}  r1 token acc {statistics.mean(v[0] for v in vals):.3f}  ep4 acc {statistics.mean(v[1] for v in vals if v[1] == v[1]) if any(v[1] == v[1] for v in vals) else float('nan'):.3f}  "
                  f"tokens/row {statistics.mean(v[2] for v in vals):.0f}  rows/task {statistics.mean(v[3] for v in vals):.1f}")
