"""Held-out validation of the BFCL router search. Scores, under the same real early exit and official checker as the search:
  * full depth and the released gate (0.7, 0.5, 0.3, 0.25, 0.2), and
  * the search's best proposals (TOP of them, default 3): the cheapest ones that hold the search-set accuracy floor, then the best
    of the rest by the objective
on the two sets the optimiser never saw: bfcl_heldout.jsonl (the other 575 non-live items) and bfcl_live_all.jsonl (the 1,350
live items). Evaluations are cached per set (<work>/heldout, <work>/live), so the script can be restarted; after every evaluation
it rewrites <work>/heldout_validation.json, which the experiments page reads.

  cd ~/boptim-agent && BFCL_BO_GPUS=0 BFCL_SHARDS=4 ./.venv/bin/python -u <this file>        (SETS="heldout live", TOP=3)
"""
import json
import os
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
GPUS = os.environ.get("BFCL_BO_GPUS", "0")
SHARDS = int(os.environ.get("BFCL_SHARDS", "4"))
TOP = int(os.environ.get("TOP", "3"))
TOL, PENALTY = 1.7, float(os.environ.get("BFCL_PENALTY", "15.0"))  # as in bfcl_bo_run.py
SETS = {"heldout": ("bfcl_heldout.jsonl", 575), "live": ("bfcl_live_all.jsonl", 1350)}
GATES = {(1.0, 1.0, 1.0): "full depth", (0.7, 0.7, 0.7): "released gate 0.7", (0.5, 0.5, 0.5): "released gate 0.5",
         (0.3, 0.3, 0.3): "released gate 0.3", (0.25, 0.25, 0.25): "released gate 0.25", (0.2, 0.2, 0.2): "released gate 0.2"}


def key(thr):
    return tuple(round(float(v), 3) for v in thr[:3])


def load(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("status") == "ok":
                out.setdefault(key(e["thresholds"]), e)
    return out


search = load(f"{WORK}/evaluations.jsonl")
full = search.get((1.0, 1.0, 1.0))
if full is None:
    sys.exit("no full-depth evaluation on the search set; nothing to validate")
floor = full["acc_pct"] - TOL


def value(e, fl=floor):
    return 100.0 * e["exec_loops"] / 4.0 + PENALTY * max(0.0, fl - e["acc_pct"])


# proposals to validate: first the ones that hold the search-set floor, cheapest first (the claim is "baseline accuracy at less
# compute"), then, if fewer than TOP hold it, the best of the rest by the objective
cands = [e for k, e in search.items() if k not in GATES]
feasible = sorted((e for e in cands if e["acc_pct"] >= floor), key=lambda e: e["exec_loops"])
others = sorted((e for e in cands if e["acc_pct"] < floor), key=value)
props = (feasible + others)[:TOP]
# order of work: the baseline that matters most first, then proposals and gate points interleaved
order = [(1.0, 1.0, 1.0), (0.3, 0.3, 0.3)]
rest_gates = [(0.25, 0.25, 0.25), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5), (0.7, 0.7, 0.7)]
for i in range(max(len(props), len(rest_gates))):
    if i < len(props):
        order.append(key(props[i]["thresholds"]))
    if i < len(rest_gates):
        order.append(rest_gates[i])
names = dict(GATES)
for i, e in enumerate(props, 1):
    k = key(e["thresholds"])
    names[k] = f"BO best #{i} ({k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f})"
print("validating:", [names[k] for k in order], flush=True)


def write_report():
    res = {s: load(f"{WORK}/{s}/evaluations.jsonl") for s in SETS}
    rows = []
    for k in sorted(order, key=lambda k: (k not in GATES, -k[0] if k in GATES else order.index(k))):
        row = {"name": names[k], "thresholds": list(k), "search": search[k]["acc_pct"] if k in search else None,
               "search_loops": search[k]["exec_loops"] if k in search else None}
        for s in SETS:
            e = res[s].get(k)
            row[s] = e["acc_pct"] if e else None
            row[s + "_loops"] = e["exec_loops"] if e else None
        rows.append(row)
    notes = [f"Search-set floor {floor:.1f} % (full depth {full['acc_pct']:.1f} % minus {TOL})."]
    for s, label in (("heldout", "held-out"), ("live", "live")):
        f = res[s].get((1.0, 1.0, 1.0))
        if f:
            notes.append(f"Full depth on the {label} set: {f['acc_pct']:.1f} %, so the same rule puts its floor at {f['acc_pct'] - TOL:.1f} %.")
    notes.append("A searched router transfers if, on the unseen sets, it holds the floor at fewer executed loops than the released gate does.")
    json.dump({"rows": rows, "note": " ".join(notes)}, open(f"{WORK}/heldout_validation.json", "w"), indent=1)


write_report()
for s in os.environ.get("SETS", "heldout live").split():
    data, n = SETS[s]
    obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus=GPUS, n_items=n, shards=SHARDS, data_name=data, tol_pts=TOL, penalty_per_pt=PENALTY, work_dir=f"{WORK}/{s}")
    for k in order:
        obj.evaluate(search[k]["thresholds"][:3] if k in search else list(k))  # the search's exact thresholds, not the rounded key
        write_report()
    print(f"set {s} done", flush=True)
print("validation finished", flush=True)
