"""Warm start for the pooled LiveCodeBench BO: write the optimizer's resume-state file (main.py --resume) with every point whose
pooled objective is already known, so the surrogate starts from the released gates and the slice-run routers instead of
random points.

Records = the real pooled evaluations in <work_dir>/evaluations.jsonl (calibrated speed) + designed points whose accuracy over
all 342 problems is pooled from the three wall-clock slices (0..119 from the slice-run BO or wall-clock table, 120..239 and
240..341 from the held-out validation) and whose speed-up is the slice-0 measurement on a quiet card (the one router checked
both ways, (0.367, 0.367, 0.367), gave 1.59x under both protocols).

  cd ~/boptim-agent && ./.venv/bin/python <this file>        (prints the state path and the records)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/jli199/boptim-agent")
TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
WORK = Path("/home/jli199/boptim_scratch/bo_lcb_router_pooled")
REF_FULL_TOK_S, FLOOR, PENALTY = 6.25, 18.3, 2.0
# main.py argv exactly as run_experiment.py builds it for configs/ouro_lcb_router_pooled_gp.yaml (max_iter is not part of the key)
ARGV = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_lcb_router", "--batch_size", "2", "--max_iter", "14",
        "--optimizer-seed", "7", "--objective-seed", "7", "--ouro_lcb_mode", "steps", "--ouro_lcb_n", "342", "--ouro_lcb_offset", "0",
        "--ouro_lcb_max_new", "1024", "--ouro_lcb_baseline_acc", "20.5", "--ouro_lcb_tol", "2.2", "--ouro_lcb_penalty", "2.0",
        "--ouro_lcb_work_dir", str(WORK), "--n_initial", "4", "--resume", "True"]


def graded_counts(label, suffix=""):
    """(passed, n) of a wall-clock / validation run from its GRADED line."""
    for rep in (f"{TMP}/wallclock_lcb.txt", f"{TMP}/wallclock_lcb_bo.txt"):
        if not Path(rep).exists():
            continue
        for line in open(rep):
            if line.startswith(f"{label}{suffix} GRADED "):
                kv = dict(p.split("=", 1) for p in line.split()[2:] if "=" in p and not p.startswith("by_difficulty"))
                n = int(kv["n"])
                return round(float(kv["pass@1"]) * n), n
    return None


def pooled_acc(label, slice0_passed):
    a, b = graded_counts(label, "_o120"), graded_counts(label, "_o240")
    if not a or not b:
        return None
    return 100.0 * (slice0_passed + a[0] + b[0]) / (120 + a[1] + b[1])


# slice-0 pass counts (of 120) and raw tok/s on a quiet card
slice0 = {"full": (27, 6.23), "gate0.7": (28, 7.58), "gate0.5": (24, 8.82), "gate0.3": (26, 11.12),
          "bo20": (26, 13.06), "bo15": (26, 12.26), "bo18": (29, 12.05)}
points = {"full": [1.0, 1.0, 1.0], "gate0.7": [0.7, 0.7, 0.7], "gate0.5": [0.5, 0.5, 0.5], "gate0.3": [0.3, 0.3, 0.3],
          "bo20": [0.529, 0.058, 0.348], "bo15": [0.408, 0.269, 0.313], "bo18": [0.49, 0.37, 0.314]}
records, notes = [], []
recal_path = WORK / "recal_speeds.json"   # speed-ups re-measured with the stratified 32-problem calibration (recalibrate_points.py)
recal = {tuple(round(v, 6) for v in d["x"]): d for d in json.load(open(recal_path)).values()} if recal_path.exists() else {}
for name, x in points.items():
    passed, tok_s = slice0[name]
    acc = pooled_acc(name, passed)
    if acc is None:
        notes.append(f"{name}: held-out slices missing, skipped")
        continue
    d = recal.get(tuple(round(v, 6) for v in x))
    if d:
        speedup, src = d["speedup"], "cal32"
    elif recal:
        notes.append(f"{name}: no cal32 speed yet, skipped")
        continue
    else:
        speedup, src = tok_s / 6.23, "slice0 raw"
    value = -(speedup * REF_FULL_TOK_S) + PENALTY * max(0.0, FLOOR - acc)
    records.append({"x": x, "fval": round(value, 4), "note": f"warm:{name} acc={acc:.1f} speedup={speedup:.2f} ({src})"})
for line in open(WORK / "evaluations.jsonl"):
    e = json.loads(line)
    if e.get("status") == "ok":
        records.append({"x": [float(v) for v in e["thresholds"][:3]], "fval": round(float(e["value"]), 4), "note": "measured"})

import main as bo_main  # noqa: E402

sys.argv = ARGV
args = bo_main.parse_args()
state = bo_main._resume_state_path(args)
existing = bo_main._load_resume_records(state) if state.exists() else []
if existing:
    notes.append(f"state file already had {len(existing)} records; replaced")
json.dump({"records": [{"x": r["x"], "fval": r["fval"]} for r in records], "note": "warm start 2026-11-11, see boptim_lcb/make_warm_start.py"}, open(state, "w"), indent=1)
print("state:", state)
for r in records:
    print(f"  {r['x']}  {r['fval']:8.3f}  {r['note']}")
print("\n".join(notes))
