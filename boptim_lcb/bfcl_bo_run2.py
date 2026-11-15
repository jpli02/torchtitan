"""Second router search on BFCL, in EXIT SHARES instead of raw thresholds.

Why: the gate's cumulative exit probabilities sit at about 0.25, 0.5 and 0.75 after loops 1, 2 and 3 (bfcl_gate_dump.py), so most
of a threshold cube is dead space (a q1 above 0.26 never releases a token after loop 1, a q3 below 0.47 always releases after
loop 3), and a 0.002 step inside the 0.25 spike moves 5 % of the tokens. The search variables here are
  f1 = share of all answer tokens released after loop 1                      in [0, 0.25]
  f2 = share of the remaining tokens released after loop 2                   in [0.7, 1]
  f3 = share of the still remaining tokens released after loop 3             in [0.5, 1]   (the rest run all four loops)
and the objective maps them to thresholds by the quantiles of the search-set gate statistics. Decode compute is then a known
function of (f1, f2, f3); what the optimiser has to learn is accuracy.

Warm start: every router already evaluated on the search set (gate curve and first search), expressed in shares, when it lies in
the box. Same objective as the first search: 100 * executed loops / 4 + PENALTY * max(0, floor - accuracy %).

  cd ~/boptim-agent && BFCL_BO_GPUS=0,5 BFCL_SHARDS=4 ./.venv/bin/python -u <this file>          (BO_ITERS=10: 20 evaluations)
  DRY=1 ... prints the warm start and exits.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
GPUS = os.environ.get("BFCL_BO_GPUS", "0,5")
TOL, N, SEED = 1.7, 575, 12
PENALTY = float(os.environ.get("BFCL_PENALTY", "15.0"))
SHARDS = int(os.environ.get("BFCL_SHARDS", "4"))
BO_ITERS = int(os.environ.get("BO_ITERS", "10"))
BOX = [(0.0, 0.25), (0.7, 1.0), (0.5, 1.0)]

obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus=GPUS, n_items=N, shards=SHARDS, tol_pts=TOL, penalty_per_pt=PENALTY, work_dir=WORK, mode="shares")
evs = []
for line in open(f"{WORK}/evaluations.jsonl"):
    try:
        e = json.loads(line)
    except Exception:
        continue
    if e.get("status") == "ok":
        evs.append(e)
full = next(e for e in evs if [round(v, 3) for v in e["thresholds"][:3]] == [1.0, 1.0, 1.0])
baseline = full["acc_pct"]
obj.acc_floor = baseline - TOL
print(f"baseline {baseline:.2f} % -> floor {obj.acc_floor:.2f} %; {len(evs)} routers evaluated so far", flush=True)

records, seen = [], set()
for e in evs:
    x = [round(v, 4) for v in obj.thresholds_to_shares(e["thresholds"][:3])]
    inside = all(lo - 1e-9 <= xi <= hi + 1e-9 for xi, (lo, hi) in zip(x, BOX))
    pred = obj.loops_of_shares(x)
    tag = "" if inside else "  (outside the box)"
    if inside and tuple(x) in seen:
        tag = "  (same shares as an earlier router)"
        inside = False
    print(f"  thr ({e['thresholds'][0]:.3f}, {e['thresholds'][1]:.3f}, {e['thresholds'][2]:.3f}) -> shares ({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})  "
          f"loops measured {e['exec_loops']:.3f} / from shares {pred:.3f}  acc {e['acc_pct']:.1f}{tag}", flush=True)
    if inside:
        seen.add(tuple(x))
        x = [min(hi, max(lo, xi)) for xi, (lo, hi) in zip(x, BOX)]
        records.append({"x": x, "fval": round(obj.value_of(e["acc_pct"], e["exec_loops"]), 4)})
if len(records) % 2:
    records.sort(key=lambda r: r["fval"])
    records = records[:-1]  # the optimizer counts completed iterations in pairs: drop the worst point
print(f"warm start: {len(records)} routers inside the box", flush=True)
if os.environ.get("DRY"):
    sys.exit(0)

cfg_path = "/home/jli199/boptim-agent/configs/ouro_bfcl_router_shares_gp.yaml"
max_iter = len(records) // 2 + BO_ITERS
bounds_flat = [v for i, (lo, hi) in enumerate(BOX) for v in (i, lo, hi)]
open(cfg_path, "w").write(f"""# Second router search on BFCL v3 (search set: 575 non-live items), in exit shares (f1, f2, f3); written by
# boptim_lcb/bfcl_bo_run2.py. Thresholds = quantiles of the search-set gate statistics ({WORK}/gate_probs.jsonl).
#   value = 100 * executed loops / 4 + {PENALTY} * max(0, {baseline - TOL:.2f} - accuracy %)
#   {len(records)} warm-start routers (gate curve and first search), then {2 * BO_ITERS} model-based proposals
replicates: 1

wandb:
  enabled: true
  project: boptim-agent

experiments:
  - optim: gp_hedge
    objective:
      type: ouro_bfcl_router
      ouro_bfcl_mode: shares
      ouro_bfcl_data: bfcl_search.jsonl
      ouro_bfcl_n: {N}
      ouro_bfcl_baseline_acc: {baseline:.4f}
      ouro_bfcl_tol: {TOL}
      ouro_bfcl_penalty: {PENALTY}
      ouro_bfcl_shards: {SHARDS}
      ouro_bfcl_work_dir: {WORK}
    batch_size: 2
    max_iter: {max_iter}
    n_initial: {len(records)}
    resume: true
    bounds: [{', '.join(str(v) for v in bounds_flat)}]
    optim_seed: {SEED}
    objective_seed: {SEED}
""")

import main as bo_main  # noqa: E402

sys.argv = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", str(max_iter),
            "--optimizer-seed", str(SEED), "--objective-seed", str(SEED), "--n_initial", str(len(records)), "--resume", "True",
            "--ouro_bfcl_mode", "shares", "--ouro_bfcl_work_dir", WORK, "--bounds"] + [str(v) for v in bounds_flat]
state = bo_main._resume_state_path(bo_main.parse_args())
json.dump({"records": records, "note": "warm start: every search-set router evaluated before, in exit shares (boptim_lcb/bfcl_bo_run2.py)"}, open(state, "w"), indent=1)
print("resume state:", state, f"({len(records)} points); config: {cfg_path}; max_iter {max_iter}", flush=True)
env = dict(os.environ, BFCL_BO_GPUS=GPUS)
rc = subprocess.run([sys.executable, "-u", "run_experiment.py", "--config", "configs/ouro_bfcl_router_shares_gp.yaml", "--log-prefix", "bfcl_router_shares_gp"],
                    cwd="/home/jli199/boptim-agent", env=env).returncode
print("BO2 finished with code", rc, flush=True)
