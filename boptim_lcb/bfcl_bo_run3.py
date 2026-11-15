"""Router search on BFCL WITH the calibrated cache fill (third search; the first two could not beat the released gate).

What changed: a token that leaves early no longer leaves plain copies of its exit-loop keys and values in the slots of the loops
it skips, but corrected entries (affine maps fitted on full-depth answers, kv_fill_fit.py; exit-lab model). With plain copies the
released gate collapsed below threshold 0.25; with the fill, gate 0.2 scores 79.7 % instead of 76.9 % on the search set, so the
region below two loops is worth searching now.

Search variables: exit shares, as in bfcl_bo_run2.py, with a wider range for loop-1 exits:
  f1 = share of all answer tokens released after loop 1                      in [0, 0.7]
  f2 = share of the remaining tokens released after loop 2                   in [0.6, 1]
  f3 = share of the still remaining tokens released after loop 3             in [0.5, 1]
Thresholds = cuts between neighbouring values of the gate's cumulative exit probabilities (search-set statistics).
Objective as before: 100 * executed loops / 4 + PENALTY * max(0, floor - accuracy %), floor = full depth - 1.7 on the search set.
Warm start: the fill-on routers already evaluated on the search set (released-gate patterns), then grid points, then the model.

  cd ~/boptim-agent && BFCL_BO_GPUS=0,5 BFCL_SHARDS=4 ./.venv/bin/python -u <this file>          (BO_ITERS=10: 20 evaluations)
  DRY=1 ... prints the warm start and exits.
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

PLAIN = "/home/jli199/boptim_scratch/bo_bfcl_router"            # gate statistics, plain search, full-depth baseline
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router_fill"
LAB = "/home/jli199/boptim_scratch/ouro_base_exitlab"
MAPS = os.environ.get("FILL_MAPS", f"{PLAIN}/fill_maps.pt")
GPUS = os.environ.get("BFCL_BO_GPUS", "0,5")
TOL, N, SEED = 1.7, 575, 13
PENALTY = float(os.environ.get("BFCL_PENALTY", "15.0"))
SHARDS = int(os.environ.get("BFCL_SHARDS", "4"))
BO_ITERS = int(os.environ.get("BO_ITERS", "10"))
N_GRID = int(os.environ.get("N_GRID", "4"))
BOX = [(0.0, 0.7), (0.6, 1.0), (0.5, 1.0)]
# the released-gate patterns, now with the fill; 0.2443 lies between two of the gate's loop-1 values and releases about 39 %
# of the tokens after loop 1, halfway between the 0.25 pattern (18 %) and the 0.2 pattern (68 %)
DESIGN_THR = [[0.3, 0.3, 0.3], [0.25, 0.25, 0.25], [0.2443, 0.2443, 0.2443], [0.2, 0.2, 0.2]]
EXTRA = {"decode_fill_maps": MAPS}

os.makedirs(WORK, exist_ok=True)
for f in ("gate_probs.jsonl", "item_cost.json"):
    if not os.path.exists(f"{WORK}/{f}"):
        shutil.copy(f"{PLAIN}/{f}", f"{WORK}/{f}")
# fill-on routers already evaluated by lab_eval.py on the search set count: copy their records into this search's cache
have = set()
if os.path.exists(f"{WORK}/evaluations.jsonl"):
    have = {tuple(round(v, 6) for v in json.loads(l)["thresholds"]) for l in open(f"{WORK}/evaluations.jsonl") if l.strip()}
if os.path.exists(f"{PLAIN}/lab_search/evaluations.jsonl"):
    with open(f"{WORK}/evaluations.jsonl", "a") as out:
        for line in open(f"{PLAIN}/lab_search/evaluations.jsonl"):
            e = json.loads(line)
            k = tuple(round(v, 6) for v in e["thresholds"])
            if e.get("status") == "ok" and e.get("extra_cfg") == EXTRA and k not in have:
                out.write(line if line.endswith("\n") else line + "\n")
                have.add(k)

full = next(json.loads(l) for l in open(f"{PLAIN}/evaluations.jsonl") if [round(v, 3) for v in json.loads(l)["thresholds"][:3]] == [1.0, 1.0, 1.0])
baseline = full["acc_pct"]
kw = dict(ckpt=LAB, gpus=GPUS, n_items=N, shards=SHARDS, tol_pts=TOL, penalty_per_pt=PENALTY, work_dir=WORK, extra_cfg=EXTRA, baseline_acc_pct=baseline)
steps = OBJECTIVE_REGISTRY["ouro_bfcl_router"](mode="steps", **kw)
shares = OBJECTIVE_REGISTRY["ouro_bfcl_router"](mode="shares", **kw)
print(f"baseline (full depth) {baseline:.2f} % -> floor {baseline - TOL:.2f} %; fill maps {MAPS}", flush=True)
if not os.environ.get("DRY"):
    for thr in DESIGN_THR:
        steps.evaluate(thr)                                          # cached when lab_eval.py already ran it

records, seen = [], set()
for line in open(f"{WORK}/evaluations.jsonl"):
    e = json.loads(line)
    if e.get("status") != "ok" or e.get("extra_cfg") != EXTRA:
        continue
    x = [round(v, 4) for v in shares.thresholds_to_shares(e["thresholds"][:3])]
    inside = all(lo - 1e-9 <= xi <= hi + 1e-9 for xi, (lo, hi) in zip(x, BOX))
    dup = tuple(x) in seen
    print(f"  thr ({e['thresholds'][0]:.4f}, {e['thresholds'][1]:.4f}, {e['thresholds'][2]:.4f}) -> shares ({x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f})  "
          f"acc {e['acc_pct']:.1f}  loops {e['exec_loops']:.3f}  value {shares.value_of(e['acc_pct'], e['exec_loops']):.2f}"
          + ("" if inside else "  (outside the box)") + ("  (same shares as an earlier router)" if dup else ""), flush=True)
    if inside and not dup:
        seen.add(tuple(x))
        records.append({"x": [min(hi, max(lo, xi)) for xi, (lo, hi) in zip(x, BOX)], "fval": round(shares.value_of(e["acc_pct"], e["exec_loops"]), 4)})
if len(records) % 2:
    records.sort(key=lambda r: r["fval"])
    records = records[:-1]                                           # the optimizer counts completed iterations in pairs
print(f"warm start: {len(records)} routers inside the box", flush=True)
if os.environ.get("DRY"):
    sys.exit(0)

cfg_path = "/home/jli199/boptim-agent/configs/ouro_bfcl_router_fill_gp.yaml"
max_iter = len(records) // 2 + BO_ITERS
bounds_flat = [v for i, (lo, hi) in enumerate(BOX) for v in (i, lo, hi)]
open(cfg_path, "w").write(f"""# Router search on BFCL v3 with the calibrated cache fill (exit-lab model), in exit shares (f1, f2, f3); written by
# boptim_lcb/bfcl_bo_run3.py. Thresholds = cuts between the gate's distinct values ({PLAIN}/gate_probs.jsonl).
#   value = 100 * executed loops / 4 + {PENALTY} * max(0, {baseline - TOL:.2f} - accuracy %)
#   {len(records)} warm-start routers, {N_GRID} grid points, then the model
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
      ouro_bfcl_ckpt: {LAB}
      ouro_bfcl_fill_maps: {MAPS}
    batch_size: 2
    max_iter: {max_iter}
    n_initial: {len(records) + N_GRID}
    resume: true
    bounds: [{', '.join(str(v) for v in bounds_flat)}]
    optim_seed: {SEED}
    objective_seed: {SEED}
""")

import main as bo_main  # noqa: E402

sys.argv = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", str(max_iter),
            "--optimizer-seed", str(SEED), "--objective-seed", str(SEED), "--n_initial", str(len(records) + N_GRID), "--resume", "True",
            "--ouro_bfcl_mode", "shares", "--ouro_bfcl_work_dir", WORK, "--bounds"] + [str(v) for v in bounds_flat]
state = bo_main._resume_state_path(bo_main.parse_args())
json.dump({"records": records, "note": "warm start: fill-on routers already evaluated on the search set, in exit shares (boptim_lcb/bfcl_bo_run3.py)"}, open(state, "w"), indent=1)
print("resume state:", state, f"({len(records)} points); config: {cfg_path}; max_iter {max_iter}", flush=True)
env = dict(os.environ, BFCL_BO_GPUS=GPUS)
rc = subprocess.run([sys.executable, "-u", "run_experiment.py", "--config", "configs/ouro_bfcl_router_fill_gp.yaml", "--log-prefix", "bfcl_router_fill_gp"],
                    cwd="/home/jli199/boptim-agent", env=env).returncode
print("BO3 finished with code", rc, flush=True)
