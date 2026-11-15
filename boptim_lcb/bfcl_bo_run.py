"""BO over per-loop exit thresholds on BFCL (search set = 575 of the 1,150 non-live items), with the released gate as a measured
baseline curve and warm start:
  1. full depth on the search set -> baseline accuracy -> floor = baseline - TOL (one standard error);
  2. the released gate at 0.7, 0.5, 0.3, 0.25, 0.2 (the baseline curve; also the warm-start points);
  3. write the optimizer's resume state with these six points and the config with the measured baseline;
  4. gp_hedge over (q1, q2, q3): BO_ITERS iterations of two proposals.
Everything goes through OuroBFCLRouterObjective (real decode-time early exit, batch 1, official AST checker, executed-loop count), so
finished points are reused after a crash and every evaluation is logged in <work_dir>/evaluations.jsonl.

  cd ~/boptim-agent && BFCL_BO_GPUS=5 ./.venv/bin/python -u <this file>            (BO_ITERS=10 by default)
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
GPUS = os.environ.get("BFCL_BO_GPUS", "5")
TOL, PENALTY, N, SEED = 1.7, 5.0, 575, 11
SHARDS = int(os.environ.get("BFCL_SHARDS", "2"))  # batch-1 processes per evaluation (outputs do not depend on it)
BO_ITERS = int(os.environ.get("BO_ITERS", "10"))
DESIGN = [[1.0, 1.0, 1.0], [0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.25, 0.25, 0.25], [0.2, 0.2, 0.2]]

obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus=GPUS, n_items=N, shards=SHARDS, tol_pts=TOL, penalty_per_pt=PENALTY, work_dir=WORK)
obj.evaluate(DESIGN[0])
full = obj._cached(obj._thresholds(DESIGN[0]))
if full is None:
    sys.exit("full-depth baseline failed; not starting the search")
baseline = full["acc_pct"]
obj.acc_floor = baseline - TOL
print(f"baseline (full depth, {N} search items): {baseline:.2f} %  -> floor {obj.acc_floor:.2f} %", flush=True)
obj.evaluate_batch(DESIGN[1:])
records = []
for x in DESIGN:
    rec = obj._cached(obj._thresholds(x))
    if rec is None:
        print("design point failed, left out of the warm start:", x, flush=True)
        continue
    v = obj.value_of(rec["acc_pct"], rec["exec_loops"])
    records.append({"x": x, "fval": round(v, 4)})
    print(f"  gate {x[0]:.2f}: acc {rec['acc_pct']:.1f} %  loops {rec['exec_loops']:.3f}  value {v:.2f}", flush=True)
if len(records) % 2:
    records = records[:-1]  # the optimizer counts completed iterations in pairs

cfg_path = "/home/jli199/boptim-agent/configs/ouro_bfcl_router_gp.yaml"
max_iter = len(records) // 2 + BO_ITERS
open(cfg_path, "w").write(f"""# Router search on BFCL v3 (search set: 575 of the 1,150 non-live items) under real decode-time early exit; written by
# boptim_lcb/bfcl_bo_run.py after it measured the full-depth baseline ({baseline:.2f} %) and the released-gate curve (warm start).
#   value = 100 * executed loops / 4 + {PENALTY} * max(0, {baseline - TOL:.2f} - accuracy %)
replicates: 1

wandb:
  enabled: true
  project: boptim-agent

experiments:
  - optim: gp_hedge
    objective:
      type: ouro_bfcl_router
      ouro_bfcl_mode: steps
      ouro_bfcl_data: bfcl_search.jsonl
      ouro_bfcl_n: {N}
      ouro_bfcl_baseline_acc: {baseline:.4f}
      ouro_bfcl_tol: {TOL}
      ouro_bfcl_penalty: {PENALTY}
      ouro_bfcl_shards: {SHARDS}
      ouro_bfcl_work_dir: {WORK}
    batch_size: 2
    max_iter: {max_iter}
    n_initial: 4
    resume: true
    optim_seed: {SEED}
    objective_seed: {SEED}
""")

import main as bo_main  # noqa: E402

sys.argv = ["main.py", "--optim", "gp_hedge", "--objective", "ouro_bfcl_router", "--batch_size", "2", "--max_iter", str(max_iter),
            "--optimizer-seed", str(SEED), "--objective-seed", str(SEED), "--n_initial", "4", "--resume", "True"]
state = bo_main._resume_state_path(bo_main.parse_args())
json.dump({"records": records, "note": "warm start: full depth and the released gate on the BFCL search set (boptim_lcb/bfcl_bo_run.py)"}, open(state, "w"), indent=1)
print("resume state:", state, f"({len(records)} points); config: {cfg_path}; max_iter {max_iter}", flush=True)
env = dict(os.environ, BFCL_BO_GPUS=GPUS)
rc = subprocess.run([sys.executable, "-u", "run_experiment.py", "--config", "configs/ouro_bfcl_router_gp.yaml", "--log-prefix", "bfcl_router_gp"],
                    cwd="/home/jli199/boptim-agent", env=env).returncode
print("BO finished with code", rc, flush=True)
