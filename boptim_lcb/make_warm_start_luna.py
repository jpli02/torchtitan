"""Warm start for the LLM-optimizer run on the pooled LiveCodeBench objective: the SAME seven records the Gaussian-process
run started from (full depth, the released gate at 0.7 / 0.5 / 0.3 and three routers of the earlier slice search), copied
from that run's resume state, written to the resume state of the chatgpt run so both optimizers start from equal information.

  cd ~/boptim-agent && ./.venv/bin/python <this file>          (prints the state path and the records)
"""
import json
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
GP_STATE = "/home/jli199/boptim-agent/runs/state_gp_hedge_ouro_lcb_router_6b77c3e5c1c32373.json"
WORK = "/home/jli199/boptim_scratch/bo_lcb_router_pooled_luna"
MODEL = "openai/gpt-5.6-luna"
# main.py argv as run_experiment.py builds it for configs/ouro_lcb_router_pooled_luna.yaml (max_iter is not part of the state key)
ARGV = ["main.py", "--optim", "chatgpt", "--model", MODEL, "--objective", "ouro_lcb_router", "--batch_size", "1", "--max_iter", "17",
        "--optimizer-seed", "7", "--objective-seed", "7", "--ouro_lcb_mode", "steps", "--ouro_lcb_n", "342", "--ouro_lcb_offset", "0",
        "--ouro_lcb_max_new", "1024", "--ouro_lcb_baseline_acc", "20.5", "--ouro_lcb_tol", "2.2", "--ouro_lcb_penalty", "2.0",
        "--ouro_lcb_work_dir", WORK, "--resume", "True"]
warm = json.load(open(GP_STATE))["records"][:7]
expect = [[1.0, 1.0, 1.0], [0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]
assert [[round(v, 3) for v in r["x"]] for r in warm[:4]] == expect, "the first records of the GP state are not the released-gate points"

import main as bo_main  # noqa: E402

sys.argv = ARGV
args = bo_main.parse_args()
state = bo_main._resume_state_path(args)
if state.exists() and bo_main._load_resume_records(state):
    print("state already has records; left as it is:", state)
else:
    json.dump({"records": [{"x": r["x"], "fval": r["fval"]} for r in warm],
               "note": "warm start 2026-11-20: the 7 records the gp_hedge pooled run started from (boptim_lcb/make_warm_start_luna.py)"},
              open(state, "w"), indent=1)
    print("state written:", state)
for r in warm:
    print("  ", [round(v, 3) for v in r["x"]], round(r["fval"], 3))
