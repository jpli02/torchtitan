"""Plumbing smoke test for OuroLCBRouterObjective: 3 problems, one point (released gate 0.3 as per-step), one GPU."""
import os
import sys
import time

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

cls = OBJECTIVE_REGISTRY["ouro_lcb_router"]
obj = cls(mode="steps", n_items=int(os.environ.get("N", "3")), gpus=os.environ.get("LCB_BO_GPUS", "3"),
          work_dir="/home/jli199/boptim_scratch/bo_lcb_router_smoke", baseline_pass1_pct=22.5)
print(obj.description)
print(obj.bounds)
t0 = time.time()
v = obj.evaluate([0.3, 0.3, 0.3])
print("value", v, "record", obj.pop_eval_records(), f"{time.time()-t0:.0f}s")
