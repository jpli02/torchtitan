"""Plumbing smoke test of OuroBFCLRouterObjective: a few items, gate 0.3 and full depth, one card."""
import os
import sys
import time

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus=os.environ.get("BFCL_BO_GPUS", "5"), n_items=int(os.environ.get("N", "8")), shards=2,
                                             baseline_acc_pct=80.0, work_dir="/home/jli199/boptim_scratch/bo_bfcl_router_smoke")
t0 = time.time()
vals = obj.evaluate_batch([[0.3, 0.3, 0.3]])
vals += obj.evaluate_batch([[1.0, 1.0, 1.0]])
print("values", vals, f"{time.time() - t0:.0f}s")
for r in obj.pop_eval_records():
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("run_dir",)})
