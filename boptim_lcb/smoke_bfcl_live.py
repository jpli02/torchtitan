"""Plumbing smoke test of the live-category path (generation, sandboxed official checker, executed-loop count) on a few items."""
import os
import sys
import time

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus=os.environ.get("BFCL_BO_GPUS", "0"), n_items=int(os.environ.get("N", "12")), shards=1,
                                             data_name=os.environ.get("DATA", "bfcl_live_all.jsonl"),
                                             work_dir="/home/jli199/boptim_scratch/bo_bfcl_router_smoke_live")
obj.MAX_ATTEMPTS = 1
t0 = time.time()
print("values", obj.evaluate_batch([[0.3, 0.3, 0.3]]), f"{time.time() - t0:.0f}s")
for r in obj.pop_eval_records():
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("run_dir",)})
