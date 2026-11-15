"""Real-generation accuracy of exit-lab routers on a BFCL set, through the search's own pipeline (real decode-time early exit at
batch 1, official AST checker in the sandbox, executed-loop count). A router is written as in bfcl_tf_exit.py:
  "q1,q2,q3"  [ "|conf=t1,t2,t3" ]  [ "|maps=path" ]  [ "|skip=n1,n2,n3" ]
Results are cached per set in <work>/lab_<set>/evaluations.jsonl (key: thresholds + settings).

  cd ~/boptim-agent && SET=search ROUTERS="...;..." ./.venv/bin/python -u <this file>        (SET: search | heldout | live)
"""
import json
import os
import shutil
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from objective import OBJECTIVE_REGISTRY  # noqa: E402
from bfcl_tf_exit import parse_router  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
LAB = "/home/jli199/boptim_scratch/ouro_base_exitlab"
SETS = {"search": ("bfcl_search.jsonl", 575, f"{WORK}/item_cost.json"), "heldout": ("bfcl_heldout.jsonl", 575, f"{WORK}/heldout/item_cost.json"),
        "live": ("bfcl_live_all.jsonl", 1350, f"{WORK}/live/item_cost.json")}
name = os.environ.get("SET", "search")
data, n, cost = SETS[name]
work = f"{WORK}/lab_{name}"
os.makedirs(work, exist_ok=True)
if os.path.exists(cost) and not os.path.exists(f"{work}/item_cost.json"):
    shutil.copy(cost, f"{work}/item_cost.json")       # balanced shards from the first evaluation on
for spec in os.environ.get("ROUTERS", "").split(";"):
    if not spec.strip():
        continue
    r = parse_router(spec.strip())
    extra = {}
    if r["conf"] is not None:
        extra["decode_exit_conf"] = r["conf"]
    if r["maps"]:
        extra["decode_fill_maps"] = r["maps"]
    if r["skip"] is not None:
        extra["decode_fill_skip"] = r["skip"]
    if r.get("stream"):
        extra["decode_fill_stream"] = r["stream"]
    obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](ckpt=os.environ.get("LAB_CKPT", LAB), gpus=os.environ.get("BFCL_BO_GPUS", "0"), n_items=n, shards=int(os.environ.get("BFCL_SHARDS", "4")),
                                                 data_name=data, work_dir=work, extra_cfg=extra or None)
    obj.evaluate(r["gate"])
    rec = obj._cached(obj._thresholds(r["gate"]))
    if rec:
        print("LAB " + json.dumps({"set": name, "router": spec, "acc_pct": round(rec["acc_pct"], 2), "unweighted_pct": round(rec["overall_unweighted_pct"], 2),
                                   "exec_loops": round(rec["exec_loops"], 4), "wall_min": round(rec["wall_s"] / 60, 1)}), flush=True)
    else:
        print("LAB " + json.dumps({"set": name, "router": spec, "status": "failed"}), flush=True)
print("lab eval finished", flush=True)
