"""Offline check of the cost-balanced shard ranges (no GPU): writes <work>/item_cost.json from the full-depth evaluation if it is
missing, then prints the ranges and each shard's share of the work for 2, 3 and 4 shards, against the equal-count split."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

WORK = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/bo_bfcl_router"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 575
obj = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus="0", n_items=N, shards=4, work_dir=WORK)
if not (Path(WORK) / "item_cost.json").exists():
    for rec in sorted(glob.glob(f"{WORK}/q1.000-1.000-1.000_*/record.json")):
        if json.load(open(rec)).get("status") == "ok":
            obj._save_item_cost(Path(rec).parent)
            break
cost = json.loads((Path(WORK) / "item_cost.json").read_text())
print("items", len(cost), "total cost", round(sum(cost)))
for k in (2, 3, 4):
    b = obj._bounds(k)
    eq = [((N * j) // k, (N * (j + 1)) // k) for j in range(k)]
    assert b[0][0] == 0 and b[-1][1] == N and all(b[j][1] == b[j + 1][0] for j in range(k - 1)), b
    share = lambda bs: [round(100 * sum(cost[lo:hi]) / sum(cost)) for lo, hi in bs]  # noqa: E731
    print(k, "shards: balanced", b, share(b), "| equal", share(eq))
