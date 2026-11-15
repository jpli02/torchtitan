"""Offline: which points will the optimizer ask for first, given the search box, n_initial and the warm start (no evaluation).
Replays the warm-start records into a fresh GPHedgeOptimizer exactly as main.py does on resume and prints the next asks.
  cd ~/boptim-agent && ./.venv/bin/python <this file> [n_initial] [n_asks]"""
import json
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from optim.gp_hedge import GPHedgeOptimizer  # noqa: E402

BOX = [(0.2, 0.6), (0.05, 0.5), (0.05, 0.5)]
n_initial = int(sys.argv[1]) if len(sys.argv) > 1 else 8
n_asks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
evs = {}
for line in open("/home/jli199/boptim_scratch/bo_bfcl_router/evaluations.jsonl"):
    e = json.loads(line)
    if e.get("status") == "ok":
        evs[tuple(round(v, 3) for v in e["thresholds"][:3])] = e
full = evs[(1.0, 1.0, 1.0)]["acc_pct"]
opt = GPHedgeOptimizer(BOX, n_initial=n_initial, seed=11)
print("grid of initial points:", [[round(v, 3) for v in p] for p in opt._opt._initial_samples] if getattr(opt._opt, "_initial_samples", None) is not None else None)
for g in (0.5, 0.3, 0.25, 0.2):
    e = evs[(g, g, g)]
    val = 100 * e["exec_loops"] / 4 + 15.0 * max(0.0, full - 1.7 - e["acc_pct"])
    opt.tell([[g, g, g]], [val])
    print(f"told gate {g}: {val:.2f}")
for k in range(n_asks):
    pts, _ = opt.ask(n_points=2)
    print(f"ask {k + 1}:", [[round(v, 3) for v in p] for p in pts])
    opt.tell(pts, [60.0, 60.0])  # placeholder values, only to advance the initial phase
