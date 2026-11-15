"""Calibrate the per-item cost used to balance evaluation shards (ouro_bfcl_router._bounds) from finished evaluations:
  shard wall time ~ a * rows + b * generated tokens + c * prompt characters          (least squares over all finished shards)
and write <work>/item_cost.json = a + b * ntok_i + c * prompt_chars_i per item (ntok from the full-depth run).
  python fit_bfcl_item_cost.py [work_dir] [data_file]            (run with the boptim venv python; reads only)"""
import glob
import json
import sys

import numpy as np

WORK = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/bo_bfcl_router"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
rows = [json.loads(l) for l in open(DATA) if l.strip()]
pchars = np.array([len(r.get("prompt", "")) + len(r.get("function_prompt", "") if isinstance(r.get("function_prompt"), str) else json.dumps(r.get("function_prompt", ""))) for r in rows], float)
n = len(rows)

X, y, full_ntok = [], [], None
for rec in sorted(glob.glob(f"{WORK}/q*/record.json")):
    r = json.load(open(rec))
    if r.get("status") != "ok":
        continue
    d = rec.rsplit("/", 1)[0]
    lo = 0
    ntoks = []
    for j in range(r["shards"]):
        gen = [json.loads(l) for l in open(f"{d}/gen_{j}.jsonl") if l.strip() and '"summary"' not in l[:12]]
        res = None
        for line in open(f"{d}/gen_{j}.log", errors="replace"):
            if line.startswith("RESULT "):
                res = json.loads(line[7:])
        if res is None:
            break
        nt = np.array([float(g["ntok"]) for g in gen])
        hi = lo + len(gen)
        # decode time scales with executed loops per token; use the shard's own mean
        loops = res.get("exec_loops") or 4.0
        X.append([len(gen), nt.sum() * loops / 4.0, pchars[lo:hi].sum()])
        y.append(res["wall_s"])
        ntoks.append(nt)
        lo = hi
    if lo == n and tuple(round(v, 3) for v in r["thresholds"][:3]) == (1.0, 1.0, 1.0):
        full_ntok = np.concatenate(ntoks)
X, y = np.array(X), np.array(y)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print(f"shards fitted: {len(y)}; a (s/row) {coef[0]:.3f}, b (s per full-depth token) {coef[1]:.4f}, c (s per prompt char) {coef[2]:.6f}")
print(f"relative error of the fit: mean {np.mean(np.abs(pred - y) / y) * 100:.1f} %, max {np.max(np.abs(pred - y) / y) * 100:.1f} %")
if full_ntok is None or len(full_ntok) != n:
    sys.exit("no complete full-depth run found; cost file not written")
a, b, c = (max(0.0, v) for v in coef)
cost = a + b * full_ntok + c * pchars
json.dump([round(float(v), 4) for v in cost], open(f"{WORK}/item_cost.json", "w"))
tot = cost.sum()
for k in (2, 3, 4):
    eq = [((n * j) // k, (n * (j + 1)) // k) for j in range(k)]
    print(k, "shards, equal-count shares of work:", [round(100 * cost[lo:hi].sum() / tot) for lo, hi in eq])
print("wrote", f"{WORK}/item_cost.json")
