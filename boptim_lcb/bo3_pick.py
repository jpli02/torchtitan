"""Prints, as one ROUTERS string for lab_eval.py, the routers of the fill-on search worth validating on unseen items: the cheapest
ones that hold the search-set accuracy floor, distinct in what they do (exit shares at loops 1 / 2 / 3 at least 5 / 5 / 8 points
apart), at most TOP of them (default 3).   python boptim_lcb/bo3_pick.py [TOP]"""
import json
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

PLAIN = "/home/jli199/boptim_scratch/bo_bfcl_router"
FILL = "/home/jli199/boptim_scratch/bo_bfcl_router_fill"
MAPS = f"{PLAIN}/fill_maps.pt"
TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 3
full = next(json.loads(l) for l in open(f"{PLAIN}/evaluations.jsonl") if [round(v, 3) for v in json.loads(l)["thresholds"][:3]] == [1.0, 1.0, 1.0])
floor = full["acc_pct"] - 1.7
conv = OBJECTIVE_REGISTRY["ouro_bfcl_router"](gpus="0", n_items=575, shards=1, work_dir=PLAIN, mode="shares")
evs = [json.loads(l) for l in open(f"{FILL}/evaluations.jsonl") if l.strip()]
evs = [e for e in evs if e.get("status") == "ok" and (e.get("extra_cfg") or {}).get("decode_fill_maps") and e["acc_pct"] >= floor]
picked, shares = [], []
for e in sorted(evs, key=lambda e: (round(e["exec_loops"], 3), -e["acc_pct"])):
    s = conv.thresholds_to_shares(e["thresholds"][:3])
    if any(abs(s[0] - t[0]) < 0.05 and abs(s[1] - t[1]) < 0.05 and abs(s[2] - t[2]) < 0.08 for t in shares):
        continue
    picked.append(e)
    shares.append(s)
    if len(picked) >= TOP:
        break
print(";".join(",".join(repr(round(float(v), 6)) for v in e["thresholds"][:3]) + f"|maps={MAPS}" for e in picked))
