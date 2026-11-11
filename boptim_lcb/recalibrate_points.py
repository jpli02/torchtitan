"""Re-measure the speed term of every known pooled-BO point with the stratified 31-problem calibration (candidate and full depth
back to back on one card), then rewrite <work_dir>/evaluations.jsonl (measured points) and the warm-start file (designed points)
so every value the optimizer sees uses one protocol.

  cd ~/boptim-agent && ./.venv/bin/python <this file> --gpu 0          (about 1 h per point)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/jli199/boptim-agent")
from objective import OBJECTIVE_REGISTRY  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--work_dir", default="/home/jli199/boptim_scratch/bo_lcb_router_pooled")
args = ap.parse_args()
WORK = Path(args.work_dir)
REF, FLOOR, PENALTY = 6.25, 18.3, 2.0
obj = OBJECTIVE_REGISTRY["ouro_lcb_router"](n_items=342, baseline_pass1_pct=20.5, tol_pts=2.2, gpus=args.gpu, work_dir=str(WORK))
cal_dir = WORK / "recal"
cal_dir.mkdir(exist_ok=True)
out_path = WORK / "recal_speeds.json"
done = json.load(open(out_path)) if out_path.exists() else {}

warm = {"full": [1.0, 1.0, 1.0], "gate0.7": [0.7, 0.7, 0.7], "gate0.5": [0.5, 0.5, 0.5], "gate0.3": [0.3, 0.3, 0.3],
        "bo20": [0.529, 0.058, 0.348], "bo15": [0.408, 0.269, 0.313], "bo18": [0.49, 0.37, 0.314]}
measured = [json.loads(l) for l in open(WORK / "evaluations.jsonl") if l.strip()]
points = [(f"m{k}", e["thresholds"][:3]) for k, e in enumerate(measured) if e.get("status") == "ok"] + list(warm.items())
for name, x in points:
    key = json.dumps([round(v, 6) for v in x])
    if key in done:
        continue
    thr = list(x) + [1.0]
    t0 = time.time()
    cal = obj._calibrate(thr, args.gpu, cal_dir, f"{name}_thr")
    full = obj._calibrate([1.0, 1.0, 1.0, 1.0], args.gpu, cal_dir, f"{name}_full") if name != "full" else cal
    if not cal or not full:
        print(name, "calibration failed", flush=True)
        continue
    speedup = cal["tok_s"] / full["tok_s"]
    done[key] = {"name": name, "x": x, "tok_s_cal": cal["tok_s"], "tok_s_full_cal": full["tok_s"], "speedup": speedup,
                 "clean": bool(cal["clean"] and full["clean"]), "min": round((time.time() - t0) / 60)}
    json.dump(done, open(out_path, "w"), indent=1)
    print(f"{name} {x}: {cal['tok_s']:.2f} / {full['tok_s']:.2f} = {speedup:.3f}x ({'clean' if done[key]['clean'] else 'SHARED'}, {done[key]['min']} min)", flush=True)

# rewrite the measured records with the new speed term
new_lines = []
for e in measured:
    key = json.dumps([round(v, 6) for v in e["thresholds"][:3]])
    if e.get("status") == "ok" and key in done:
        d = done[key]
        e.update(tok_s_cal=d["tok_s_cal"], tok_s_full_cal=d["tok_s_full_cal"], speedup=d["speedup"], tok_s=d["speedup"] * REF, cal_clean=d["clean"], cal_set="cal31")
        e["value"] = -e["tok_s"] + PENALTY * max(0.0, FLOOR - e["pass1_pct"])
    new_lines.append(json.dumps(e) + "\n")
(WORK / "evaluations_pre_cal31.jsonl").write_text("".join(json.dumps(e) + "\n" for e in measured))
(WORK / "evaluations.jsonl").write_text("".join(new_lines))
print("evaluations.jsonl rewritten with the cal31 speed term")
