"""BFCL live set: accuracy of every evaluated router on the 900 live items that were never used to fit a correction
(index % 3 != 0), and on the 450 items of the fit pool, from the per-item grades of the existing runs.

  python boptim_lcb/bfcl_live_heldout_compare.py
Rows: plain gate points, the same thresholds with the correction fitted on non-live answers (fill_maps.pt) and with the
correction fitted on live answers (fill_maps_live.pt). Writes <work>/live_heldout_compare.json.
"""
import json
import os

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_live_all.jsonl"


def main():
    ids = [json.loads(line).get("id") for line in open(DATA) if line.strip()]
    idx = {i: k for k, i in enumerate(ids)}
    runs = {}
    for d in ("live", "lab_live"):
        p = f"{WORK}/{d}/evaluations.jsonl"
        if not os.path.exists(p):
            continue
        for line in open(p):
            e = json.loads(line)
            g = os.path.join(e.get("run_dir") or "", "graded.json")
            if e.get("status") != "ok" or not os.path.exists(g):
                continue
            maps = (e.get("extra_cfg") or {}).get("decode_fill_maps") or ""
            kind = "plain" if not maps else ("live maps" if maps.endswith("fill_maps_live.pt") else "non-live maps")
            key = (tuple(round(v, 4) for v in e["thresholds"][:3]), kind)
            if key not in runs:
                runs[key] = (e, json.load(open(g))["details"])
    out = []
    for (thr, kind), (e, det) in sorted(runs.items(), key=lambda kv: (-kv[1][0]["exec_loops"], kv[0][1])):
        held = [d["ok"] for d in det if idx.get(d["id"], 0) % 3 != 0]
        pool = [d["ok"] for d in det if idx.get(d["id"], 0) % 3 == 0]
        row = {"thresholds": list(thr), "kind": kind, "loops": e["exec_loops"], "all_pct": e["acc_pct"],
               "heldout_n": len(held), "heldout_pct": 100.0 * sum(held) / max(len(held), 1), "heldout_ok": sum(held),
               "pool_n": len(pool), "pool_pct": 100.0 * sum(pool) / max(len(pool), 1)}
        out.append(row)
    json.dump({"rows": out}, open(f"{WORK}/live_heldout_compare.json", "w"), indent=1)
    print(f"{'router':28s} {'correction':14s} {'loops':>6s} {'all 1,350':>10s} {'900 untouched':>14s} {'450 fit pool':>13s}")
    for r in out:
        print(f"{str(tuple(r['thresholds'])):28s} {r['kind']:14s} {r['loops']:6.3f} {r['all_pct']:9.2f}% {r['heldout_pct']:9.2f}% ({r['heldout_ok']:3d}) {r['pool_pct']:11.2f}%")


if __name__ == "__main__":
    main()
