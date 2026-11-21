"""End-to-end smoke test of the gate-NAS objective on a few short LiveCodeBench problems (no optimizer): one point that is decoded
(linear router, gamma 0.005, q 0.5) and one that should be scored as fixed depth 1 without decoding (gamma 0.9, q 0.15).

  python boptim_lcb/gate_nas_smoke.py --gpus 9
"""
import argparse
import json
import shutil
import sys

sys.path.insert(0, "/home/jli199/boptim-agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="9")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from objective.ouro_lcb_gate_nas import OuroLCBGateNASObjective

    work = "/home/jli199/boptim_scratch/randrouter/gatelab/smoke"
    shutil.rmtree(work, ignore_errors=True)
    o = OuroLCBGateNASObjective(n_items=a.n, data=f"/home/jli199/boptim_scratch/randrouter/gatelab/lcb_short{a.n}.jsonl", baseline_pass1_pct=20.5,
                                tol_pts=2.2, gpus=a.gpus, work_dir=work)
    o.MIN_FREE_MIB = 8000      # the test's 6 short problems fit next to another user's idle server on cards 8 / 9
    print(o.description, flush=True)
    for x in ([0, 1, 1, 0, 0.005, 0.5], [0, 1, 1, 0, 0.9, 0.15]):
        v = o.evaluate(x)
        print("point", x, "value", round(v, 3), flush=True)
    for line in open(f"{work}/evaluations.jsonl"):
        r = json.loads(line)
        fr = r.get("fit_report") or {}
        print({k: r.get(k) for k in ("status", "shortcut", "router_arch", "gamma", "thresholds", "pass1_pct", "loops_eff", "exec_loops", "value")},
              "loop1 share", fr.get("loop1_exit_share_train"), "proxy", {k: round(v, 3) for k, v in (fr.get("proxy") or {}).items() if isinstance(v, float)})
    print("GATE NAS SMOKE done")


if __name__ == "__main__":
    main()
