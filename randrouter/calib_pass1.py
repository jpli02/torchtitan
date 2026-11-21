"""Calibrate the teacher-forced proxy against real execution: what on the proxy predicts LiveCodeBench pass@1?

Every router of the released model that was run for real on the 342 test problems (the single knob at five settings, the
evaluations of the Gaussian-process and LLM searches: decode-time early exit with the decoder's copy fill; and fixed depth 1
and 2) is replayed on the proxy (34 held-out answers of the base model, same cache behaviour). Per router the proxy gives
    loops       executed loops per answer token
    own  (a)    % of answer tokens whose exit readout is wrong even with an exact cache: the token left too early
    cache (b)   % of answer tokens that are right with an exact cache and wrong with the decoder's cache: damage done by the
                copied keys/values of tokens that left earlier
    agreement = 100 - a - b
Token agreement alone does not order the routers (gate 0.2: 94.4 % and 15.0 % pass@1; every token leaving after loop 1: 95.3 %
and 5.6 % at fixed depth 1), because the two kinds of error do not cost the same. The fitted model

    pass@1 = c - alpha * a - beta * b

is what the proxy objectives use to express their penalty in pass@1 points, like the real objective
(value = -6.25 * speed-up + 2.0 * max(0, 18.3 - pass@1)).

  CUDA_VISIBLE_DEVICES=9 python randrouter/calib_pass1.py --mem_gb 8.5
Writes /home/jli199/boptim_scratch/randrouter/calib_pass1.json (points, fitted coefficients, fidelity).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import tf_eval  # noqa: E402

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
SEARCHES = {"gp": "/home/jli199/boptim_scratch/bo_lcb_router_pooled/evaluations.jsonl",
            "luna6": "/home/jli199/boptim_scratch/bo_lcb_router_pooled_luna6/evaluations.jsonl"}
SINGLE = {"full": 1.0, "gate0.7": 0.7, "gate0.5": 0.5, "gate0.3": 0.3, "gate0.2": 0.2}
FIXED = {"k1": "iid:1,0,0,0", "k2": "iid:0,1,0,0"}     # fixed depth 1 / 2 = every answer token leaves after loop 1 / 2


def graded(path):
    d = json.load(open(path))
    rs = d["results"]
    return [bool(r.get("passed", False)) for r in (rs.values() if isinstance(rs, dict) else rs)]


def real_points():
    pts = []
    for name, q in SINGLE.items():                      # the single knob, three slices of the 342 problems
        passed = []
        for f in [f"{TMP}/wcl_{name}.graded.json", f"{TMP}/wcl_{name}_o120.graded.json", f"{TMP}/wcl_{name}_o240.graded.json"]:
            if os.path.exists(f):
                passed += graded(f)
        if passed:
            pts.append({"spec": f"{q:.6f},{q:.6f},{q:.6f}", "thr": [q, q, q], "pass1": 100.0 * sum(passed) / len(passed), "n": len(passed),
                        "src": f"single knob {name}", "kind": "gate"})
    for name, spec in FIXED.items():
        f = f"{TMP}/bench_lcb342_{name}.graded.json"
        if os.path.exists(f):
            passed = graded(f)
            pts.append({"spec": spec, "thr": None, "pass1": 100.0 * sum(passed) / len(passed), "n": len(passed), "src": f"fixed depth {name[1]}", "kind": "fixed"})
    for tag, path in SEARCHES.items():
        if not os.path.exists(path):
            continue
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or r.get("pass1_pct") is None:
                continue
            thr = [float(v) for v in r["thresholds"][:3]]
            pts.append({"spec": ",".join(f"{v:.6f}" for v in thr), "thr": thr, "pass1": float(r["pass1_pct"]), "n": 342, "src": f"search {tag}",
                        "kind": "gate", "exec_loops": r.get("exec_loops"), "speedup": r.get("speedup"), "trunc": r.get("trunc_rate")})
    return pts


def fit(a, b, y, w):
    """Weighted least squares y = c - alpha a - beta b with alpha, beta >= 0."""
    def solve(cols):
        X = np.stack([np.ones_like(y)] + [-c for c in cols], axis=1) * np.sqrt(w)[:, None]
        return np.linalg.lstsq(X, y * np.sqrt(w), rcond=None)[0]
    c, al, be = solve([a, b])
    if al < 0 and be < 0:
        return float(np.average(y, weights=w)), 0.0, 0.0
    if al < 0:
        c, be = solve([b])
        return float(c), 0.0, float(max(be, 0.0))
    if be < 0:
        c, al = solve([a])
        return float(c), float(max(al, 0.0)), 0.0
    return float(c), float(al), float(be)


def spearman(u, v):
    def rank(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        x = np.asarray(x)
        for val in np.unique(x):
            m = x == val
            r[m] = r[m].mean()
        return r
    ru, rv = rank(u), rank(v)
    return float(np.corrcoef(ru, rv)[0, 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_gb", type=float, default=8.5)
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/calib_pass1.json")
    a_ = ap.parse_args()
    pts = real_points()
    print(f"{len(pts)} real measurements", flush=True)
    if a_.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a_.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(core.SERVE_DIR)
    seqs = core.load_teacher("/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split="val")
    specs = sorted({p["spec"] for p in pts})
    res = tf_eval.evaluate(model, seqs, [tf_eval.parse_router(s) for s in specs], tok.pad_token_id, 2500, log=lambda s: None, exact=True)
    for p in pts:
        r = res[p["spec"]]
        p["tf_loops"], p["tf_agree"], p["tf_agree_exact"] = r["loops"], 100.0 * r["agree_real"], 100.0 * r["agree_exact"]
        p["tf_ce"], p["tf_ce_exact"], p["tf_share"] = r["ce_real"], r["ce_exact"], r["share"]
        p["a"], p["b"] = 100.0 - p["tf_agree_exact"], p["tf_agree_exact"] - p["tf_agree"]
    pts.sort(key=lambda p: p["tf_agree"])
    y = np.array([p["pass1"] for p in pts])
    w = np.array([p["n"] / 342.0 for p in pts])
    a = np.array([p["a"] for p in pts])
    b = np.array([p["b"] for p in pts])
    c, al, be = fit(a, b, y, w)
    pred = c - al * a - be * b
    loo = []
    for i in range(len(pts)):                            # leave one measurement out
        m = np.arange(len(pts)) != i
        ci, ai, bi = fit(a[m], b[m], y[m], w[m])
        loo.append(ci - ai * a[i] - bi * b[i])
    loo = np.array(loo)
    for p, pr, pl in zip(pts, pred, loo):
        p["pass1_fit"], p["pass1_loo"] = float(pr), float(pl)
        el = "" if p.get("exec_loops") is None else f"  executed loops {p['exec_loops']:.2f}"
        name = str([round(v, 3) for v in p["thr"]]) if p["thr"] else p["src"]
        print(f"  {name:<24} loops {p['tf_loops']:.2f}  agreement {p['tf_agree']:.2f} (own {p['a']:.2f}, cache {p['b']:.2f})  ce {p['tf_ce']:.3f}  "
              f"pass@1 {p['pass1']:.1f} (n={p['n']}) fit {pr:.1f} left-out {pl:.1f}{el}  [{p['src']}]")
    rm = lambda d: float(np.sqrt(np.average(d ** 2, weights=w)))  # noqa: E731
    print(f"model pass@1 = {c:.2f} - {al:.3f} * own - {be:.3f} * cache   (weighted RMSE {rm(pred - y):.2f}, left-out {rm(loo - y):.2f} pass@1 points; "
          f"binomial standard error of one measurement on 342 problems is about 2.1)")
    agree = np.array([p["tf_agree"] for p in pts])
    ce = np.array([p["tf_ce"] for p in pts])
    print(f"rank correlation with real pass@1 over {len(pts)} measurements: fitted model {spearman(pred, y):.3f} (left-out {spearman(loo, y):.3f}), "
          f"agreement alone {spearman(agree, y):.3f}, cross-entropy alone {spearman(-ce, y):.3f}")
    g = np.array([p["kind"] == "gate" for p in pts])
    print(f"  among the {int(g.sum())} gate routers only: fitted model {spearman(pred[g], y[g]):.3f}, agreement alone {spearman(agree[g], y[g]):.3f}")
    lp = [p for p in pts if p.get("exec_loops")]
    if len(lp) > 2:
        d = np.array([p["tf_loops"] - p["exec_loops"] for p in lp])
        print(f"proxy loops against executed loops ({len(lp)} runs with the exact counter): mean difference {d.mean():+.3f}, largest {np.abs(d).max():.3f}")
    json.dump({"points": pts, "model": {"c": c, "alpha": al, "beta": be}, "rmse": rm(pred - y), "rmse_loo": rm(loo - y),
               "spearman_model": spearman(pred, y), "spearman_loo": spearman(loo, y), "spearman_agreement": spearman(agree, y)},
              open(a_.out, "w"), indent=1)
    print("saved", a_.out)


if __name__ == "__main__":
    main()
