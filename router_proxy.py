"""Fast router proxy objective over dumped per-token features (fidelity 0 for boptim).

Features per generated token (from oracle_depth.py --dump): agree[T] (bool: step s argmax ==
final argmax), lam[T] (gate exit prob per step), hdiff[T-1] (relative hidden-state change
from step s-1 to s), gen_id. A router rule maps features -> exit step per token; the proxy
reports mean loops and flip rate (fraction of tokens exited before the first agreeing step).

Rule families (params dict):
  gate      : {"q": float}                     exit at first step with CDF(lam) >= q
  gate_steps: {"q1","q2","q3"}                 per-step thresholds on the CDF
  hdiff     : {"eps": float}                   exit at step s if hdiff[s] < eps  (s>=1, i.e. after 2 loops)
  gate_hdiff: {"q", "eps"}                     exit when either condition fires
  lam_steps : {"l1","l2","l3"}                 exit at step s if the per-step lambda_s >= l_s (not cumulative)
  min_loops : any of the above + {"min": int}  never exit before `min` loops

Usage:
  python router_proxy.py FEAT.pt                     # frontier table for the gate, per-step and hdiff families
  python router_proxy.py FEAT.pt --rule gate --params '{"q":0.2}'
"""
import argparse
import itertools
import json

import torch


def load(path):
    d = torch.load(path)
    return {"agree": d["agree"].bool(), "lam": d["lam"].float(), "hdiff": d["hdiff"].float(), "gen_id": d["gen_id"]}


def cdf_from_lam(lam):
    T = lam.shape[1]
    cdf = torch.zeros_like(lam)
    remaining = torch.ones(lam.shape[0])
    for s in range(T - 1):
        cdf[:, s] = (cdf[:, s - 1] if s > 0 else 0) + lam[:, s] * remaining
        remaining = remaining * (1 - lam[:, s])
    cdf[:, T - 1] = 1.0
    return cdf


def exit_steps(feat, rule, p):
    lam, hd = feat["lam"], feat["hdiff"]
    N, T = lam.shape
    fire = torch.zeros(N, T, dtype=torch.bool)
    fire[:, T - 1] = True
    if rule in ("gate", "gate_hdiff"):
        cdf = cdf_from_lam(lam)
        fire |= cdf >= float(p["q"])
    if rule == "gate_steps":
        cdf = cdf_from_lam(lam)
        for s, k in enumerate(("q1", "q2", "q3")):
            if s < T - 1:
                fire[:, s] |= cdf[:, s] >= float(p[k])
    if rule in ("hdiff", "gate_hdiff"):
        # hdiff[:, s-1] is the change going INTO step s; small change => step s added little => exit at s
        for s in range(1, T - 1):
            fire[:, s] |= hd[:, s - 1] < float(p["eps"])
    if rule == "lam_steps":
        for s, k in enumerate(("l1", "l2", "l3")):
            if s < T - 1:
                fire[:, s] |= lam[:, s] >= float(p[k])
    mn = int(p.get("min", 1))
    if mn > 1:
        fire[:, : mn - 1] = False
    return fire.float().argmax(1)  # first firing step (0-based)


def evaluate(feat, rule, p):
    ex = exit_steps(feat, rule, p)
    agree = feat["agree"]
    oracle = agree.float().argmax(1)
    N = ex.numel()
    chosen_ok = agree[torch.arange(N), ex]
    return {"rule": rule, "params": p, "mean_loops": float((ex + 1).float().mean()),
            "flip_rate": float((~chosen_ok).float().mean()), "under_think": float((ex < oracle).float().mean()),
            "over_think": float((ex > oracle).float().mean()), "tokens": N}


def frontier(feat):
    rows = []
    for q in (0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
        rows.append(evaluate(feat, "gate", {"q": q}))
    for eps in (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2):
        rows.append(evaluate(feat, "hdiff", {"eps": eps}))
    for q1, q2, q3 in itertools.product((0.1, 0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 0.5), (0.1, 0.3, 0.5)):
        rows.append(evaluate(feat, "gate_steps", {"q1": q1, "q2": q2, "q3": q3}))
    for q, eps in itertools.product((0.3, 0.5, 0.7), (0.03, 0.05, 0.08, 0.1)):
        rows.append(evaluate(feat, "gate_hdiff", {"q": q, "eps": eps}))
    return rows


def pareto(rows):
    rows = sorted(rows, key=lambda r: (r["mean_loops"], r["flip_rate"]))
    best, out = 1e9, []
    for r in rows:
        if r["flip_rate"] < best:
            best = r["flip_rate"]
            out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("feat")
    ap.add_argument("--rule")
    ap.add_argument("--params")
    ap.add_argument("--out")
    a = ap.parse_args()
    feat = load(a.feat)
    oracle = feat["agree"].float().argmax(1)
    print(f"tokens={feat['agree'].shape[0]}  oracle mean loops={float((oracle + 1).float().mean()):.3f}")
    if a.rule:
        r = evaluate(feat, a.rule, json.loads(a.params or "{}"))
        print(json.dumps(r))
    else:
        rows = frontier(feat)
        print(f"\nPareto frontier (loops vs flip rate) over {len(rows)} rule settings:")
        print(f"{'rule':<11} {'params':<36} {'loops':>6} {'flip':>7} {'under':>7} {'over':>6}")
        for r in pareto(rows):
            print(f"{r['rule']:<11} {json.dumps(r['params']):<36} {r['mean_loops']:6.3f} {r['flip_rate']:7.4f} {r['under_think']:7.4f} {r['over_think']:6.3f}")
        print("\nreleased gate reference points:")
        for q in (0.5, 0.3, 0.2, 0.1):
            r = evaluate(feat, "gate", {"q": q})
            print(f"  q={q}: loops {r['mean_loops']:.3f} flip {r['flip_rate']:.4f}")
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)
