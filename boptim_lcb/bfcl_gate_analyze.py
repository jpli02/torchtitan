"""Reads the gate-probability dump (bfcl_gate_dump.py) and answers three questions about the exit gate on BFCL answers:
  1. where the cumulative exit probabilities P1, P2, P3 of the answer tokens lie (quantiles, mass in bands, the spike near 1/4);
  2. how well the dump predicts the executed loops measured under real early exit, for every router already evaluated
     (teacher-forced on full-depth answers, so an exit cannot change later tokens here, unlike in real decoding);
  3. which tokens leave after loop 1 at the usual thresholds.
  python bfcl_gate_analyze.py [gate_probs.jsonl]            -> prints a report and writes <work>/gate_analysis.json"""
import collections
import json
import sys

import numpy as np

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{WORK}/gate_probs.jsonl"
P, toks, cats = [], [], []
for line in open(SRC):
    r = json.loads(line)
    P.extend(r["P"])
    toks.extend(r["next_tok"])
    cats.extend([r["category"]] * r["n"])
P = np.array(P, float)
n = len(P)
print(f"answer tokens: {n} from {sum(1 for _ in open(SRC))} items")
out = {"tokens": n}

# 1. distribution
qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
for s in range(3):
    q = np.quantile(P[:, s], qs)
    print(f"P{s + 1} quantiles " + " ".join(f"{a:.0%}:{b:.3f}" for a, b in zip(qs, q)) + f"  max {P[:, s].max():.3f}")
    out[f"P{s + 1}_quantiles"] = dict(zip([str(a) for a in qs], [round(float(b), 4) for b in q]))
bands = [0, 0.05, 0.1, 0.15, 0.2, 0.24, 0.25, 0.252, 0.26, 0.28, 0.3, 0.333, 0.4, 0.467, 0.6, 1.0001]
for s in range(3):
    h, _ = np.histogram(P[:, s], bins=bands)
    print(f"P{s + 1} share of tokens per band: " + "  ".join(f"[{bands[i]:g},{bands[i + 1]:g}) {100 * h[i] / n:.1f}%" for i in range(len(h)) if h[i]))
    out[f"P{s + 1}_bands"] = {f"{bands[i]:g}-{bands[i + 1]:g}": round(100 * h[i] / n, 2) for i in range(len(h))}
fine = np.arange(0.20, 0.3401, 0.005)
h, _ = np.histogram(P[:, 0], bins=fine)
print("P1 between 0.20 and 0.34 in steps of 0.005 (% of tokens): " + " ".join(f"{fine[i]:.3f}:{100 * h[i] / n:.1f}" for i in range(len(h))))
out["P1_fine"] = {f"{fine[i]:.3f}": round(100 * h[i] / n, 2) for i in range(len(h))}


# 2. predicted loops for a router, against the measured ones
def loops(q1, q2, q3):
    e1 = P[:, 0] >= q1
    e2 = ~e1 & (P[:, 1] >= q2)
    e3 = ~e1 & ~e2 & (P[:, 2] >= q3)
    frac = [e1.mean(), e2.mean(), e3.mean()]
    frac.append(1 - sum(frac))
    return sum((i + 1) * f for i, f in enumerate(frac)), frac


rows = []
for line in open(f"{WORK}/evaluations.jsonl"):
    e = json.loads(line)
    if e.get("status") != "ok":
        continue
    q = e["thresholds"][:3]
    pred, frac = loops(*q)
    rows.append((q, e["exec_loops"], pred, frac, e["acc_pct"]))
print("\nrouter (q1, q2, q3)      measured loops  predicted  exit share at loop 1/2/3/4 (predicted)   accuracy")
for q, meas, pred, frac, acc in rows:
    print(f"({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})   {meas:.3f}          {pred:.3f}     " + " / ".join(f"{100 * f:.0f}%" for f in frac) + f"      {acc:.1f}")
err = np.array([p - m for _, m, p, _, _ in rows])
print(f"prediction error: mean {err.mean():+.3f} loops, mean absolute {np.abs(err).mean():.3f}, max absolute {np.abs(err).max():.3f}")
out["loops_pred"] = [{"q": q, "measured": round(m, 4), "predicted": round(float(p), 4), "exit_share": [round(float(f), 4) for f in fr], "acc": a} for q, m, p, fr, a in rows]

# 3. which tokens leave after loop 1
for thr in (0.3, 0.252, 0.25, 0.2):
    m = P[:, 0] >= thr
    c = collections.Counter(t for t, k in zip(toks, m) if k)
    tot = collections.Counter(toks)
    top = ", ".join(f"{t!r} {c[t]}/{tot[t]}" for t, _ in c.most_common(12))
    print(f"\nq1 = {thr}: {100 * m.mean():.1f} % of tokens leave after loop 1; most frequent next tokens (left/total): {top}")
json.dump(out, open(f"{WORK}/gate_analysis.json", "w"), indent=1)
print("\nwrote", f"{WORK}/gate_analysis.json")
