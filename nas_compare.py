"""Compare optimizers on the 6-D Ouro router NAS.

Final-best alone is a weak summary: it throws away the trajectory and hides how
much budget each optimizer wasted on infeasible points. Reports the anytime
best-so-far curve, the feasibility rate, and iterations-to-target instead.

Infeasibility matters especially here. The earlier 2-D comparison was decided by
it: gp_hedge spent 3/10 evaluations on configs that missed the accuracy floor
(scored with the 10.0 penalty) while the LLM optimizers spent 0/10, so what
looked like a search-quality gap was largely a constraint-handling gap.
"""
import re
import statistics

LOGS = {
    "gp_hedge": "/home/jli199/.claude/jobs/c0d2da0a/tmp/nas_gphedge.log",
    "random":   "/home/jli199/.claude/jobs/c0d2da0a/tmp/nas_random.log",
    "claude":   "/home/jli199/.claude/jobs/c0d2da0a/tmp/nas_claude.log",
}
PAT = re.compile(
    r"eval (\d+): gamma=([0-9.e+-]+) thr=([0-9.]+)\s+pass@1=([0-9.]+)\s+"
    r"avg_loops=([0-9.]+)\s+accepted=(\w+)\s+time=([0-9.]+)s\s+value=([0-9.]+)")
PENALTY = 10.0
BASELINE_LOOPS = 4.0
TARGET = 1.5  # "found a strong architecture": avg_loops <= 1.5

rows = {}
for name, path in LOGS.items():
    recs = []
    try:
        for m in PAT.finditer(open(path).read()):
            recs.append(dict(i=int(m.group(1)), pass1=float(m.group(4)),
                             loops=float(m.group(5)),
                             ok=m.group(6) == "True", t=float(m.group(7)),
                             val=float(m.group(8))))
    except FileNotFoundError:
        continue
    if recs:
        rows[name] = recs

print(f"{'optimizer':<10} {'evals':>6} {'feasible':>9} {'best':>8} "
      f"{'vs 4.0':>8} {'iters->1.5':>11} {'med time':>9}")
print("-" * 68)
for name, recs in rows.items():
    feas = [r for r in recs if r["val"] < PENALTY]
    best = min((r["val"] for r in feas), default=float("nan"))
    hit = next((r["i"] + 1 for r in recs
                if r["val"] < PENALTY and r["val"] <= TARGET), None)
    print(f"{name:<10} {len(recs):>6} {len(feas):>4}/{len(recs):<4} {best:>8.3f} "
          f"{(best/BASELINE_LOOPS-1)*100:>7.1f}% {str(hit or '-'):>11} "
          f"{statistics.median(r['t'] for r in recs):>8.0f}s")

print("\nbest-so-far (feasible only), by evaluation:")
marks = [1, 3, 5, 10, 15, 20, 25]
print(f"{'optimizer':<10}" + "".join(f"{m:>8}" for m in marks))
for name, recs in rows.items():
    line = f"{name:<10}"
    cur = float("inf")
    curve = {}
    for r in recs:
        if r["val"] < PENALTY:
            cur = min(cur, r["val"])
        curve[r["i"] + 1] = cur
    for m in marks:
        v = curve.get(m, float("inf"))
        line += f"{('-' if v == float('inf') else f'{v:.3f}'):>8}"
    print(line)

print("\naccuracy of the best feasible point (the constraint that matters):")
for name, recs in rows.items():
    feas = [r for r in recs if r["val"] < PENALTY]
    if feas:
        b = min(feas, key=lambda r: r["val"])
        print(f"  {name:<10} loops={b['loops']:.3f}  pass@1={b['pass1']:.3f}")
