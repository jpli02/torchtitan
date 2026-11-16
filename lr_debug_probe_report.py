"""Summary of an optimizer-probe file (torchtitan/components/opt_probe.py): applied update against exact fp32 AdamW.

  python lr_debug_probe_report.py <opt_probe.jsonl> [first_step] [last_step]
"""
import collections
import json
import statistics
import sys

path = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
rows = collections.defaultdict(list)
meta = {}
for line in open(path):
    r = json.loads(line)
    if r.get("probe") == "start":
        meta = {t["name"]: t for t in r["tensors"]}
        continue
    if lo <= r["step"] <= hi:
        rows[r["name"]].append(r)
if not rows:
    sys.exit("no probe rows")
steps = max(len(v) for v in rows.values())
last = max(r["step"] for v in rows.values() for r in v)
print(f"{path.split('/')[-1]}: {len(rows)} tensors, steps {lo}..{last} ({steps} rows each)")
print("  applied step / exact AdamW step: 'norm x' = ratio of update norms (median over steps, and the largest); "
      "'>1.5lr' = share of elements stepping more than 1.5 x lr (mean over steps); 'max' = largest single step in units of lr")
print(f"{'tensor':46s} {'8bit':4s} {'norm x med':>10s} {'norm x max':>10s} {'cos med':>8s} {'>1.5lr':>8s} {'>10lr':>8s} {'max/lr':>9s} "
      f"{'exact max/lr':>12s} {'moved x':>8s} {'moved cos':>9s}")
agg = []
for name, v in rows.items():
    ratio = [r["u_real"] / r["u_exact"] for r in v if r["u_exact"] > 0]
    fin = v[-1]
    moved = fin["d_real"] / fin["d_exact"] if fin["d_exact"] > 0 else float("nan")
    q = "yes" if meta.get(name, {}).get("quantised") else "no"
    print(f"{name[-46:]:46s} {q:4s} {statistics.median(ratio):10.2f} {max(ratio):10.1f} {statistics.median(r['u_cos'] for r in v):8.3f} "
          f"{statistics.mean(r['share_over_1p5lr'] for r in v):8.4f} {statistics.mean(r['share_over_10lr'] for r in v):8.5f} "
          f"{max(r['max_step_in_lr'] for r in v):9.1f} {max(r['max_exact_in_lr'] for r in v):12.2f} {moved:8.2f} {fin['d_cos']:9.3f}")
    if q == "yes":
        agg.append((statistics.median(ratio), moved, fin["d_cos"], statistics.mean(r["share_over_1p5lr"] for r in v)))
if agg:
    print(f"quantised tensors: median step-norm ratio {statistics.median(a[0] for a in agg):.2f}, median displacement ratio after {last} steps "
          f"{statistics.median(a[1] for a in agg):.2f} (largest {max(a[1] for a in agg):.1f}), median displacement cosine "
          f"{statistics.median(a[2] for a in agg):.3f}, mean share of elements over 1.5 x lr {statistics.mean(a[3] for a in agg):.4f}")
