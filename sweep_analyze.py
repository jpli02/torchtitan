"""Router early-exit sweep: does efficiency buy ability, or trade against it?

The BO experiment we want to run needs a win-win region to exist. Ouro's router
exits the UT loop early per token, cutting compute; if that also buys more agent
turns inside the 720s budget -- and 65/80 failures are agent_timeout -- then
pass rate should RISE as avg_loops FALLS, over some range.

Reports, per threshold:
  avg_loops      mean UT steps/token (1-4), from the server's ouro_avg_loops.
                 Averages over thousands of tokens, so near-noiseless.
  partial credit fraction of per-test outcomes passed. ~6x denser than binary
                 pass; sigma ~3% across near-identical models vs binary's 17-21%.
                 The pretrained no-op agent floors at 42.4%, so subtract that.
  resolved       binary tasks solved, for reference only -- too noisy to steer on.
"""
import json, glob, re, collections

FLOOR = 0.424  # pass-by-default floor, measured on the pretrained no-op agent

RUNS = [("thr02", 0.2), ("thr04", 0.4), ("thr06", 0.6),
        ("thr08", 0.8), ("thr10", 1.0)]

print(f"{'thr':>5} {'avg_loops':>10} {'partial':>9} {'-floor':>8} "
      f"{'resolved':>9} {'timeout':>8} {'turns':>7}")
print("-" * 62)
rows = []
for name, thr in RUNS:
    loops, turns = [], 0
    tp = tn = solved = trials = timeouts = 0
    for rf in glob.glob(f"/tmp/tb_runs/{name}/*/*/results.json"):
        try:
            d = json.load(open(rf))
        except Exception:
            continue
        trials += 1
        solved += bool(d.get("is_resolved"))
        timeouts += (d.get("failure_mode") == "agent_timeout")
        pr = d.get("parser_results") or {}
        if isinstance(pr, dict) and pr:
            tn += len(pr)
            tp += sum(1 for v in pr.values()
                      if str(v).lower() in ("passed", "true", "ok"))
        for f in glob.glob(f"{rf.rsplit('/',1)[0]}/agent-logs/episode-*/debug.json"):
            turns += 1
            try:
                j = json.loads(json.load(open(f))["original_response"])
                al = j.get("ouro_avg_loops")
                if al:
                    loops.append(float(al))
            except Exception:
                pass
    if not trials:
        print(f"{thr:>5} {'(no results yet)':>10}")
        continue
    ml = sum(loops) / len(loops) if loops else float("nan")
    pc = tp / tn if tn else 0.0
    rows.append((thr, ml, pc, solved, trials))
    print(f"{thr:>5} {ml:>10.3f} {pc:>8.1%} {pc-FLOOR:>+8.1%} "
          f"{solved:>4}/{trials:<4} {timeouts:>8} {turns:>7}")

if len(rows) >= 2:
    base = [r for r in rows if r[0] == 1.0]
    if base:
        _, bl, bp, _, _ = base[0]
        print(f"\nvs full recurrence (thr=1.0, loops={bl:.3f}, partial={bp:.1%}):")
        for thr, ml, pc, _, _ in rows:
            if thr == 1.0:
                continue
            print(f"  thr={thr}: compute {(ml/bl-1)*100:+.1f}%   "
                  f"ability {(pc-bp)*100:+.1f}pp   "
                  f"{'WIN-WIN' if ml < bl and pc > bp else ''}")
