"""Best pass@1 over the full 80-task set.

The 80-task evals were sharded across GPUs (s0/s1/s2), so a model's score is the
union of its shards. Reports solved tasks, coverage (shards can miss tasks), and
per-test partial credit alongside, since binary at this scale has a +-1 floor.
"""
import json, glob, collections

MODELS = {
    "pretrained": "e80_pretrained_s*",
    "sft20k":     "e80_sft20k_s*",
    "continue10k": "e80_c10k80_s*",
    "repro11k":   "e80_repro80_s*",
}

for name, pat in MODELS.items():
    seen = {}                    # task -> solved bool (any attempt)
    tests_p = tests_n = 0
    modes = collections.Counter()
    trials = 0
    for f in glob.glob(f"/tmp/tb_runs/{pat}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        t = d.get("task_id") or "?"
        trials += 1
        ok = bool(d.get("is_resolved"))
        seen[t] = seen.get(t, False) or ok
        modes[d.get("failure_mode") or "none"] += 1
        pr = d.get("parser_results") or {}
        if isinstance(pr, dict) and pr:
            tests_n += len(pr)
            tests_p += sum(1 for v in pr.values()
                           if str(v).lower() in ("passed", "true", "ok"))
    if not seen:
        continue
    solved = sorted(t for t, v in seen.items() if v)
    print(f"=== {name} ===")
    print(f"  tasks covered: {len(seen)}/80    trials: {trials}")
    print(f"  SOLVED: {len(solved)}/{len(seen)}  -> pass@1 = {len(solved)/80:.3f} of 80")
    print(f"  tasks: {solved}")
    print(f"  per-test partial credit: {tests_p}/{tests_n} "
          f"({100*tests_p/max(tests_n,1):.1f}%)")
    print(f"  failure modes: {dict(modes.most_common(5))}")
    print()
