"""Binary task success is too coarse to optimize against: at 4/24 the metric
cannot resolve anything smaller than the noise floor. terminal-bench records
per-test outcomes in results.json['parser_results'], which is a much denser
signal -- and also tells us WHY tasks fail (agent_timeout vs test failure).
"""
import json, glob, collections

RUNS = ["clean_basethinking", "clean_continue10k", "clean_repro11k",
        "clean_batching5k"]

for run in RUNS:
    tests_pass = tests_tot = trials = 0
    modes = collections.Counter()
    got_any = 0
    for f in glob.glob(f"/tmp/tb_runs/{run}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        trials += 1
        modes[d.get("failure_mode") or "none"] += 1
        pr = d.get("parser_results") or {}
        if isinstance(pr, dict) and pr:
            n = len(pr)
            p = sum(1 for v in pr.values() if str(v).lower() in ("passed", "true", "ok"))
            tests_tot += n
            tests_pass += p
            if p:
                got_any += 1
    if not trials:
        continue
    print(f"=== {run} ===")
    print(f"  trials {trials}   tests passed {tests_pass}/{tests_tot} "
          f"({100*tests_pass/max(tests_tot,1):.1f}%)   trials w/ >=1 test passed: {got_any}")
    print(f"  failure modes: {dict(modes.most_common(6))}")
    print()
