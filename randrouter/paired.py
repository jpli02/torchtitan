"""Paired comparison of a real-execution run with the released model's runs on the same LiveCodeBench problems.

References (true decode-time early exit, plain copy fill, graded with the official tests): full depth and the released gate at
0.7 / 0.5 / 0.3, each pooled from the three slices of the 342 problems (wcl_*.graded.json), and the router (0.25, 0.3, 0.3)
(first proposal of both LLM optimizers). For every reference: problems both solve, only the new run solves, only the reference
solves, and the exact two-sided sign test on the discordant problems.

  python randrouter/paired.py RUN_DIR_OR_GRADED_JSON [...]
"""
import json
import math
import os
import sys

STAGE = "/srv/scratch/jli199/lcb_grade_stage"
REFS = {
    "full depth": [f"{STAGE}/wcl_full.graded.json", f"{STAGE}/wcl_full_o120.graded.json", f"{STAGE}/wcl_full_o240.graded.json"],
    "gate 0.7": [f"{STAGE}/wcl_gate0.7.graded.json", f"{STAGE}/wcl_gate0.7_o120.graded.json", f"{STAGE}/wcl_gate0.7_o240.graded.json"],
    "gate 0.5": [f"{STAGE}/wcl_gate0.5.graded.json", f"{STAGE}/wcl_gate0.5_o120.graded.json", f"{STAGE}/wcl_gate0.5_o240.graded.json"],
    "gate 0.3": [f"{STAGE}/wcl_gate0.3.graded.json", f"{STAGE}/wcl_gate0.3_o120.graded.json", f"{STAGE}/wcl_gate0.3_o240.graded.json"],
    "(0.25, 0.3, 0.3)": ["/home/jli199/boptim_scratch/bo_lcb_router_pooled_luna6/q0.250-0.300-0.300_61836/graded.json"],
}


def passed(paths):
    out = {}
    for p in paths:
        for r in json.load(open(p))["results"]:
            out[r["question_id"]] = bool(r["passed"])
    return out


def sign_test(a, b):
    """Exact two-sided binomial test of a against b discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def compare(new, log=print):
    res = {}
    for name, paths in REFS.items():
        if not all(os.path.exists(p) for p in paths):
            continue
        ref = passed(paths)
        common = [q for q in new if q in ref]
        both = sum(new[q] and ref[q] for q in common)
        only_new = sum(new[q] and not ref[q] for q in common)
        only_ref = sum(ref[q] and not new[q] for q in common)
        p = sign_test(only_new, only_ref)
        res[name] = {"n": len(common), "ref_pass": both + only_ref, "new_pass": both + only_new, "only_new": only_new, "only_ref": only_ref, "p": p}
        log(f"  vs {name:<18} on {len(common)} problems: reference {100 * (both + only_ref) / len(common):.1f} %, this run {100 * (both + only_new) / len(common):.1f} %; "
            f"only this run solves {only_new}, only the reference {only_ref}; sign test p = {p:.3f}")
    return res


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        path = arg if arg.endswith(".json") else os.path.join(arg, "graded.json")
        new = passed([path])
        print(f"{arg}: {sum(new.values())}/{len(new)} = {100 * sum(new.values()) / len(new):.1f} %")
        compare(new)
