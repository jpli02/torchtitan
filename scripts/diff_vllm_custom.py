#!/usr/bin/env python3
"""Diff vLLM vs custom-generator HumanEval solutions on the same problems.

Both pipelines use EvalPlus's canonical prompt, so per-problem differences
isolate *implementation/decoding* (vLLM's ouro.py vs torchtitan OuroModel) from
prompt effects. Re-scores each sanitized solution against the base HumanEval
test (per-problem timeout) and buckets agreement; prints the problems vLLM
solves that the custom generator misses (and vice versa) side by side.

Usage: python scripts/diff_vllm_custom.py <vllm-sanitized.jsonl> <custom-sanitized.jsonl>
"""
import json, sys, signal
from evalplus.data import get_human_eval_plus

prob = get_human_eval_plus()
load = lambda p: {json.loads(l)["task_id"]: json.loads(l)["solution"]
                  for l in open(p) if l.strip()}
V, C = load(sys.argv[1]), load(sys.argv[2])

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
def ok(tid, sol):
    p = prob[tid]; src = sol + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
    signal.alarm(10)
    try:
        exec(compile(src, "<s>", "exec"), {}); return True
    except Exception:
        return False
    finally:
        signal.alarm(0)

ids = sorted(set(V) & set(C), key=lambda x: int(x.split("/")[1]))
vp = {t: ok(t, V[t]) for t in ids}
cp = {t: ok(t, C[t]) for t in ids}
vonly = [t for t in ids if vp[t] and not cp[t]]
conly = [t for t in ids if cp[t] and not vp[t]]
print(f"common={len(ids)}  vLLM pass@1={sum(vp.values())/len(ids):.3f}  "
      f"custom pass@1={sum(cp.values())/len(ids):.3f}")
print(f"both_pass={sum(1 for t in ids if vp[t] and cp[t])}  "
      f"both_fail={sum(1 for t in ids if not vp[t] and not cp[t])}  "
      f"vLLM_only={len(vonly)}  custom_only={len(conly)}")
print(f"\nvLLM solves, custom misses: {vonly}")
print(f"custom solves, vLLM misses: {conly}")
for t in (vonly + conly)[:4]:
    tag = "vLLM-only" if t in vonly else "custom-only"
    print("=" * 72, f"{t}  ({tag})")
    print("--- vLLM ---\n" + V[t][:600])
    print("--- custom ---\n" + C[t][:600])
