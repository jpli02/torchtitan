"""Grade IFBench generations (depth_eval.py --task ifb --out GEN) with the official verifiers (strict and loose).
Runs inside the sandboxed ifbench-grader image (boptim_lcb/ifbench_grader.Dockerfile), e.g.

  docker run --rm --network none -v STAGE:/w ifbench-grader python /w/ifb_grade.py /w/ifbench.jsonl /w/GEN.jsonl /w/OUT.json

Prints: GRADED n=... strict_prompt=... strict_inst=... loose_prompt=... loose_inst=...
(prompt-level = every constraint of the prompt satisfied; instruction-level = fraction of constraints satisfied).
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("IFBENCH_SRC", "/app"))  # the checkout, /app inside the ifbench-grader image
import evaluation_lib as ev  # noqa: E402

data_path, gen_path, out_path = sys.argv[1:4]
rows = [json.loads(l) for l in open(data_path) if l.strip()]
gens = [json.loads(l) for l in open(gen_path) if l.strip()]
summary = gens[0].get("summary") if "summary" in gens[0] else None
gens = [g for g in gens if "i" in g]
resp = {rows[g["i"]]["prompt"]: (g.get("pred") or "") for g in gens}
inputs = [x for x in ev.read_prompt_list(data_path) if x.prompt in resp]

res, per_prompt = {}, {}
for name, fn in (("strict", ev.test_instruction_following_strict), ("loose", ev.test_instruction_following_loose)):
    outs = [fn(inp, resp) for inp in inputs]
    inst = [b for o in outs for b in o.follow_instruction_list]
    res[name] = {"prompt_acc": sum(o.follow_all_instructions for o in outs) / len(outs), "inst_acc": sum(inst) / len(inst)}
    for inp, o in zip(inputs, outs):
        per_prompt.setdefault(inp.key, {"key": inp.key, "instruction_id_list": list(inp.instruction_id_list)})[name] = {
            "ok": bool(o.follow_all_instructions), "inst": [bool(b) for b in o.follow_instruction_list]}
out = {"summary": summary, "n": len(inputs), **res, "per_prompt": list(per_prompt.values())}
json.dump(out, open(out_path, "w"))
print(f"GRADED n={len(inputs)} strict_prompt={res['strict']['prompt_acc']:.3f} strict_inst={res['strict']['inst_acc']:.3f} "
      f"loose_prompt={res['loose']['prompt_acc']:.3f} loose_inst={res['loose']['inst_acc']:.3f}", flush=True)
