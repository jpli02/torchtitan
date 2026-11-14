"""Exact compute accounting for decode-time early exit (idempotent):
  1. the deployed early-exit model files (ouro_base_kvshare, and any other serving dir given as an argument) count the loops actually
     executed for every decoded token: model.model._dee_exec_sum / _dee_exec_n. The existing _exit_loops_decode counter is derived
     from the padded per-step lists and reads about one loop too high for tokens that exit early.
  2. depth_eval.py reports it as "exec_loops" (and "exec_n") in the RESULT summary.
  3. make_kvshare_modeling.py gets the same replacement, so regenerated serving dirs carry the counter.
The patched model file is compiled before it replaces the deployed one.
  python boptim_lcb/patch_exec_counter.py [extra_serving_dir ...]"""
import os
import py_compile
import sys

WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"
ANCHOR = "        return (\n            BaseModelOutputWithPast(\n                last_hidden_state=hidden_states,"
COUNTER = ("        if do_dee:\n"
           "            # exact accounting: recurrence loops actually executed for this decoded token (current_ut is the exit step)\n"
           "            self._dee_exec_sum = getattr(self, \"_dee_exec_sum\", 0) + (current_ut + 1)\n"
           "            self._dee_exec_n = getattr(self, \"_dee_exec_n\", 0) + 1\n\n")

for d in ["/home/jli199/boptim_scratch/ouro_base_kvshare"] + sys.argv[1:]:
    p = f"{d}/modeling_ouro.py"
    s = open(p, encoding="utf-8").read()
    if "_dee_exec_sum" in s:
        print("already patched:", p)
        continue
    assert s.count(ANCHOR) == 1, (p, s.count(ANCHOR))
    s2 = s.replace(ANCHOR, COUNTER + ANCHOR, 1)
    tmp = p + ".new"
    open(tmp, "w", encoding="utf-8").write(s2)
    py_compile.compile(tmp, doraise=True)
    open(p + ".before_exec_counter", "w", encoding="utf-8").write(s)
    os.replace(tmp, p)
    print("patched:", p)

# depth_eval.py: report the counter
p = f"{WT}/depth_eval.py"
s = open(p).read()
if "exec_loops" not in s:
    old = '''           "wall_s": dt, "tok_s": total_tok / dt, "max_new": args.max_new}\n'''
    assert s.count(old) == 1, s.count(old)
    new = old + ('''_inner = getattr(model, "model", None)  # exact executed-loop count under decode-time early exit (see boptim_lcb/patch_exec_counter.py)\n'''
                 '''if getattr(_inner, "_dee_exec_n", 0):\n'''
                 '''    summary["exec_loops"] = _inner._dee_exec_sum / _inner._dee_exec_n\n'''
                 '''    summary["exec_n"] = _inner._dee_exec_n\n''')
    open(p, "w").write(s.replace(old, new, 1))
    print("patched: depth_eval.py")
else:
    print("already patched: depth_eval.py")

# make_kvshare_modeling.py: keep the generator in step
p = f"{WT}/make_kvshare_modeling.py"
s = open(p).read()
if "_dee_exec_sum" not in s:
    old = "for out, weights in OUT.items():"
    if s.count(old) == 1:
        add = ("# exact executed-loop counter for decode-time early exit (boptim_lcb/patch_exec_counter.py)\n"
               "rep(" + repr(ANCHOR) + ",\n    " + repr(COUNTER + ANCHOR) + ")\n\n")
        open(p, "w").write(s.replace(old, add + old, 1))
        print("patched: make_kvshare_modeling.py")
    else:
        print("make_kvshare_modeling.py: write loop not found, generator left unchanged")
else:
    print("already patched: make_kvshare_modeling.py")
