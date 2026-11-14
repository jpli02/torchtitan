"""Add --task bfcl (BFCL, ICML 2025) to depth_eval.py (idempotent): the official prompt-mode prompt built by bfcl_tool.py prep; the
response is the function-call list; graded externally by the official AST checker (bfcl_tool.py grade, inside the sandbox image)."""
p = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/depth_eval.py"
s = open(p).read()
if '"bfcl"' in s:
    print("already patched")
    raise SystemExit(0)
reps = [
    ('"ifb", "lcbt", "tot", "tbench"], required=True)', '"ifb", "lcbt", "tot", "tbench", "bfcl"], required=True)'),
    ('    if args.task == "tot":  # Test of Time (ICLR 2025): the dataset prompt already specifies the JSON answer format\n',
     '    if args.task == "bfcl":  # BFCL (ICML 2025): the official prompt-mode prompt, built by bfcl_tool.py prep\n        return r["prompt"].rstrip()\n'
     '    if args.task == "tot":  # Test of Time (ICLR 2025): the dataset prompt already specifies the JSON answer format\n'),
    ('    if args.task == "tot":\n        return tot_json(gen)\n',
     '    if args.task == "tot":\n        return tot_json(gen)\n    if args.task == "bfcl":  # the function-call list; checked by the official AST checker (bfcl_tool.py grade)\n'
     '        g = gen.split("</think>")[-1].strip()\n        return g or None\n'),
    ('    if args.task in ("lcb", "bcb", "codepro", "ifb"):\n        return False  # graded separately',
     '    if args.task in ("lcb", "bcb", "codepro", "ifb", "bfcl"):\n        return False  # graded separately'),
    ('    if args.task == "ifb":\n        rec.update({"key": r.get("key"), "instruction_id_list": r.get("instruction_id_list")})\n',
     '    if args.task == "ifb":\n        rec.update({"key": r.get("key"), "instruction_id_list": r.get("instruction_id_list")})\n'
     '    if args.task == "bfcl":\n        rec.update({"id": r.get("id"), "category": r.get("category")})\n'),
]
for old, new in reps:
    assert s.count(old) == 1, (s.count(old), old[:60])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("depth_eval: task bfcl added")
