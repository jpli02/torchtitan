"""Add --task ifb (IFBench, NeurIPS 2025 D&B) to depth_eval.py: prompt as is, response = text without the reasoning block,
graded externally by ifb_grade.py (official strict/loose verifiers)."""
p = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/depth_eval.py"
s = open(p).read()
if '"ifb"' in s:
    print("already patched")
    raise SystemExit(0)
reps = [
    ('"lbr", "lbm", "lbd", "lbl"], required=True)', '"lbr", "lbm", "lbd", "lbl", "ifb"], required=True)'),
    ('    # LiveCodeBench, official prompt shape\n',
     '    if args.task == "ifb":  # IFBench: the prompt as is; the verifiers check the whole response\n        return r["prompt"].rstrip()\n'
     '    # LiveCodeBench, official prompt shape\n'),
    ('def extract(gen, r=None):\n    if args.task in ("lcb", "bcb", "codepro"):\n        return last_code_block(gen)\n',
     'def extract(gen, r=None):\n    if args.task in ("lcb", "bcb", "codepro"):\n        return last_code_block(gen)\n'
     '    if args.task == "ifb":  # the response without the reasoning block (a truncated, unclosed block stays as is)\n'
     '        g = re.sub(r"<think>.*?</think>", "", gen, flags=re.S).strip()\n'
     '        return g or None\n'),
    ('    if args.task in ("lcb", "bcb", "codepro"):\n        return False  # graded separately',
     '    if args.task in ("lcb", "bcb", "codepro", "ifb"):\n        return False  # graded separately'),
    ('    if args.task == "bcb":\n        rec.update({"task_id": r.get("task_id")})\n',
     '    if args.task == "bcb":\n        rec.update({"task_id": r.get("task_id")})\n'
     '    if args.task == "ifb":\n        rec.update({"key": r.get("key"), "instruction_id_list": r.get("instruction_id_list")})\n'),
]
for old, new in reps:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched depth_eval.py with task ifb")
