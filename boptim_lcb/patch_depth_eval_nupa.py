"""Add --task nupa (NUPA, ICLR 2025: number understanding and processing) to depth_eval.py (idempotent): the dataset's own prompt
("Directly return the answer ... <question> ="), the answer is a bare number, exact match after trimming (numeric equality accepted
for integers and floats; fractions and scientific notation compared as written, with a numeric fallback)."""
p = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/depth_eval.py"
s = open(p).read()
if '"nupa"' in s:
    print("already patched")
    raise SystemExit(0)
reps = [
    ('"ifb", "lcbt", "tot", "tbench", "bfcl"], required=True)', '"ifb", "lcbt", "tot", "tbench", "bfcl", "nupa"], required=True)'),
    ('    if args.task == "bfcl":  # BFCL (ICML 2025): the official prompt-mode prompt, built by bfcl_tool.py prep\n        return r["prompt"].rstrip()\n',
     '    if args.task == "bfcl":  # BFCL (ICML 2025): the official prompt-mode prompt, built by bfcl_tool.py prep\n        return r["prompt"].rstrip()\n'
     '    if args.task == "nupa":  # NUPA (ICLR 2025): the dataset prompt ends with "=" and asks for the bare number\n        return r["prompt"].rstrip()\n'),
    ('    if args.task == "tot":\n        return tot_json(gen)\n',
     '    if args.task == "tot":\n        return tot_json(gen)\n    if args.task == "nupa":  # the first number-like chunk of the answer\n'
     '        g = gen.split("</think>")[-1].strip().replace(",", "")\n'
     '        m = re.search(r"-?\\d[\\d./eE+\\-]*", g)\n'
     '        return m.group(0).rstrip(".") if m else None\n'),
    ('    if args.task == "tot":\n        return tot_correct(pred, gold)\n',
     '    if args.task == "tot":\n        return tot_correct(pred, gold)\n    if args.task == "nupa":\n'
     '        if pred is None:\n            return False\n'
     '        a, b = pred.strip(), str(gold).strip().replace(",", "")\n'
     '        if a == b:\n            return True\n'
     '        if "/" in a or "/" in b:\n            return False\n'
     '        try:\n            from decimal import Decimal\n            return Decimal(a) == Decimal(b)\n        except Exception:\n            return False\n'),
    ('    if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and r.get("task")):',
     '    if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench", "nupa") or (args.task == "crux" and r.get("task")):'),
    ('if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and recs and recs[0].get("subtask")):',
     'if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench", "nupa") or (args.task == "crux" and recs and recs[0].get("subtask")):'),
]
for old, new in reps:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("depth_eval: task nupa added")
