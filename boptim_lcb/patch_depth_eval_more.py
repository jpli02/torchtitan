"""Add three tasks to depth_eval.py (idempotent) and let crux rows carry a full call:
  lcbt    LiveCodeBench test-output prediction: problem statement + call -> expected output as a Python/JSON literal in \\boxed{}
  tot     Test of Time: the dataset's own prompt; the answer is a JSON object, compared field by field with the label
  tbench  TableBench (NumericalReasoning + FactChecking): official direct-prompting format, last line 'Final Answer: ...', exact match
"""
p = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/depth_eval.py"
s = open(p).read()
if '"lcbt"' in s:
    print("already patched")
    raise SystemExit(0)
reps = [
    ('"lbr", "lbm", "lbd", "lbl", "ifb"], required=True)', '"lbr", "lbm", "lbd", "lbl", "ifb", "lcbt", "tot", "tbench"], required=True)'),
    # crux rows may carry the whole call (LiveCodeBench code execution)
    ('''                f"```python\\n{r['code']}\\n```\\n\\nInput: `f({r['input']})`\\n\\n"''',
     '''                f"```python\\n{r['code']}\\n```\\n\\nInput: `{r.get('call') or 'f(' + r['input'] + ')'}`\\n\\n"'''),
    ('''    if args.task == "lbr":  # LiveBench reasoning: the official prompt asks for the answer in **bold**; we also accept \\boxed{}
''',
     '''    if args.task == "lcbt":  # LiveCodeBench test-output prediction (ICLR 2025)
        return ("You are helping to write a test case for the following programming problem.\\n\\n### Problem\\n" + r["question"].rstrip()
                + "\\n\\n### Signature\\n```python\\n" + r["starter_code"].rstrip() + "\\n```\\n\\nWhat is the correct output of the call below?\\n\\n"
                f"`{r['call']}`\\n\\nReason step by step, then give the expected output as a Python literal within \\\\boxed{{}}.")
    if args.task == "tot":  # Test of Time (ICLR 2025): the dataset prompt already specifies the JSON answer format
        return r["prompt"].rstrip()
    if args.task == "tbench":  # TableBench (AAAI 2025), official direct-prompting instruction
        return ("You are a table analyst. Your task is to answer questions based on the table content.\\n\\n"
                "The answer should follow the format below:\\n[Answer Format]\\nFinal Answer: AnswerName1, AnswerName2...\\n\\n"
                "Ensure the final answer format is the last output line and can only be in the \\"Final Answer: AnswerName1, AnswerName2...\\" form, "
                "no other form. Ensure the \\"AnswerName\\" is a number or entity name, as short as possible, without any explanation.\\n\\n"
                "Give the final answer to the question directly without any explanation.\\n\\nRead the table below in JSON format:\\n[TABLE]\\n"
                + r["table"] + "\\n\\nLet's get start!\\nQuestion: " + r["question"])
    if args.task == "lbr":  # LiveBench reasoning: the official prompt asks for the answer in **bold**; we also accept \\boxed{}
'''),
    ('''def extract(gen, r=None):
    if args.task in ("lcb", "bcb", "codepro"):
        return last_code_block(gen)
''',
     '''def tot_json(gen):
    """Last JSON-looking object of the response (Test of Time answers are JSON objects)."""
    g = gen.split("</think>")[-1]
    for m in reversed(list(re.finditer(r"\\{", g))):
        depth = 0
        for k in range(m.start(), len(g)):
            if g[k] == "{":
                depth += 1
            elif g[k] == "}":
                depth -= 1
                if depth == 0:
                    cand = g[m.start():k + 1]
                    for load in (json.loads, lambda x: __import__("ast").literal_eval(x)):
                        try:
                            d = load(cand)
                            if isinstance(d, dict):
                                return d
                        except Exception:
                            pass
                    break
    return None


def tot_norm(v):
    if isinstance(v, (list, tuple)):
        return [tot_norm(x) for x in v]
    s = str(v).strip().strip("'\\"").lower()
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except Exception:
        return re.sub(r"\\s+", " ", s)


def tot_correct(pred, label):
    if not isinstance(pred, dict):
        return False
    import ast
    try:
        lab = ast.literal_eval(label) if str(label).strip().startswith("{") else {"answer": label}
    except Exception:
        try:
            lab = json.loads(label)
        except Exception:
            lab = {"answer": label}
    for k, v in lab.items():
        if k == "explanation":
            continue
        pv = pred.get(k)
        if pv is None and k == "unordered_list":
            pv = pred.get("unorered_list")  # the dataset's own prompt spells the key this way
        if pv is None:
            return False
        if k == "unordered_list" or k == "unorered_list":
            if not isinstance(pv, (list, tuple)) or sorted(tot_norm(pv)) != sorted(tot_norm(v)):
                return False
        elif isinstance(v, (list, tuple)):
            if not isinstance(pv, (list, tuple)) or tot_norm(pv) != tot_norm(v):
                return False
        else:
            a, b = tot_norm(pv), tot_norm(v)
            if a != b and a.replace(" ", "") != b.replace(" ", ""):
                return False
    return True


def tbench_norm(s):
    s = str(s).strip().strip(".").lower().replace("%", "").replace("$", "").replace(",", " , ")
    parts = [x.strip() for x in s.split(",") if x.strip()]
    out = []
    for x in parts:
        try:
            f = float(x.replace(" ", ""))
            out.append(str(int(f)) if f == int(f) else str(round(f, 2)))
        except Exception:
            out.append(re.sub(r"\\s+", " ", x))
    return out


def tbench_correct(pred, gold):
    if pred is None:
        return False
    a, b = tbench_norm(pred), tbench_norm(gold)
    if a == b:
        return True
    return len(b) > 1 and "".join(a).replace(" ", "") == "".join(b).replace(" ", "")  # thousands separators split one number


def extract(gen, r=None):
    if args.task in ("lcb", "bcb", "codepro"):
        return last_code_block(gen)
    if args.task == "tot":
        return tot_json(gen)
    if args.task == "tbench":
        m = re.findall(r"Final Answer:\\s*(.+)", gen.split("</think>")[-1])
        return m[-1].strip() if m else None
'''),
    ('''    if args.task == "crux":
        return pred is not None and crux_equal(pred, gold)
''',
     '''    if args.task == "crux":
        return pred is not None and crux_equal(pred, gold)
    if args.task == "lcbt":  # expected outputs are JSON literals (true/false/null) or Python literals
        if pred is None:
            return False
        g = str(gold)
        for a, b in (("true", "True"), ("false", "False"), ("null", "None")):
            g = re.sub(rf"\\b{a}\\b", b, g)
        return crux_equal(pred, g) or crux_equal(re.sub(r"\\btrue\\b", "True", re.sub(r"\\bfalse\\b", "False", pred)), g)
    if args.task == "tot":
        return tot_correct(pred, gold)
    if args.task == "tbench":
        return tbench_correct(pred, gold)
'''),
    ('''    gold = {"crux": r.get("output"), "zebra": r.get("solution")}.get(args.task, r.get("answer"))''',
     '''    gold = {"crux": r.get("output"), "lcbt": r.get("output"), "tot": r.get("label"), "zebra": r.get("solution")}.get(args.task, r.get("answer"))'''),
    ('''    if args.task in ("lbr", "lbm", "lbd", "lbl"):
        rec.update({"question_id": r.get("question_id"), "subtask": r.get("task")})''',
     '''    if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and r.get("task")):
        rec.update({"question_id": r.get("question_id"), "subtask": r.get("task")})'''),
    ('''if args.task in ("lbr", "lbm", "lbd", "lbl"):
    by_t = {}''',
     '''if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and recs and recs[0].get("subtask")):
    by_t = {}'''),
]
for old, new in reps:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched depth_eval.py: crux call field, lcbt, tot, tbench")
