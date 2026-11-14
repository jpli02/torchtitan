"""Batched greedy evaluation of an Ouro HF checkpoint at a fixed recurrence depth.

Usage: python depth_eval.py CKPT DATA.jsonl --task gsm8k|math --steps K [--n N] [--batch B]
           [--max_new T] [--out OUT.jsonl] [--threshold TH]
Reports accuracy, truncation rate, mean generated tokens, wall-clock and tok/s.
--steps sets total_ut_steps (uniform depth, including 5/6 = beyond training).
--threshold sets the stock early_exit_threshold (gather-only selection; no speedup).
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("data")
ap.add_argument("--task", choices=["gsm8k", "math", "crux", "lcb", "mc", "bcb", "zebra", "codepro", "lbr", "lbm", "lbd", "lbl", "ifb", "lcbt", "tot", "tbench"], required=True)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument("--threshold_steps", default=None, help="per-step CDF thresholds q1,q2,q3 (last step always exits), gather mode")
ap.add_argument("--n", type=int, default=10**9)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--out", default=None)
ap.add_argument("--seed_offset", type=int, default=0, help="skip this many rows first")
ap.add_argument("--cfg", action="append", default=[], help="config override key=value (bool/int/float parsed), e.g. kv_share_decode=true")
ap.add_argument("--prompt_suffix", default="", help="text appended to every prompt (e.g. 'Respond with only the code block, no explanation.')")
ap.add_argument("--assistant_prefix", default="", help="text the assistant turn starts with (forced), e.g. '```python\\n' to skip the reasoning preamble")
args = ap.parse_args()


def _parse_val(v):
    if v.startswith("["):
        return json.loads(v)  # list, e.g. decode_early_exit_threshold=[0.1,0.5,0.5,1.0]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


CFG = {k: _parse_val(v) for k, v in (s.split("=", 1) for s in args.cfg)}

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        m = re.findall(r"####\s*([^\n]+)", s)
        return m[-1].strip() if m else None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        v = float(m[-1] if len(m) == 1 else m[0])
    except ValueError:
        return None
    return round(v, 6)


def norm_math(s):
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("tfrac", "frac"),
                 ("dfrac", "frac"), ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""),
                 (" ", ""), ("\\displaystyle", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    # simple fraction a/b -> \frac{a}{b}
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        s = f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    return s


def last_code_block(s):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", s, re.S)
    if blocks:
        return blocks[-1].strip("\n")
    i = s.rfind("```python")
    return s[i + 9:].strip("\n") if i >= 0 else None


def mc_prompt(r):
    opts = "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i, o in enumerate(r["options"]))
    return (f"Answer the following multiple choice question.\n\n{r['question']}\n\n{opts}\n\n"
            "Think step by step, then give only the letter of the correct option within \\boxed{}.")


def build_prompt(r):
    if args.task in ("gsm8k", "math"):
        return r["question"] + SUFFIX
    if args.task == "mc":
        return mc_prompt(r)
    if args.task == "bcb":
        return (r["instruct_prompt"].rstrip() + "\n\nWrite the complete, self-contained Python code (imports and the full "
                "function) inside a single ```python code block.")
    if args.task == "crux":
        return ("Here is a Python function and an input. Determine the exact output of calling the function on this input.\n\n"
                f"```python\n{r['code']}\n```\n\nInput: `{r.get('call') or 'f(' + r['input'] + ')'}`\n\n"
                "Reason step by step, then give the exact output as a Python literal within \\boxed{}.")
    if args.task == "lcbt":  # LiveCodeBench test-output prediction (ICLR 2025)
        return ("You are helping to write a test case for the following programming problem.\n\n### Problem\n" + r["question"].rstrip()
                + "\n\n### Signature\n```python\n" + r["starter_code"].rstrip() + "\n```\n\nWhat is the correct output of the call below?\n\n"
                f"`{r['call']}`\n\nReason step by step, then give the expected output as a Python literal within \\boxed{{}}.")
    if args.task == "tot":  # Test of Time (ICLR 2025): the dataset prompt already specifies the JSON answer format
        return r["prompt"].rstrip()
    if args.task == "tbench":  # TableBench (AAAI 2025), official direct-prompting instruction
        return ("You are a table analyst. Your task is to answer questions based on the table content.\n\n"
                "The answer should follow the format below:\n[Answer Format]\nFinal Answer: AnswerName1, AnswerName2...\n\n"
                "Ensure the final answer format is the last output line and can only be in the \"Final Answer: AnswerName1, AnswerName2...\" form, "
                "no other form. Ensure the \"AnswerName\" is a number or entity name, as short as possible, without any explanation.\n\n"
                + ("Give the final answer to the question directly without any explanation." if args.assistant_prefix else
                   "Let's think step by step and then give the final answer to the question.")
                + "\n\nRead the table below in JSON format:\n[TABLE]\n"
                + r["table"] + "\n\nLet's get start!\nQuestion: " + r["question"])
    if args.task == "lbr":  # LiveBench reasoning: the official prompt asks for the answer in **bold**; we also accept \boxed{}
        return r["question"].rstrip() + "\n\nAfter your reasoning, also give the final answer within \\boxed{}."
    if args.task in ("lbm", "lbd", "lbl"):  # LiveBench math / data analysis / language: the official prompt as is
        return r["question"].rstrip()
    if args.task == "zebra":  # ZebraLogic grid mode (ICML 2025)
        attrs = ", ".join(r["header"])
        return (f"{r['puzzle'].rstrip()}\n\nSolve the puzzle. Reason step by step, then give the final solution as a JSON object inside a "
                "```json code block, with one key per house, like {\"House 1\": {\"" + r["header"][0] + "\": \"...\", ...}, \"House 2\": {...}, ...}, "
                f"using exactly these attribute names: {attrs}.")
    if args.task == "codepro":  # HumanEval Pro / MBPP Pro: self-invoking code generation (2025)
        return ("Base problem:\n```python\n" + r["raw_problem"].rstrip() + "\n```\n\nNew problem, which builds on the base problem:\n```python\n"
                + r["problem"].rstrip() + "\n```\n\nWrite the complete, self-contained Python code for the new problem (include every helper "
                "you need, such as a solution to the base problem) inside a single ```python code block.")
    if args.task == "ifb":  # IFBench: the prompt as is; the verifiers check the whole response
        return r["prompt"].rstrip()
    # LiveCodeBench, official prompt shape
    q = "### Question:\n" + r["question_content"] + "\n\n"
    if r.get("starter_code", "").strip():
        q += ("### Format: You will use the following starter code to write the solution to the problem and enclose your code "
              f"within delimiters.\n```python\n{r['starter_code']}\n```\n\n")
    else:
        q += ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
              "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads "
              "the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
    return q + "### Answer: (use the provided format with backticks)\n\n"


def last_json_block(s):
    blocks = re.findall(r"```json\s*\n(.*?)```", s, re.S)
    cand = blocks[-1] if blocks else None
    if cand is None:  # fall back to the last balanced {...}
        i = s.rfind("}")
        depth, j = 0, i
        while j >= 0 and i >= 0:
            if s[j] == "}":
                depth += 1
            elif s[j] == "{":
                depth -= 1
                if depth == 0:
                    cand = s[j:i + 1]
                    break
            j -= 1
    return cand


def zebra_cells(pred, gold):
    """(matching cells, total cells) between the predicted JSON and the gold {"House k": {attr: value}} dict."""
    total = sum(len(v) for v in gold.values())
    if not pred:
        return 0, total
    try:
        p = json.loads(pred)
    except Exception:
        try:
            import ast
            p = ast.literal_eval(pred)
        except Exception:
            return 0, total
    if not isinstance(p, dict):
        return 0, total
    p = p.get("solution", p) if isinstance(p.get("solution", None), dict) else p
    pn = {re.sub(r"\s+", " ", str(k)).strip().lower(): v for k, v in p.items()}
    ok = 0
    for house, attrs in gold.items():
        hv = pn.get(house.lower()) or pn.get(house.lower().replace("house ", "")) or {}
        if not isinstance(hv, dict):
            continue
        hvn = {str(k).strip().lower(): str(v).strip().lower() for k, v in hv.items()}
        for a, val in attrs.items():
            if hvn.get(a.lower()) == str(val).strip().lower():
                ok += 1
    return ok, total


def tot_json(gen):
    """Last JSON-looking object of the response (Test of Time answers are JSON objects)."""
    g = gen.split("</think>")[-1]
    for m in reversed(list(re.finditer(r"\{", g))):
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
    s = str(v).strip().strip("'\"").lower()
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(round(f, 4))
    except Exception:
        return re.sub(r"\s+", " ", s)


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
            out.append(re.sub(r"\s+", " ", x))
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
        m = re.findall(r"Final Answer:\s*(.+)", gen.split("</think>")[-1])
        return m[-1].strip() if m else None
    if args.task == "ifb":  # the response after the reasoning block; an unclosed (truncated) block stays as is
        g = (gen.split("</think>")[-1] if "</think>" in gen else gen).strip()
        return g or None
    if args.task == "zebra":
        return last_json_block(gen)
    if args.task == "lbm":
        return lbm_extract(gen, r.get("task") if r else None)
    if args.task == "lbd":
        return lbd_extract(gen, r.get("task") if r else None)
    if args.task == "lbl":
        return lbl_extract(gen, r.get("task") if r else None)
    if args.task == "lbr":  # official format is <solution>...</solution>; also accept \boxed{} and the last **bold** span
        sol = re.findall(r"<solution>(.*?)</solution>", gen, re.S)
        if sol:
            return sol[-1].strip()
        b = last_boxed(gen)
        if b is not None and "\\boxed" in gen:
            return b
        bold = re.findall(r"\*\*(.+?)\*\*", gen, re.S)
        return bold[-1].strip() if bold else None
    return last_boxed(gen)


def lbm_extract(gen, subtask):
    """LiveBench math: AMPS_Hard answers live in \\boxed{}; math_comp answers are a letter the prompt asks to repeat five times."""
    if subtask == "math_comp":
        m = re.findall(r"\b([A-E])\1{4}\b", gen)
        if m:
            return m[-1]
        b = last_boxed(gen)
        if b:
            mm = re.findall(r"[A-E]", b.upper())
            return mm[-1] if mm else b
        return None
    return last_boxed(gen) if "\\boxed" in gen else None


def lbd_extract(gen, task):
    """LiveBench data analysis: the answer is the response itself (table / class name / dict); drop reasoning before the last block."""
    g = gen.strip()
    g = re.sub(r"<think>.*?</think>", "", g, flags=re.S).strip()
    if task == "cta":
        lines = [l.strip().strip("*`'\"") for l in g.splitlines() if l.strip()]
        return lines[-1] if lines else None
    blocks = re.findall(r"```[a-zA-Z]*\s*\n(.*?)```", g, re.S)
    if blocks:
        return blocks[-1].strip()
    if task == "tablejoin":
        i, j = g.rfind("{"), g.rfind("}")
        return g[i:j + 1] if 0 <= i < j else g
    # tablereformat without a code block: the table is the trailing run of table-looking lines after the reasoning
    html = re.findall(r"<table.*?</table>", g, re.S | re.I)
    if html:
        return html[-1]
    lines = g.splitlines()
    tail = []
    for line in reversed(lines):
        s = line.strip()
        if not s:
            if tail:
                break
            continue
        if s.startswith(("{", "[", "|", "<")) or ("," in s or "\t" in s) and not s.endswith((":", ".")) and len(s.split()) <= 60:
            tail.append(s)
        else:
            if tail:
                break
    if tail:
        return "\n".join(reversed(tail))
    return g


def _records(text, fmt):
    """Parse a table given as JSON / JSONL / CSV / TSV / markdown / HTML into a sorted list of {col: str} records, or None.
    Standard library only (the serving venv has no pandas)."""
    import csv
    import html as _html
    import io

    def cell(v):
        s = str(v).strip() if v is not None else ""
        s = re.sub(r"\.0$", "", s)
        return s

    def table_to_records(header, body):
        header = [str(h).strip().lower() for h in header]
        out = []
        for row in body:
            if not any(str(c).strip() for c in row):
                continue
            out.append({header[k] if k < len(header) else str(k): cell(row[k]) for k in range(len(row))})
        return out

    try:
        t = text.strip()
        if fmt == "json":
            try:
                obj = json.loads(t)
            except Exception:
                lines = [l.strip().rstrip(",") for l in t.splitlines() if l.strip().startswith("{")]
                if len(lines) >= 2:  # JSON lines (LiveBench's ground truth for JSONL targets)
                    obj = [json.loads(l) for l in lines]
                else:
                    i, j = t.find("{"), t.rfind("}")
                    if i < 0:
                        i, j = t.find("["), t.rfind("]")
                    obj = json.loads(t[i:j + 1])
            if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
                recs = [{str(k).strip().lower(): cell(v) for k, v in row.items()} for row in obj.values()]
            elif isinstance(obj, dict) and obj and all(isinstance(v, (list, dict)) for v in obj.values()):  # column-oriented
                cols = list(obj.keys())
                n = max(len(v) for v in obj.values())
                recs = [{str(c).strip().lower(): cell((obj[c][k] if isinstance(obj[c], list) else list(obj[c].values())[k]) if k < len(obj[c]) else "")
                         for c in cols} for k in range(n)]
            elif isinstance(obj, list):
                recs = [{str(k).strip().lower(): cell(v) for k, v in row.items()} for row in obj if isinstance(row, dict)]
            else:
                return None
        elif fmt in ("csv", "tsv"):
            rows = list(csv.reader(io.StringIO(t), delimiter="," if fmt == "csv" else "\t"))
            rows = [r for r in rows if any(c.strip() for c in r)]
            if len(rows) < 2:
                return None
            recs = table_to_records(rows[0], rows[1:])
        elif fmt == "html":
            m = re.search(r"<table.*?</table>", t, re.S | re.I)
            if not m:
                return None
            trs = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), re.S | re.I)
            rows = [[_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S | re.I)] for tr in trs]
            rows = [r for r in rows if r]
            if len(rows) < 2:
                return None
            recs = table_to_records(rows[0], rows[1:])
        elif fmt == "markdown":
            lines = [l for l in t.splitlines() if l.strip().startswith("|")]
            if len(lines) < 2:
                return None
            cols = [c.strip() for c in lines[0].strip().strip("|").split("|")]
            body = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[1:] if not re.fullmatch(r"\|?[\s:\-|]+\|?", l.strip())]
            recs = table_to_records(cols, body)
        else:
            return None
        # pandas-style index columns ("Unnamed: 0") are dropped on both sides
        recs = [{k: v for k, v in r.items() if not k.startswith("unnamed:")} for r in recs]
        return sorted(recs, key=lambda d: json.dumps(d, sort_keys=True))
    except Exception:
        return None


def lbd_correct(pred, gold, task, prompt):
    if pred is None:
        return False
    if task == "cta":
        return pred.strip().lower() == str(gold).strip().lower()
    if task == "tablejoin":
        try:
            import ast
            p = ast.literal_eval(pred) if pred.strip().startswith("{") else json.loads(pred)
            g = ast.literal_eval(gold) if str(gold).strip().startswith("{") else json.loads(gold)
            return {str(k).strip().lower(): str(v).strip().lower() for k, v in p.items()} == {str(k).strip().lower(): str(v).strip().lower() for k, v in g.items()}
        except Exception:
            return False
    # tablereformat: the target format is named in the prompt ("convert ... from X format to Y format")
    m = re.search(r"to (JSON|CSV|TSV|HTML|Markdown|JSONL) format", prompt, re.I)
    fmt = (m.group(1).lower() if m else "json")
    if fmt == "jsonl":
        fmt = "json"
    gm = re.search(r"from (JSON|CSV|TSV|HTML|Markdown|JSONL) format", prompt, re.I)
    gold_fmt = fmt  # LiveBench's ground truth is given in the target format
    pr, gr = _records(pred, fmt), _records(str(gold), gold_fmt)
    if gr is None:  # try the other formats for the gold
        for f2 in ("json", "csv", "tsv", "markdown", "html"):
            gr = _records(str(gold), f2)
            if gr is not None:
                break
    return pr is not None and gr is not None and pr == gr


def lbl_extract(gen, task):
    """LiveBench language: connections and typos / plot answers live in <solution>...</solution>; else the last paragraph."""
    g = re.sub(r"<think>.*?</think>", "", gen, flags=re.S)
    sol = re.findall(r"<solution>(.*?)</solution>", g, re.S)
    if sol:
        return sol[-1].strip()
    if "<PLOT_SUMMARY>" in g:  # plot_unscrambling: the prompt asks to begin the answer with this tag
        return g.rsplit("<PLOT_SUMMARY>", 1)[1].replace("</PLOT_SUMMARY>", "").strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", g.strip()) if p.strip()]
    return paras[-1] if paras else None


def lbl_correct(pred, gold, task):
    if pred is None:
        return False
    if task == "connections":
        def groups(s):
            items = [x.strip().lower() for x in re.split(r"[,\n]", s) if x.strip()]
            return {frozenset(items[k:k + 4]) for k in range(0, len(items), 4)} if len(items) == 8 else None
        return groups(pred) is not None and groups(pred) == groups(str(gold))
    n = lambda s: re.sub(r"\s+", " ", str(s)).strip().strip("\"'").lower()  # noqa: E731
    return n(pred) == n(gold)


def lbr_norm(s):
    s = str(s).lower().strip().strip(".").replace("**", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return s.strip()


def crux_equal(pred, gold):
    import ast
    try:
        g = ast.literal_eval(gold)
    except Exception:
        return pred.strip() == gold.strip()
    # the model sometimes writes \boxed{1: None} for a dict or \boxed{1, 2} for a tuple:
    # the outer braces are eaten by \boxed{}, so retry with them restored
    for cand in (pred, "{" + pred + "}", "(" + pred + ")"):
        try:
            if ast.literal_eval(cand) == g:
                return True
        except Exception:
            continue
    return pred.strip() == gold.strip()


def correct(pred, gold, r=None):
    if args.task in ("lcb", "bcb", "codepro", "ifb"):
        return False  # graded separately in a sandbox (lcb_grade.py / bcb_grade.py / codepro_grade.py)
    if args.task == "lbd":
        return lbd_correct(pred, gold, r.get("task"), r.get("question", ""))
    if args.task == "lbl":
        return lbl_correct(pred, gold, r.get("task"))
    if args.task == "lbm":
        if pred is None:
            return False
        if r.get("task") == "math_comp":
            return pred.strip().upper() == str(gold).strip().upper()
        p, g = norm_math(pred), norm_math(str(gold))
        if p == g:
            return True
        pn, gn = norm_num(p), norm_num(g)
        return pn is not None and gn is not None and re.fullmatch(r"-?[\d.,]+", p or "") is not None and abs(pn - gn) < 1e-6
    if args.task == "zebra":
        ok, total = zebra_cells(pred, gold)
        return total > 0 and ok == total
    if args.task == "lbr":
        return pred is not None and lbr_norm(pred) == lbr_norm(gold)
    if args.task == "mc":
        if pred is None:
            return False
        m = re.findall(r"\b([A-J])\b", pred.strip().upper().replace("\\TEXT{", "").replace("}", " "))
        return bool(m) and m[-1] == str(gold).strip().upper()
    if args.task == "crux":
        return pred is not None and crux_equal(pred, gold)
    if args.task == "lcbt":  # expected outputs are JSON literals (true/false/null) or Python literals
        if pred is None:
            return False
        g = str(gold)
        for a, b in (("true", "True"), ("false", "False"), ("null", "None")):
            g = re.sub(rf"\b{a}\b", b, g)
        return crux_equal(pred, g) or crux_equal(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", pred)), g)
    if args.task == "tot":
        return tot_correct(pred, gold)
    if args.task == "tbench":
        return tbench_correct(pred, gold)
    if args.task == "gsm8k":
        p, g = norm_num(pred), norm_num(gold)
        return p is not None and g is not None and abs(p - g) < 1e-6
    p, g = norm_math(pred), norm_math(gold)
    if p is None:
        return False
    if p == g:
        return True
    pn, gn = norm_num(p), norm_num(g)
    return pn is not None and gn is not None and re.fullmatch(r"-?[\d.,]+", p) is not None and abs(pn - gn) < 1e-6


rows = [json.loads(l) for l in open(args.data) if l.strip()][args.seed_offset:args.seed_offset + args.n]
tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
kw = {"total_ut_steps": args.steps}
if args.threshold is not None:
    kw["early_exit_threshold"] = args.threshold
if args.threshold_steps:
    qs = [float(x) for x in args.threshold_steps.split(",")]
    kw["early_exit_threshold"] = (qs + [1.0] * args.steps)[: args.steps]
    args.threshold = kw["early_exit_threshold"]
kw.update(CFG)
config = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
for _k, _v in kw.items():
    setattr(config, _k, _v)
model = AutoModelForCausalLM.from_pretrained(args.ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
print(f"loaded {args.ckpt} steps={model.config.total_ut_steps} thr={getattr(model.config, 'early_exit_threshold', None)} cfg={CFG} rows={len(rows)}", flush=True)

SFX = ("\n\n" + args.prompt_suffix.strip()) if args.prompt_suffix.strip() else ""
APFX = args.assistant_prefix.encode().decode("unicode_escape") if args.assistant_prefix else ""
texts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r) + SFX}], tokenize=False, add_generation_prompt=True) + APFX for r in rows]
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
gens = [None] * len(texts)
ntoks = [0] * len(texts)
t0 = time.time()
total_tok = 0
with torch.inference_mode():
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, i in enumerate(idx):
            g = gen[j]
            n = int((g != tok.pad_token_id).sum().item())
            ntoks[i] = n
            total_tok += n
            gens[i] = APFX + tok.decode(g, skip_special_tokens=True)
        done = b + len(idx)
        print(f"  {done}/{len(order)} rows, {time.time()-t0:.0f}s, {total_tok/(time.time()-t0):.1f} tok/s", flush=True)
dt = time.time() - t0

n_ok = n_trunc = n_parse = 0
recs = []
for i, r in enumerate(rows):
    pred = extract(gens[i], r)
    gold = {"crux": r.get("output"), "lcbt": r.get("output"), "tot": r.get("label"), "zebra": r.get("solution")}.get(args.task, r.get("answer"))
    ok = correct(pred, gold, r)
    trunc = ntoks[i] >= args.max_new
    n_ok += ok
    n_trunc += trunc
    n_parse += pred is not None
    rec = {"i": args.seed_offset + i, "gold": gold, "pred": pred, "correct": ok, "truncated": trunc, "ntok": ntoks[i], "gen": gens[i]}
    if args.task == "lcb":
        rec.update({"question_id": r.get("question_id"), "difficulty": r.get("difficulty")})
    if args.task == "bcb":
        rec.update({"task_id": r.get("task_id")})
    if args.task == "ifb":
        rec.update({"key": r.get("key"), "instruction_id_list": r.get("instruction_id_list")})
    if args.task == "codepro":
        rec.update({"task_id": r.get("task_id"), "source": r.get("source")})
    if args.task == "zebra":
        c_ok, c_tot = zebra_cells(pred, gold)
        rec.update({"id": r.get("id"), "size": r.get("size"), "cell_acc": c_ok / max(c_tot, 1)})
    if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and r.get("task")):
        rec.update({"question_id": r.get("question_id"), "subtask": r.get("task")})
    recs.append(rec)
acc = n_ok / len(rows)
loops_dec = getattr(model, "_exit_loops_decode_sum", None)
mean_loops = (loops_dec / model._exit_loops_decode_n) if loops_dec is not None and getattr(model, "_exit_loops_decode_n", 0) else float(args.steps)
summary = {"ckpt": args.ckpt, "task": args.task, "steps": args.steps, "threshold": args.threshold, "cfg": CFG, "n": len(rows), "acc": acc,
           "mean_loops": mean_loops,
           "parse_rate": n_parse / len(rows), "trunc_rate": n_trunc / len(rows), "mean_tokens": total_tok / len(rows),
           "wall_s": dt, "tok_s": total_tok / dt, "max_new": args.max_new}
if args.task == "zebra":
    summary["cell_acc"] = sum(rec["cell_acc"] for rec in recs) / len(recs)
    by_size = {}
    for rec in recs:
        d = by_size.setdefault(rec["size"], [0, 0])
        d[0] += rec["correct"]
        d[1] += 1
    summary["by_size"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_size.items())}
if args.task in ("lbr", "lbm", "lbd", "lbl", "lcbt", "tot", "tbench") or (args.task == "crux" and recs and recs[0].get("subtask")):
    by_t = {}
    for rec in recs:
        d = by_t.setdefault(rec["subtask"], [0, 0])
        d[0] += rec["correct"]
        d[1] += 1
    summary["by_subtask"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_t.items())}
print("RESULT " + json.dumps(summary), flush=True)
if args.out:
    with open(args.out, "w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
