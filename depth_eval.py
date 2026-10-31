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
ap.add_argument("--task", choices=["gsm8k", "math", "crux", "lcb", "mc"], required=True)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument("--threshold_steps", default=None, help="per-step CDF thresholds q1,q2,q3 (last step always exits), gather mode")
ap.add_argument("--n", type=int, default=10**9)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--out", default=None)
ap.add_argument("--seed_offset", type=int, default=0, help="skip this many rows first")
ap.add_argument("--cfg", action="append", default=[], help="config override key=value (bool/int/float parsed), e.g. kv_share_decode=true")
args = ap.parse_args()


def _parse_val(v):
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
    if args.task == "crux":
        return ("Here is a Python function and an input. Determine the exact output of calling the function on this input.\n\n"
                f"```python\n{r['code']}\n```\n\nInput: `f({r['input']})`\n\n"
                "Reason step by step, then give the exact output as a Python literal within \\boxed{}.")
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


def extract(gen):
    return last_code_block(gen) if args.task == "lcb" else last_boxed(gen)


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


def correct(pred, gold):
    if args.task == "lcb":
        return False  # graded separately by lcb_grade.py in a sandbox
    if args.task == "mc":
        if pred is None:
            return False
        m = re.findall(r"\b([A-J])\b", pred.strip().upper().replace("\\TEXT{", "").replace("}", " "))
        return bool(m) and m[-1] == str(gold).strip().upper()
    if args.task == "crux":
        return pred is not None and crux_equal(pred, gold)
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

texts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r)}], tokenize=False, add_generation_prompt=True) for r in rows]
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
            gens[i] = tok.decode(g, skip_special_tokens=True)
        done = b + len(idx)
        print(f"  {done}/{len(order)} rows, {time.time()-t0:.0f}s, {total_tok/(time.time()-t0):.1f} tok/s", flush=True)
dt = time.time() - t0

n_ok = n_trunc = n_parse = 0
recs = []
for i, r in enumerate(rows):
    pred = extract(gens[i])
    gold = r.get("answer") if args.task != "crux" else r.get("output")
    ok = correct(pred, gold)
    trunc = ntoks[i] >= args.max_new
    n_ok += ok
    n_trunc += trunc
    n_parse += pred is not None
    rec = {"i": args.seed_offset + i, "gold": gold, "pred": pred, "correct": ok, "truncated": trunc, "ntok": ntoks[i], "gen": gens[i]}
    if args.task == "lcb":
        rec.update({"question_id": r.get("question_id"), "difficulty": r.get("difficulty")})
    recs.append(rec)
acc = n_ok / len(rows)
loops_dec = getattr(model, "_exit_loops_decode_sum", None)
mean_loops = (loops_dec / model._exit_loops_decode_n) if loops_dec is not None and getattr(model, "_exit_loops_decode_n", 0) else float(args.steps)
summary = {"ckpt": args.ckpt, "task": args.task, "steps": args.steps, "threshold": args.threshold, "cfg": CFG, "n": len(rows), "acc": acc,
           "mean_loops": mean_loops,
           "parse_rate": n_parse / len(rows), "trunc_rate": n_trunc / len(rows), "mean_tokens": total_tok / len(rows),
           "wall_s": dt, "tok_s": total_tok / dt, "max_new": args.max_new}
print("RESULT " + json.dumps(summary), flush=True)
if args.out:
    with open(args.out, "w") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
