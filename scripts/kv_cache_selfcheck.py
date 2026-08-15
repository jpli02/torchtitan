#!/usr/bin/env python3
"""Isolate KV-cache correctness for the Ouro early-exit decode.

Decodes the SAME EvalPlus prompt with the SAME HF weights + SAME early-exit gate
two ways and compares token-for-token:
  (A) KV-cache incremental  -- the path under test (_generate_evalplus_chat_hf)
  (B) full-recompute no-cache -- feed the whole sequence every step, take last pos

If A == B for every token, the cache reproduces the no-cache recompute exactly, so
any pass@1 gap vs the torchtitan reference is a torchtitan-vs-HF implementation
difference (bf16), not a KV-cache bug. If A != B, the divergence point + top-2
logit gap tells us whether it's a cache bug or a genuine near-tie.
"""
import sys, torch
sys.path.insert(0, "scripts")
from evaluate_humaneval_evalplus import (
    _load_hf_generate_model, _ouro_exit_steps, _build_evalplus_prompt,
)
from evalplus.data import get_human_eval_plus

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--fp32", action="store_true")
_ap.add_argument("--threshold", type=float, default=0.5)
_ap.add_argument("--max_new", type=int, default=512)
_ap.add_argument("--tasks", default="HumanEval/90,HumanEval/50,HumanEval/100,HumanEval/20")
_args = _ap.parse_args()

HF = "./assets/hf/Ouro-1.4B"
THR = _args.threshold
MAXNEW = _args.max_new
TASKS = [t.strip() for t in _args.tasks.split(",") if t.strip()]
DTYPE = torch.float32 if _args.fp32 else torch.bfloat16

model, tok, eos_ids = _load_hf_generate_model(HF, DTYPE)
problems = get_human_eval_plus()


def prompt_ids(prompt):
    s = _build_evalplus_prompt(
        lambda m: tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False),
        prompt, None, True, True)  # system=None, prefill=True, no_system=True
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


@torch.no_grad()
def decode_kv(ids):
    """Incremental KV-cache greedy decode with early exit. Returns token list."""
    cur, L = ids, ids.shape[1]
    cache_position = torch.arange(L, device=ids.device)
    cache, out = None, []
    for step in range(MAXNEW):
        base, hidden, gate = model.model(
            input_ids=cur, past_key_values=cache, use_cache=True,
            cache_position=cache_position)
        cache = base.past_key_values
        R = len(gate)
        es = int(_ouro_exit_steps(gate, THR)[0, -1].item()) if THR < 1.0 else R - 1
        nt = int(model.lm_head(hidden[es][:, -1:, :])[0, -1].argmax(-1))
        if nt in eos_ids:
            break
        out.append(nt)
        cur = torch.tensor([[nt]], device=ids.device)
        cache_position = torch.tensor([L + step], device=ids.device)
    return out


@torch.no_grad()
def decode_full(ids):
    """No-cache: recompute the full sequence every step. Returns (tokens, gaps)."""
    seq, out, gaps = ids, [], []
    for step in range(MAXNEW):
        base, hidden, gate = model.model(input_ids=seq, use_cache=False)
        R = len(gate)
        es = int(_ouro_exit_steps(gate, THR)[0, -1].item()) if THR < 1.0 else R - 1
        logits = model.lm_head(hidden[es][:, -1:, :])[0, -1].float()
        top2 = logits.topk(2).values
        nt = int(logits.argmax(-1))
        gaps.append((top2[0] - top2[1]).item())
        if nt in eos_ids:
            break
        out.append(nt)
        seq = torch.cat([seq, torch.tensor([[nt]], device=ids.device)], dim=1)
    return out, gaps


print(f"dtype={DTYPE}  threshold={THR}\n")
allmatch = True
for tid in TASKS:
    ids = prompt_ids(problems[tid]["prompt"])
    a = decode_kv(ids)
    b, gaps = decode_full(ids)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == len(a) == len(b):
        print(f"{tid:16} IDENTICAL  ({len(a)} tokens)")
    else:
        allmatch = False
        gap = gaps[i] if i < len(gaps) else float("nan")
        print(f"{tid:16} DIVERGE at token {i}/{n}  logit top1-top2 gap there = {gap:.4f}")
        print(f"   kv  : {tok.decode(a[i:i+8])!r}")
        print(f"   full: {tok.decode(b[i:i+8])!r}")
print("\nRESULT:", "KV-CACHE EXACT vs no-cache recompute"
      if allmatch else "KV-cache diverges from recompute -- inspect gap above")
