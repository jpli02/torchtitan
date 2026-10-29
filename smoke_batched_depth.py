"""Smoke: batched greedy generation with the Ouro HF model at a given loop count.

Checks (1) batch>1 with left padding works, (2) batched output == batch-1 output
for the same prompt (padding correctness), (3) total_ut_steps can be set to 2 and
6 (extrapolation), (4) tokens/s per setting.
Usage: python smoke_batched_depth.py CKPT_DIR
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = sys.argv[1]
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

qs = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def prompts(questions):
    out = []
    for q in questions:
        msgs = [{"role": "user", "content": q + SUFFIX}]
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def run(model, texts, max_new):
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    texts_out = tok.batch_decode(gen, skip_special_tokens=True)
    ntok = int((gen != tok.pad_token_id).sum().item())
    return texts_out, dt, ntok


for steps in (4, 2, 6):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16, total_ut_steps=steps)
    model = model.cuda().eval()
    print(f"\n=== total_ut_steps={model.config.total_ut_steps} (load {time.time()-t0:.0f}s) ===", flush=True)
    outs, dt, ntok = run(model, prompts(qs), 64)
    print(f"batch of 4, 64 new tokens max: {dt:.1f}s, {ntok/dt:.1f} tok/s aggregate", flush=True)
    for i, o in enumerate(outs):
        print(f"  [{i}] {o[:140]!r}")
    single, dt1, n1 = run(model, prompts(qs[:1]), 64)
    print(f"batch-1 same prompt: {dt1:.1f}s, {n1/dt1:.1f} tok/s; MATCHES batched: {single[0] == outs[0]}", flush=True)
    del model
    torch.cuda.empty_cache()
print("peak mem GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))
