"""How much does one decode step cost with 1, 2 or 4 rows? (Is batch-1 decoding of the looped model bound by per-call overhead?)

If a step with 4 rows costs about as much as a step with 1 row, the loops a token skips need not be skipped at all: the deeper
loops of the tokens already emitted can run in the same batched pass as the first loop of the next token ("deferred cache
completion"), so the cache stays exact and only latency, not accuracy, depends on the router. This measures the premise.

  CUDA_VISIBLE_DEVICES=0 python randrouter/bench_batch_step.py [--prompt 500] [--steps 40]
Prints seconds per full-depth decode step (4 loops x 48 layers, one new token per row) for batch 1, 2, 4.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", type=int, default=500)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--mem_gb", type=float, default=0.0)
args = ap.parse_args()
if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = core.load_model(core.SERVE_DIR)
seq = core.load_teacher("/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split="val")[0]["ids"]
ids = (seq * (args.prompt // len(seq) + 1))[:args.prompt + args.steps]
res = {}
with torch.inference_mode():
    for b in (1, 2, 4, 1):
        x = torch.tensor([ids[:args.prompt]] * b, device="cuda")
        o = model(input_ids=x, use_cache=True, logits_to_keep=1)
        past = o.past_key_values
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(args.steps):
            o = model(input_ids=torch.tensor([[ids[args.prompt + i]]] * b, device="cuda"), past_key_values=past, use_cache=True)
            past = o.past_key_values
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.steps
        res.setdefault(b, []).append(dt)
        print(f"batch {b}: {1000 * dt:.1f} ms per step ({b / dt:.1f} tokens/s over the rows)", flush=True)
        del past, o
        torch.cuda.empty_cache()
one = min(res[1])
print("relative cost of a step: " + ", ".join(f"batch {b} = {min(v) / one:.2f}x" for b, v in sorted(res.items())))
