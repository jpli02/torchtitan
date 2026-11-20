"""Does the parallel routed forward (core.routed_forward) reproduce the serving model's decode-time early exit?

For a few teacher sequences and a few routers:
  real      the serving model decodes teacher-forced, one token per call with its cache (prefill at full depth, then every
            answer token under decode_early_exit_threshold): per token, the loops it executed and the token its head predicts;
  parallel  one routed_forward call on the whole sequence with the same thresholds: per token, exit loop and prediction
            (read at core.stock_readout_loop, the serving head's selection rule).
Reported: share of tokens with the same exit loop, share with the same predicted token, and for the tokens whose exit differs how
close the gate's cumulative probability was to the threshold (differences should be bfloat16 noise at the threshold).
Also: full depth, parallel forward against the model's own one-call forward (no cache), as a check of the layer code.

  CUDA_VISIBLE_DEVICES=0 python randrouter/check_equiv.py [--n 3] [--max_gen 300] [--routers "0.3,0.3,0.3;0.5,0.5,0.5;1.0,0.38,0.3"]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import SERVE_DIR, load_fill_maps, load_model, load_teacher, routed_forward, stock_readout_loop  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--max_gen", type=int, default=300)
ap.add_argument("--routers", default="0.3,0.3,0.3;0.5,0.5,0.5;1.0,0.38,0.3;0.2,0.6,0.4")
ap.add_argument("--mem_gb", type=float, default=0.0, help="cap this process's GPU memory (shared card)")
ap.add_argument("--ckpt", default=SERVE_DIR, help="serving directory (the exit-lab one for --fill_maps)")
ap.add_argument("--fill_maps", default=None, help="check the calibrated fill: decode_fill_maps in the serving model against fill_maps in the parallel forward")
args = ap.parse_args()

if args.mem_gb:
    torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
model, tok = load_model(args.ckpt)
FILL = load_fill_maps(args.fill_maps) if args.fill_maps else None
inner = model.model
seqs = load_teacher(args.teacher)[:args.n]
print(f"{len(seqs)} sequences; lengths {[len(s['ids']) for s in seqs]}, prompts {[s['n_prompt'] for s in seqs]}", flush=True)


@torch.inference_mode()
def readouts(out, ids):
    """Prediction of every position, read like the serving head reads it."""
    read = stock_readout_loop(out["gate_logit"], out["exit"])[0]
    hid = torch.stack(out["hidden"], dim=0)[:, 0]                       # (4, T, D)
    sel = hid[read, torch.arange(ids.shape[1], device=ids.device)]      # (T, D)
    return model.lm_head(sel).float().argmax(-1), read


with torch.inference_mode():
    # 1. full depth: parallel forward against the model's own forward on the whole sequence
    for s in seqs:
        ids = torch.tensor([s["ids"][:s["n_prompt"] + args.max_gen]], device="cuda")
        ref = model(input_ids=ids, use_cache=False).logits[0].float()
        out = routed_forward(model, ids, torch.tensor([s["n_prompt"]], device="cuda"))
        pred, read = readouts(out, ids)
        last = model.lm_head(out["hidden"][-1][0]).float()
        same = (pred == ref.argmax(-1)).float().mean().item()
        print(f"full depth {s['question_id']}: same prediction {100 * same:.2f} % of {ids.shape[1]} positions; "
              f"max |logit diff| at the last loop {float((last - ref).abs().max()):.3f} (positions the stock rule reads earlier: "
              f"{int((read < 3).sum())})", flush=True)

    # 2. decode-time early exit, token by token, against the parallel forward
    for spec in args.routers.split(";"):
        thr = [float(v) for v in spec.split(",")]
        n_tok = same_exit = same_pred = 0
        loops_real = loops_par = 0
        margins = []
        t0 = time.time()
        for s in seqs:
            ids = s["ids"][:s["n_prompt"] + args.max_gen]
            p = s["n_prompt"]
            inner.decode_early_exit_threshold = thr + [1.0]
            inner.decode_fill_maps = args.fill_maps
            o = model(input_ids=torch.tensor([ids[:p]], device="cuda"), use_cache=True)
            past = o.past_key_values
            r_pred, r_loops = [], []
            for i in range(p, len(ids)):
                s0 = getattr(inner, "_dee_exec_sum", 0)
                o = model(input_ids=torch.tensor([[ids[i]]], device="cuda"), past_key_values=past, use_cache=True)
                past = o.past_key_values
                r_pred.append(int(o.logits[0, -1].argmax()))
                r_loops.append(int(getattr(inner, "_dee_exec_sum", 0) - s0))
            inner.decode_early_exit_threshold = None
            del past, o
            t_ids = torch.tensor([ids], device="cuda")
            out = routed_forward(model, t_ids, torch.tensor([p], device="cuda"), thresholds=thr, fill_maps=FILL)
            pred, _ = readouts(out, t_ids)
            p_loops = (out["exit"][0, p:] + 1).tolist()
            p_pred = pred[p:].tolist()
            lam = out["lam"][0, p:]                                      # (n, 4) float32
            cum = torch.zeros(lam.shape[0], device=lam.device)
            rem = torch.ones_like(cum)
            cdf = []
            for t in range(3):
                cum = cum + lam[:, t] * rem
                rem = rem * (1 - lam[:, t])
                cdf.append(cum.clone())
            for j in range(len(r_loops)):
                n_tok += 1
                same_exit += r_loops[j] == p_loops[j]
                same_pred += r_pred[j] == p_pred[j]
                if r_loops[j] != p_loops[j]:
                    t = min(r_loops[j], p_loops[j]) - 1                  # the loop at which the two disagreed about leaving
                    margins.append(abs(float(cdf[t][j]) - thr[t]))
            loops_real += sum(r_loops)
            loops_par += sum(p_loops)
            torch.cuda.empty_cache()
        m = sorted(margins)
        print(f"router {spec}: {n_tok} tokens; same exit loop {100 * same_exit / n_tok:.2f} %, same prediction {100 * same_pred / n_tok:.2f} %; "
              f"loops per token real {loops_real / n_tok:.3f} parallel {loops_par / n_tok:.3f}; exit differences {len(m)}"
              + (f", distance of the gate's cumulative probability from the threshold there: median {m[len(m) // 2]:.4f}, max {m[-1]:.4f}" if m else "")
              + f"; {time.time() - t0:.0f}s", flush=True)
