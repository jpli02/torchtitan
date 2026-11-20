"""Teacher-forced fidelity of routers: how often does the model, decoding its own full-depth answer under a router with the real
cache fill, predict the full-depth token? And how much of the loss is the cache (same exits, exact cache)?

For every router: loops per answer token, top-1 agreement of the exit readout with the teacher token and its cross-entropy,
  * "real"   the decoder's cache (skipped loops hold a copy of the exit loop's keys and values), exits decided on that cache;
  * "exact"  the SAME exits with every position's exact keys and values ("gather" mode): the wrong tokens here are tokens that
             should not have left (selection); the extra wrong tokens of "real" are the cache's.
One parallel forward per (batch, router, mode); a sequence takes about a second instead of a minute of decoding.

  CUDA_VISIBLE_DEVICES=0 python randrouter/tf_eval.py [--ckpt DIR] [--lora FILE] [--gate FILE] [--split val] [--n 99]
        [--routers "full;0.7;0.5;0.3;0.2;1.0,0.38,0.3;iid:0.25,0.25,0.25,0.25"] [--out FILE.json]
A router is "full", one threshold q (the released single knob), "q1,q2,q3", "iid:p1,p2,p3,p4" (the random router with those
exit shares, seed 0), or "rank:f1,f2,f3" (after loop s the share f_s of the still-running tokens that the gate is surest about
leaves; the gate-ordered counterpart of the random router).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

DEFAULT_ROUTERS = "full;0.7;0.5;0.3;0.2;1.0,0.38,0.3;iid:0.25,0.25,0.25,0.25;iid:0.1,0.4,0.4,0.1"


def parse_router(spec):
    if spec == "full":
        return {"spec": spec, "thr": None, "rates": None, "shares": None}
    if spec.startswith("iid:"):
        return {"spec": spec, "thr": None, "rates": [float(v) for v in spec[4:].split(",")], "shares": None}
    if spec.startswith("rank:"):
        return {"spec": spec, "thr": None, "rates": None, "shares": [float(v) for v in spec[5:].split(",")]}
    v = [float(x) for x in spec.split(",")]
    return {"spec": spec, "thr": v * 3 if len(v) == 1 else v, "rates": None, "shares": None}


@torch.inference_mode()
def evaluate(model, seqs, routers, pad_id, budget_tok=6000, gate=None, log=print, fill_maps=None):
    """Returns {router spec: {"loops", "agree_real", "ce_real", "agree_exact", "ce_exact", "n", per-loop shares}}."""
    batches = core.make_batches(seqs, budget_tok)
    res = {}
    for r in routers:
        acc = {"n": 0, "loops": 0.0, "ok_real": 0.0, "ce_real": 0.0, "ok_exact": 0.0, "ce_exact": 0.0, "share": [0, 0, 0, 0], "ok_by_exit": [0.0] * 4}
        gen = torch.Generator().manual_seed(0)
        t0 = time.time()
        for batch in batches:
            ids, n_prompt, lengths = core.collate(seqs, batch, pad_id)
            plan = None
            if r["rates"] is not None:
                plan, _ = core.sample_iid_plan(ids, gen, rates=r["rates"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = core.routed_forward(model, ids, n_prompt, plan=plan, thresholds=r["thr"], shares=r["shares"], lengths=lengths, gate=gate,
                                          fill_maps=fill_maps)
                _, st = core.readout_stats(model, out, ids, n_prompt, lengths, loss_on="exit", need_grad=False)
                if r["spec"] == "full":
                    st_x = st
                else:
                    # the same exits with an exact cache: reuse the decided exits as a plan, no key/value replacement
                    out_x = core.routed_forward(model, ids, n_prompt, plan=out["exit"], exact_cache=True, gate=gate)
                    _, st_x = core.readout_stats(model, out_x, ids, n_prompt, lengths, loss_on="exit", need_grad=False)
            acc["n"] += st["n_read"]
            acc["loops"] += st["loops_sum"]
            acc["ok_real"] += st["exit_ok"]
            acc["ce_real"] += st["exit_ce"]
            acc["ok_exact"] += st_x["exit_ok"]
            acc["ce_exact"] += st_x["exit_ce"]
            pos = torch.arange(ids.shape[1], device=ids.device)[None]
            read = (pos >= (n_prompt[:, None] - 1)) & (pos < (lengths[:, None] - 1))
            ex = out["exit"][read]
            for t in range(4):
                acc["share"][t] += int((ex == t).sum())
        n = max(acc["n"], 1)
        res[r["spec"]] = {"n": acc["n"], "loops": acc["loops"] / n, "agree_real": acc["ok_real"] / n, "ce_real": acc["ce_real"] / n,
                          "agree_exact": acc["ok_exact"] / n, "ce_exact": acc["ce_exact"] / n, "share": [s / n for s in acc["share"]]}
        d = res[r["spec"]]
        log(f"  {r['spec']:<28} loops {d['loops']:.3f}  agree real {100 * d['agree_real']:.2f} %  exact cache {100 * d['agree_exact']:.2f} %  "
            f"ce real {d['ce_real']:.4f} exact {d['ce_exact']:.4f}  exit shares {[round(100 * s) for s in d['share']]}  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=core.SERVE_DIR)
    ap.add_argument("--lora", default=None, help="LoRA state of randrouter/train.py to load on top of --ckpt")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=float, default=64.0)
    ap.add_argument("--gate", default=None, help="gate file of randrouter/stage2.py (replaces the model's exit gate)")
    ap.add_argument("--fill_maps", default=None, help="calibrated cache fill (randrouter/fit_fill_maps.py) instead of the plain copy")
    ap.add_argument("--mem_gb", type=float, default=0.0)
    ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--routers", default=DEFAULT_ROUTERS)
    ap.add_argument("--budget_tok", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mem_gb:
        torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(args.ckpt)
    if args.lora:
        core.add_lora(model, args.lora_r, args.lora_alpha)
        core.load_lora_state(model, torch.load(args.lora, map_location="cpu"))
    if args.gate:
        g = torch.load(args.gate, map_location="cpu")
        with torch.no_grad():
            model.model.early_exit_gate.weight.copy_(g["weight"].to(model.model.early_exit_gate.weight))
            model.model.early_exit_gate.bias.copy_(g["bias"].to(model.model.early_exit_gate.bias))
    seqs = core.load_teacher(args.teacher, split=args.split)[:args.n]
    print(f"{len(seqs)} sequences ({args.split}), {sum(len(s['ids']) - s['n_prompt'] for s in seqs)} answer tokens; ckpt {args.ckpt} lora {args.lora} gate {args.gate}", flush=True)
    fm = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
    res = evaluate(model, seqs, [parse_router(s) for s in args.routers.split(";")], tok.pad_token_id, args.budget_tok, fill_maps=fm)
    if args.out:
        json.dump({"ckpt": args.ckpt, "lora": args.lora, "gate": args.gate, "fill_maps": args.fill_maps, "split": args.split, "n_seqs": len(seqs), "routers": res}, open(args.out, "w"), indent=1)
