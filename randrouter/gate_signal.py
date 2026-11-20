"""How much does the released gate know, loop by loop? (Why a threshold search over it cannot go below about 2 loops.)

Full-depth teacher-forced pass over the teacher answers (exact cache). For every answer token and loops 1 to 3:
  * the gate's exit probability lambda_t and the cumulative exit probability the router thresholds (its spread over tokens);
  * whether the loop-t readout already predicts the full-depth token ("safe to leave");
  * AUC of the cumulative probability (and of lambda_t) as a score for "safe to leave after loop t": 0.5 = no information.
The paper's Stage II gives the gate after loop 1 no target, so that is where the signal is expected to be weakest.

  CUDA_VISIBLE_DEVICES=0 python randrouter/gate_signal.py [--split val] [--n 99] [--gate FILE] [--lora FILE] [--out FILE.json]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def auc(score, label):
    """Area under the ROC curve by ranks (ties get the mean rank)."""
    score, label = score.double().cpu(), label.bool().cpu()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = torch.argsort(score)
    s = score[order]
    ranks = torch.empty_like(s)
    # mean ranks for ties
    uniq, inv, cnt = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(cnt, 0).double()
    mean_rank = ends - (cnt.double() - 1) / 2
    ranks[order] = mean_rank[inv]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=float, default=64.0)
    ap.add_argument("--budget_tok", type=int, default=4000)
    ap.add_argument("--mem_gb", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mem_gb:
        torch.cuda.set_per_process_memory_fraction(args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(core.SERVE_DIR)
    if args.lora:
        core.add_lora(model, args.lora_r, args.lora_alpha)
        core.load_lora_state(model, torch.load(args.lora, map_location="cpu"))
    if args.gate:
        g = torch.load(args.gate, map_location="cpu")
        with torch.no_grad():
            model.model.early_exit_gate.weight.copy_(g["weight"].to(model.model.early_exit_gate.weight))
            model.model.early_exit_gate.bias.copy_(g["bias"].to(model.model.early_exit_gate.bias))
    seqs = core.load_teacher(args.teacher, split=args.split)[:args.n]
    lam_all, ok_all = [], []
    with torch.inference_mode():
        for batch in core.make_batches(seqs, args.budget_tok):
            ids, n_prompt, lengths = core.collate(seqs, batch, tok.pad_token_id)
            out = core.routed_forward(model, ids, n_prompt)
            pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
            read = (pos >= n_prompt[:, None]) & (pos < (lengths[:, None] - 1))     # decoded positions (the prefill's last position never exits)
            bi, ti = read.nonzero(as_tuple=True)
            target = ids[bi, ti + 1]
            lam_all.append(out["lam"][bi, ti].cpu())
            ok_all.append(torch.stack([(model.lm_head(out["hidden"][t][bi, ti]).float().argmax(-1) == target) for t in range(4)], dim=-1).cpu())
    lam, ok = torch.cat(lam_all), torch.cat(ok_all)
    n = lam.shape[0]
    cum = torch.zeros(n)
    rem = torch.ones(n)
    res = {"n_tokens": n, "n_seqs": len(seqs), "gate": args.gate, "lora": args.lora, "loops": []}
    print(f"{len(seqs)} sequences, {n} answer tokens; readout already right after loop 1/2/3/4: "
          f"{[round(100 * float(ok[:, t].float().mean()), 2) for t in range(4)]} %", flush=True)
    for t in range(3):
        cum = cum + lam[:, t] * rem
        rem = rem * (1 - lam[:, t])
        q = torch.quantile(cum, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
        d = {"loop": t + 1, "safe_share": float(ok[:, t].float().mean()), "cdf_quantiles_5_25_50_75_95": [float(v) for v in q],
             "cdf_std": float(cum.std()), "auc_cdf": auc(cum, ok[:, t]), "auc_lambda": auc(lam[:, t], ok[:, t])}
        # what a threshold on the cumulative probability buys at this loop: among the x % most confident tokens, how many are safe
        order = torch.argsort(cum, descending=True)
        d["safe_among_top"] = {str(p): float(ok[order[:max(1, int(n * p / 100))], t].float().mean()) for p in (10, 25, 50, 75, 90)}
        res["loops"].append(d)
        print(f"after loop {t + 1}: cumulative exit probability quantiles (5/25/50/75/95 %) {[round(float(v), 3) for v in q]}, std {d['cdf_std']:.3f}; "
              f"AUC for 'readout already right' {d['auc_cdf']:.3f} (lambda alone {d['auc_lambda']:.3f}); "
              f"right among the most confident 10/25/50/75/90 %: {[round(100 * v, 2) for v in d['safe_among_top'].values()]}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
