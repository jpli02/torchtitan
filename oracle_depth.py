"""Oracle per-token depth bound and gate-vs-oracle comparison, teacher-forced.

For each generation (from a depth_eval --out file at K=4): run the full prompt+generation
through the model once (no cache), take the per-step hidden states, apply the LM head per
step, and for every generated token find the smallest step whose argmax already equals the
final step's argmax. Its mean is the lowest loop count any per-token router could reach
with the greedy output unchanged. Alongside, apply the released gate's stick-breaking exit
distribution at several thresholds to the same tokens: mean loops, and how often the gate's
chosen step disagrees with the final step (flip rate), split into under-thinking (exits
before the oracle step) and over-thinking (continues past it).

Usage: python oracle_depth.py CKPT GEN.jsonl --task crux|gsm8k|math [--n N] [--max_len 4096]
                              [--thr 0.9,0.7,0.5,0.3,0.2,0.1] [--out OUT.json]
"""
import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("gen")
ap.add_argument("--task", required=True)
ap.add_argument("--n", type=int, default=10**9)
ap.add_argument("--max_len", type=int, default=4096)
ap.add_argument("--thr", default="0.9,0.7,0.5,0.3,0.2,0.1")
ap.add_argument("--out", default=None)
ap.add_argument("--dump", default=None, help="save per-token features (agreement per step, gate lambdas, hidden-state deltas) to this .pt")
args = ap.parse_args()
dump = {"agree": [], "lam": [], "hdiff": [], "gen_id": []} if args.dump else None
THRS = [float(x) for x in args.thr.split(",")]
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_prompt(r, task):
    if task in ("gsm8k", "math"):
        return r["question"] + SUFFIX
    if task == "crux":
        return ("Here is a Python function and an input. Determine the exact output of calling the function on this input.\n\n"
                f"```python\n{r['code']}\n```\n\nInput: `f({r['input']})`\n\n"
                "Reason step by step, then give the exact output as a Python literal within \\boxed{}.")
    raise SystemExit("task not supported here")


gens = [json.loads(l) for l in open(args.gen) if l.strip()]
summary0 = gens[0].get("summary", {})
gens = [g for g in gens if "gen" in g][:args.n]
data_path = {"crux": "/home/jli199/.claude/jobs/c0d2da0a/tmp/cruxeval.jsonl", "gsm8k": "/home/jli199/.claude/jobs/c0d2da0a/tmp/gsm8k_test.jsonl",
             "math": "/home/jli199/.claude/jobs/c0d2da0a/tmp/math500.jsonl"}[args.task]
rows = [json.loads(l) for l in open(data_path) if l.strip()]

tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
config = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
T = int(config.total_ut_steps)
print(f"loaded {args.ckpt} T={T} gens={len(gens)}", flush=True)

tot_tokens = 0
oracle_hist = [0] * T
oracle_sum = 0
gate_sum = {q: 0.0 for q in THRS}
gate_flip = {q: 0 for q in THRS}
gate_under = {q: 0 for q in THRS}
gate_over = {q: 0 for q in THRS}
step_agree = [0] * T  # tokens whose step-s argmax equals the final argmax
t0 = time.time()
with torch.inference_mode():
    for gi, g in enumerate(gens):
        r = rows[g["i"]]
        prompt = tok.apply_chat_template([{"role": "user", "content": build_prompt(r, args.task)}], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        g_ids = tok(g["gen"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = torch.cat([p_ids, g_ids])[: args.max_len]
        gen_start = len(p_ids)
        if len(ids) <= gen_start + 1:
            continue
        out, hs_list, gate_list = model.model(input_ids=ids[None].cuda(), use_cache=False)
        pos = slice(gen_start - 1, len(ids) - 1)  # positions predicting generated tokens
        arg = []
        for s in range(T):
            arg.append(model.lm_head(hs_list[s][0, pos]).argmax(-1))
        final = arg[T - 1]
        n = final.numel()
        agree = torch.stack([a == final for a in arg], dim=0)  # [T, n]
        oracle = agree.float().argmax(0)  # first step that agrees (final always agrees)
        # gate exit step per token
        lam = torch.stack([torch.sigmoid(gl[0, pos, 0].float()) for gl in gate_list], dim=0)  # [T, n]
        remaining = torch.ones(n, device=lam.device)
        cdf = torch.zeros(T, n, device=lam.device)
        for s in range(T - 1):
            cdf[s] = (cdf[s - 1] if s > 0 else 0) + lam[s] * remaining
            remaining = remaining * (1 - lam[s])
        cdf[T - 1] = 1.0
        for q in THRS:
            ex = (cdf >= q).float().argmax(0)  # first step crossing q
            gate_sum[q] += float((ex + 1).sum())
            chosen_agree = agree[ex, torch.arange(n, device=ex.device)]
            gate_flip[q] += int((~chosen_agree).sum())
            gate_under[q] += int((ex < oracle).sum())
            gate_over[q] += int((ex > oracle).sum())
        for s in range(T):
            step_agree[s] += int(agree[s].sum())
            oracle_hist[s] += int((oracle == s).sum())
        oracle_sum += int((oracle + 1).sum())
        tot_tokens += n
        if dump is not None:
            hs = torch.stack([h[0, pos].float() for h in hs_list], dim=0)  # [T, n, d]
            hd = (hs[1:] - hs[:-1]).norm(dim=-1) / hs[1:].norm(dim=-1).clamp_min(1e-6)  # [T-1, n]
            dump["agree"].append(agree.t().cpu())
            dump["lam"].append(lam.t().half().cpu())
            dump["hdiff"].append(hd.t().half().cpu())
            dump["gen_id"].append(torch.full((n,), g["i"], dtype=torch.int32))
        if (gi + 1) % 25 == 0:
            print(f"  {gi+1}/{len(gens)} gens, {tot_tokens} tokens, {time.time()-t0:.0f}s", flush=True)

res = {
    "ckpt": args.ckpt, "gen": args.gen, "task": args.task, "gens": len(gens), "tokens": tot_tokens,
    "oracle_mean_loops": oracle_sum / tot_tokens,
    "oracle_step_fraction": [h / tot_tokens for h in oracle_hist],
    "uniform_step_agreement_with_final": [a / tot_tokens for a in step_agree],
    "gate": {str(q): {"mean_loops": gate_sum[q] / tot_tokens, "flip_rate": gate_flip[q] / tot_tokens,
                      "under_think": gate_under[q] / tot_tokens, "over_think": gate_over[q] / tot_tokens} for q in THRS},
}
print("\nORACLE " + json.dumps(res))
print(f"\ntokens={tot_tokens}  oracle mean loops={res['oracle_mean_loops']:.3f}  "
      f"oracle step fractions={[round(x, 3) for x in res['oracle_step_fraction']]}")
print(f"uniform-depth agreement with final argmax: {[round(x, 3) for x in res['uniform_step_agreement_with_final']]}")
print(f"{'q':>5} {'loops':>6} {'flip':>6} {'under':>6} {'over':>6}")
for q in THRS:
    gq = res["gate"][str(q)]
    print(f"{q:>5} {gq['mean_loops']:6.2f} {gq['flip_rate']:6.3f} {gq['under_think']:6.3f} {gq['over_think']:6.3f}")
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
if dump is not None:
    torch.save({k: torch.cat(v, dim=0) for k, v in dump.items()}, args.dump)
    print(f"dumped per-token features for {tot_tokens} tokens -> {args.dump}")
