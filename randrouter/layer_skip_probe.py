"""Static sub-networks of the looped model: what does it cost to leave out (loop, layer) applications for EVERY token?

Why (2026-11-21). Per-token early exit on the released model is stuck near 2 loops: tokens that leave after loop 1 damage the
cache of the later tokens and, beyond about half of the tokens, make free-running answers drift (README findings 1, 5, 7). The
released gate at 0.3 already sends 87 % of the tokens out after loop 2, so its 2.14 loops are almost entirely loops 1 and 2 of
every token. The only way below 2 loops that does not use loop-1 exits is to make loops 1-2 themselves cheaper: a static mask
over the 4 x 48 (loop, layer) applications. A static mask is the same for every token, so no cache slot is ever missing or
stale, the gate is not involved, and the saving is real compute that also survives batching.

This probe measures whether that space has room, teacher-forced on the held-out answers (base model's own full-depth answers):
  1. full depth with ONE (loop, layer) application skipped: agreement with the full-depth tokens and cross-entropy (192 passes);
  2. cumulative masks: the k cheapest single skips together (greedy by cross-entropy, no interaction search), at full depth;
  3. the same masks under the released gate at 0.3 (exits from loop 2 on, decoder's copy fill): compute in loop-equivalents
     (executed layer applications / 48 per answer token), the two kinds of token error and the value in the real objective's
     units (proxy_value.py). Routers and masks here keep loop-1 exits at zero, the regime in which the calibration was checked.

  CUDA_VISIBLE_DEVICES=9 python randrouter/layer_skip_probe.py --mem_gb 8.5
Writes /home/jli199/boptim_scratch/randrouter/layer_skip/{single.json, greedy.json}; finished stages are reused.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
import proxy_value  # noqa: E402

N_LAYERS = 48


@torch.no_grad()
def run(model, seqs, batches, pad_id, skip=None, thresholds=None):
    """Returns agreement % and cross-entropy of the exit readouts with the decoder's cache, the same with an exact cache
    (equal to the first when nothing exits early), executed loops and loop-equivalents per answer token."""
    skip = set(skip or ())
    per_loop = [sum((t, l) not in skip for l in range(N_LAYERS)) / N_LAYERS for t in range(core.N_LOOPS)]
    cum = [sum(per_loop[:t + 1]) for t in range(core.N_LOOPS)]
    n = ok = ce = okx = cex = loops = equiv = 0.0
    for batch in batches:
        ids, n_prompt, lengths = core.collate(seqs, batch, pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = core.routed_forward(model, ids, n_prompt, thresholds=thresholds, lengths=lengths, skip=skip)
            _, st = core.readout_stats(model, out, ids, n_prompt, lengths, loss_on="exit", need_grad=False)
            if thresholds is not None:
                out_x = core.routed_forward(model, ids, n_prompt, plan=out["exit"], exact_cache=True, skip=skip)
                _, st_x = core.readout_stats(model, out_x, ids, n_prompt, lengths, loss_on="exit", need_grad=False)
            else:
                st_x = st
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        read = (pos >= (n_prompt[:, None] - 1)) & (pos < (lengths[:, None] - 1))
        ex = out["exit"][read]
        n += st["n_read"]
        ok += st["exit_ok"]
        ce += st["exit_ce"]
        okx += st_x["exit_ok"]
        cex += st_x["exit_ce"]
        loops += float((ex + 1).sum())
        equiv += float(torch.tensor(cum, device=ex.device)[ex].sum())
    return {"agree": 100 * ok / n, "ce": ce / n, "agree_exact": 100 * okx / n, "ce_exact": cex / n, "loops": loops / n, "equiv": equiv / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_gb", type=float, default=8.5)
    ap.add_argument("--n_items", type=int, default=99)
    ap.add_argument("--budget_tok", type=int, default=2500)
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/layer_skip")
    ap.add_argument("--ks", default="4,8,12,16,24,32,48,64,80,96")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(core.SERVE_DIR)
    global N_LAYERS
    N_LAYERS = len(model.model.layers)
    print(f"{N_LAYERS} layers per loop, {core.N_LOOPS} loops", flush=True)
    seqs = core.load_teacher("/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split="val")[:a.n_items]
    batches = core.make_batches(seqs, a.budget_tok)
    pad = tok.pad_token_id
    calib = proxy_value.load_calib()
    t0 = time.time()

    # ---- 1. single skips at full depth
    p1 = f"{a.out}/single.json"
    if os.path.exists(p1):
        single = json.load(open(p1))
    else:
        base = run(model, seqs, batches, pad)
        print(f"full depth, nothing skipped: agreement {base['agree']:.2f} %, cross-entropy {base['ce']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        rows = []
        for t in range(core.N_LOOPS):
            for l in range(N_LAYERS):
                r = run(model, seqs, batches, pad, skip={(t, l)})
                rows.append({"loop": t, "layer": l, "agree": r["agree"], "ce": r["ce"]})
            sub = [x for x in rows if x["loop"] == t]
            print(f"  loop {t + 1}: agreement lost per skipped layer: median {sorted(base['agree'] - x['agree'] for x in sub)[len(sub) // 2]:.2f} points, "
                  f"worst {max(base['agree'] - x['agree'] for x in sub):.2f}, layers losing under 0.05 points: {sum(base['agree'] - x['agree'] < 0.05 for x in sub)} of {N_LAYERS}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        single = {"base": base, "rows": rows, "n_items": len(seqs)}
        json.dump(single, open(p1, "w"), indent=1)
    base = single["base"]
    order = sorted(single["rows"], key=lambda x: x["ce"])          # cheapest first
    print("cheapest 12 single skips (loop, layer, agreement lost, cross-entropy added): " +
          ", ".join(f"({x['loop'] + 1},{x['layer']}: {base['agree'] - x['agree']:+.2f}, {x['ce'] - base['ce']:+.4f})" for x in order[:12]), flush=True)

    # ---- 2. + 3. cumulative masks, at full depth and under the released gate at 0.3
    gate03 = run(model, seqs, batches, pad, thresholds=[0.3, 0.3, 0.3])
    v0, p0 = proxy_value.value(gate03["equiv"], 100 - gate03["agree_exact"], gate03["agree_exact"] - gate03["agree"], calib)
    print(f"released gate at 0.3, nothing skipped: {gate03['equiv']:.2f} loop-equivalents, agreement {gate03['agree']:.2f} %, predicted pass@1 {p0:.1f}, value {v0:.2f}", flush=True)
    res = []
    for k in [int(v) for v in a.ks.split(",")]:
        mask = {(x["loop"], x["layer"]) for x in order[:k]}
        full = run(model, seqs, batches, pad, skip=mask)
        g = run(model, seqs, batches, pad, skip=mask, thresholds=[0.3, 0.3, 0.3])
        own, cache = 100 - g["agree_exact"], g["agree_exact"] - g["agree"]
        val, p = proxy_value.value(g["equiv"], own, cache, calib)
        vf, pf = proxy_value.value(full["equiv"], 100 - full["agree"], 0.0, calib)
        by_loop = [sum(1 for (t, _) in mask if t == i) for i in range(core.N_LOOPS)]
        res.append({"k": k, "mask": sorted(mask), "skipped_by_loop": by_loop, "full": full, "full_pass1": pf, "full_value": vf,
                    "gate03": g, "gate03_own": own, "gate03_cache": cache, "gate03_pass1": p, "gate03_value": val})
        print(f"  {k:>3} skipped (per loop {by_loop}): full depth {full['equiv']:.2f} loop-eq, agreement {full['agree']:.2f} %, pass@1~{pf:.1f}, value {vf:.2f} | "
              f"gate 0.3: {g['equiv']:.2f} loop-eq, agreement {g['agree']:.2f} % (own {own:.2f}, cache {cache:.2f}), pass@1~{p:.1f}, value {val:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"gate03_base": gate03, "gate03_base_value": v0, "gate03_base_pass1": p0, "masks": res}, open(f"{a.out}/greedy.json", "w"), indent=1)
    best = min(res, key=lambda r: r["gate03_value"])
    print(f"best greedy mask under the gate: {best['k']} skipped, value {best['gate03_value']:.2f} against {v0:.2f} without a mask "
          f"({best['gate03']['equiv']:.2f} against {gate03['equiv']:.2f} loop-equivalents, predicted pass@1 {best['gate03_pass1']:.1f} against {p0:.1f})")
    print("LAYER SKIP PROBE done", flush=True)


if __name__ == "__main__":
    main()
