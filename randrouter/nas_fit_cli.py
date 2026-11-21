"""Train one router from a random initialisation with the Stage-II loss on cached states (nas_fit.py) and save it for the gate-lab
serving model; report what it does teacher-forced. Run by the gate-search objective (boptim_lcb/ouro_lcb_gate_nas.py) as a
subprocess on the card it acquired, so the optimizer's process holds no GPU memory.

  CUDA_VISIBLE_DEVICES=0 python randrouter/nas_fit_cli.py --spec '{"layers":1,"hidden":128,"act":"gelu","norm":true}' --gamma 0.005 \
      --q 0.3 --out router.pt [--proxy]
Prints one line "FIT {json}": the fit diagnostics, the share of training positions whose cumulative exit probability after loop 1
reaches q (the loop-1 exit share a real decoder would start from), and with --proxy the teacher-forced loops, the two kinds of token
error and the calibrated pass@1 on the 34 held-out answers (randrouter/proxy_value.py).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_fit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats", default="/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--mem_gb", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.time()
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    spec = json.loads(a.spec)
    feats = nas_fit.load_feats(a.feats)
    router, diag = nas_fit.fit_router(feats, spec, a.gamma, k=a.k, steps=a.steps, lr=a.lr, seed=a.seed)
    out = {"spec": spec, "gamma": a.gamma, "q": a.q, "fit": diag}
    with torch.no_grad():   # cumulative exit probability after loop 1 = the gate's probability after loop 1 (stick-breaking)
        h1 = feats["h"][:, 0]
        lam1 = []
        for s in range(0, h1.shape[0], 16384):
            lam1.append(torch.sigmoid(router(h1[s:s + 16384].cuda().float()).squeeze(-1)).cpu())
        lam1 = torch.cat(lam1)
        out["loop1_exit_share_train"] = float((lam1 >= a.q).float().mean())
    nas_fit.save_router(a.out, router, spec, gamma=a.gamma, k=a.k, steps=a.steps, lr=a.lr, seed=a.seed, fit=diag)
    if a.proxy:
        import core
        import proxy_value
        import tf_eval

        model, tok = core.load_model(core.SERVE_DIR)
        seqs = core.load_teacher("/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl", split="val")
        spec_r = ",".join(f"{a.q:.6f}" for _ in range(3))
        res = tf_eval.evaluate(model, seqs, [tf_eval.parse_router(spec_r)], tok.pad_token_id, 2500, gate=router.to(torch.bfloat16),
                               log=lambda s: None, exact=True)[spec_r]
        own = 100.0 - 100.0 * res["agree_exact"]
        cache = 100.0 * (res["agree_exact"] - res["agree_real"])
        m = proxy_value.load_calib()
        val, p1 = proxy_value.value(res["loops"], own, cache, m)
        out["proxy"] = {"loops": res["loops"], "agree_pct": 100.0 * res["agree_real"], "own_err_pct": own, "cache_err_pct": cache,
                        "pass1_hat": p1, "value_hat": val, "exit_share": res["share"]}
    out["wall_s"] = time.time() - t0
    print("FIT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
