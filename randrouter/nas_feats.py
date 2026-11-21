"""Features for the fast router NAS (ouro_tf_nas.py).

The NAS objective of boptim-agent (objective/ouro_router_nas.py) trains every candidate router from a random initialisation
with the Stage-II loss while the language model is frozen, so the router's inputs never change during that training: they are
the normed states of the frozen model at full depth. This script computes them once. Training a candidate router on the cached
states is then the same optimisation problem as the real Stage II on the same sequences, at a few seconds per candidate.

Per readout position of the base model's own full-depth answers (LiveCodeBench v1-v4, split train: no test problem):
    h    (N, 3, D) float16   normed states after loops 1, 2, 3 (the gate after loop 4 is never used)
    ce   (N, 4)    float32   cross-entropy of the teacher-forced token under the readout of every loop (L_stop of the loss)
    ok   (N, 4)    bool      whether that readout's top token is the teacher-forced token
    gate (N, 3)    float32   the released gate's logits (reference only)
    seq  (N,)      int32     index of the answer the position belongs to

  CUDA_VISIBLE_DEVICES=0 python randrouter/nas_feats.py --mem_gb 8.5
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="/home/jli199/boptim_scratch/randrouter/teacher_lcb_v1v4*.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/nas_feats_train.pt")
    ap.add_argument("--budget_tok", type=int, default=2500)
    ap.add_argument("--mem_gb", type=float, default=0.0)
    ap.add_argument("--max_items", type=int, default=0)
    a = ap.parse_args()
    if a.mem_gb:
        torch.cuda.set_per_process_memory_fraction(a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    model, tok = core.load_model(core.SERVE_DIR)
    seqs = core.load_teacher(a.teacher, split=a.split)
    if a.max_items:
        seqs = seqs[:a.max_items]
    print(f"{len(seqs)} answers of split {a.split}", flush=True)
    hs, ces, oks, gates, seq_idx = [], [], [], [], []
    t0 = time.time()
    batches = core.make_batches(seqs, a.budget_tok)
    for bn, batch in enumerate(batches):
        ids, n_prompt, lengths = core.collate(seqs, batch, tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hid = core.teacher_hidden(model, ids, n_prompt)                    # 4 x (B, T, D), full depth, exact cache
            pos = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
            read = (pos >= (n_prompt[:, None] - 1)) & (pos < (lengths[:, None] - 1))
            bi, ti = read.nonzero(as_tuple=True)
            target = ids[bi, ti + 1]
            ce, ok = [], []
            for t in range(core.N_LOOPS):
                x = hid[t][bi, ti]
                c_parts, o_parts = [], []
                for s in range(0, x.shape[0], 2048):                             # vocabulary-sized logits in chunks
                    c, o = core._head_ce(model.lm_head, x[s:s + 2048], target[s:s + 2048])
                    c_parts.append(c)
                    o_parts.append(o)
                ce.append(torch.cat(c_parts))
                ok.append(torch.cat(o_parts))
            h3 = torch.stack([hid[t][bi, ti] for t in range(core.N_LOOPS - 1)], dim=1)   # (n, 3, D)
            g3 = model.model.early_exit_gate(h3).squeeze(-1).float()
        hs.append(h3.to(torch.float16).cpu())
        ces.append(torch.stack(ce, dim=1).float().cpu())
        oks.append(torch.stack(ok, dim=1).bool().cpu())
        gates.append(g3.cpu())
        seq_idx.append(torch.tensor([batch[int(b)] for b in bi.tolist()], dtype=torch.int32))
        if bn % 10 == 0 or bn == len(batches) - 1:
            print(f"  batch {bn + 1}/{len(batches)}  positions {sum(x.shape[0] for x in hs)}  {time.time() - t0:.0f}s", flush=True)
    out = {"h": torch.cat(hs), "ce": torch.cat(ces), "ok": torch.cat(oks), "gate": torch.cat(gates), "seq": torch.cat(seq_idx),
           "question_ids": [s["question_id"] for s in seqs], "split": a.split, "teacher": a.teacher}
    torch.save(out, a.out)
    n = out["h"].shape[0]
    ce_m = out["ce"].mean(0).tolist()
    ok_m = out["ok"].float().mean(0).tolist()
    imp = torch.relu(out["ce"][:, :-1] - out["ce"][:, 1:])
    print(f"saved {a.out}: {n} positions of {len(seqs)} answers, {os.path.getsize(a.out) / 2**20:.0f} MB")
    print("  mean cross-entropy per loop " + ", ".join(f"{v:.4f}" for v in ce_m) + "; top-1 agreement per loop " + ", ".join(f"{100 * v:.2f}" for v in ok_m))
    for j in range(3):
        q = torch.quantile(imp[:, j], torch.tensor([0.5, 0.9, 0.99]))
        print(f"  improvement loop {j + 1} -> {j + 2}: mean {imp[:, j].mean():.4f}, median {q[0]:.5f}, 90 % {q[1]:.4f}, 99 % {q[2]:.3f}, "
              f"share above 0.005 / 0.05 / 0.5 nats: {100 * (imp[:, j] > 0.005).float().mean():.1f} / {100 * (imp[:, j] > 0.05).float().mean():.1f} / {100 * (imp[:, j] > 0.5).float().mean():.1f} %")


if __name__ == "__main__":
    main()
