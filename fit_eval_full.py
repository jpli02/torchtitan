"""Full-context teacher-forced fit by absolute position, against the 4,096-token windowed view used in training.

Usage: python fit_eval_full.py CKPT RENDERED.jsonl OUT.json [GROUP=new] [MAX_ROWS] [CAP=12288]
Training packs rows into 4,096-token windows, so a 10k-token trajectory is seen as three context-free chunks; at
eval time the agent runs with up to 12k tokens of real context. For each row (truncated to CAP tokens) this runs
(a) one forward pass over the whole row and (b) each 4,096 window alone (the training view), and reports supervised
top-1 accuracy / CE per absolute-position bucket [0,4k) [4k,8k) [8k,12k) for both views. A full-context accuracy
well below the windowed one at positions >= 4k is a context-length mismatch the SFT never trained."""
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

SEQ = 4096
ckpt, data, out = sys.argv[1:4]
group = sys.argv[4] if len(sys.argv) > 4 else "new"
max_rows = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
CAP = int(sys.argv[6]) if len(sys.argv) > 6 else 12288
NB = CAP // SEQ

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
print(f"loaded {ckpt} in {time.time()-t0:.0f}s", flush=True)

agg = {"full": [[0, 0.0, 0] for _ in range(NB)], "win": [[0, 0.0, 0] for _ in range(NB)]}  # per bucket: correct, ce_sum, n
rows_out = []


def score(x, y, offset, key):
    """x: input ids (list), y: labels aligned with x; offset = absolute position of x[0]."""
    inp = torch.tensor(x[:-1], device="cuda")[None]
    tgt = torch.tensor(y[1:], device="cuda")
    mask = tgt != -100
    if not mask.any():
        return
    logits = model(input_ids=inp).logits[0]
    pos = torch.arange(offset + 1, offset + len(x), device="cuda")
    lg = logits[mask].float()
    tt = tgt[mask]
    pb = (pos[mask] // SEQ).clamp(max=NB - 1)
    ce = F.cross_entropy(lg, tt, reduction="none")
    ok = (lg.argmax(-1) == tt)
    for b in range(NB):
        sel = pb == b
        if sel.any():
            agg[key][b][0] += int(ok[sel].sum().item())
            agg[key][b][1] += float(ce[sel].sum().item())
            agg[key][b][2] += int(sel.sum().item())
    del logits, lg
    return int(ok.sum().item()), int(mask.sum().item())


t0 = time.time()
n_rows = 0
with torch.inference_mode():
    for line in open(data):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["group"] != group:
            continue
        if n_rows >= max_rows:
            break
        ids, labels = row["ids"][:CAP], row["labels"][:CAP]
        if len(ids) < 2:
            continue
        before = [list(b) for b in agg["full"]]
        score(ids, labels, 0, "full")
        for w in range(0, len(ids) - 1, SEQ):
            score(ids[w:w + SEQ + 1], labels[w:w + SEQ + 1], w, "win")
        per_row = [(agg["full"][b][0] - before[b][0], agg["full"][b][2] - before[b][2]) for b in range(NB)]
        rows_out.append({"i": row["i"], "tokens": len(row["ids"]), "full_bucket_correct_n": per_row})
        n_rows += 1
        if n_rows % 20 == 0:
            print(f"  {n_rows} rows, {time.time()-t0:.0f}s", flush=True)

summary = {}
for key in ("full", "win"):
    summary[key] = [{"bucket": f"[{b*SEQ},{(b+1)*SEQ})", "n": n, "acc": (c / n if n else None), "ce": (s / n if n else None)}
                    for b, (c, s, n) in enumerate(agg[key])]
json.dump({"ckpt": ckpt, "data": data, "group": group, "rows": n_rows, "cap": CAP, "summary": summary, "per_row": rows_out}, open(out, "w"), indent=1)
print(f"\n{ckpt.split('/')[-1]} on {group} rows of {data.split('/')[-1]} (n={n_rows}, cap {CAP}) {time.time()-t0:.0f}s")
print(f"{'bucket':>16} {'full acc':>9} {'win acc':>8} {'full CE':>8} {'win CE':>7} {'tokens':>8}")
for b in range(NB):
    f_, w_ = summary["full"][b], summary["win"][b]
    if f_["n"]:
        print(f"{f_['bucket']:>16} {f_['acc']:9.3f} {w_['acc']:8.3f} {f_['ce']:8.3f} {w_['ce']:7.3f} {f_['n']:8d}")
