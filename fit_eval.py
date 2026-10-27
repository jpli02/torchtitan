"""Teacher-forced fit of a checkpoint on pre-rendered SFT rows (underfitting check).

Usage: python -u fit_eval.py CKPT_DIR RENDERED.jsonl OUT.json [MAX_ROWS]
Runs in the serving venv (transformers 4.57). RENDERED.jsonl comes from
render_sft_rows.py: {"i", "group", "ids", "labels"} per row, labels = -100 where
not supervised. Each row is cut into 4096-token windows like training and run
teacher-forced; reports per-token CE and top-1 accuracy on supervised tokens,
split by group ("new" generated rows vs "replay" v3 rows).
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

SEQ = 4096
ckpt, data, out = sys.argv[1:4]
max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.cuda().eval()
print(f"loaded {ckpt} in {time.time()-t0:.0f}s; params={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

rows_out = []
t0 = time.time()
with torch.inference_mode():
    for n_done, line in enumerate(open(data)):
        if not line.strip():
            continue
        if n_done >= max_rows:
            break
        row = json.loads(line)
        ids, labels = row["ids"], row["labels"]
        n_sup = ce_sum = correct = 0
        for w in range(0, len(ids) - 1, SEQ):
            x = ids[w:w + SEQ + 1]
            y = labels[w:w + SEQ + 1]
            if len(x) < 2:
                continue
            inp = torch.tensor(x[:-1], device="cuda")[None]
            tgt = torch.tensor(y[1:], device="cuda")
            mask = tgt != -100
            if not mask.any():
                continue
            logits = model(input_ids=inp).logits[0]
            lg = logits[mask].float()
            tt = tgt[mask]
            ce_sum += F.cross_entropy(lg, tt, reduction="sum").item()
            correct += (lg.argmax(-1) == tt).sum().item()
            n_sup += int(mask.sum().item())
            del logits, lg
        rows_out.append({"i": row["i"], "group": row["group"], "tokens": len(ids), "sup": n_sup,
                         "ce": ce_sum / max(n_sup, 1), "acc": correct / max(n_sup, 1)})
        if len(rows_out) % 25 == 0:
            print(f"  {len(rows_out)} rows, {time.time()-t0:.0f}s", flush=True)

summary = {}
for grp in ("new", "replay"):
    rs = [r for r in rows_out if r["group"] == grp]
    if not rs:
        continue
    sup = sum(r["sup"] for r in rs)
    summary[grp] = {
        "rows": len(rs),
        "sup_tokens": sup,
        "token_ce": sum(r["ce"] * r["sup"] for r in rs) / sup,
        "token_acc": sum(r["acc"] * r["sup"] for r in rs) / sup,
        "row_ce_median": statistics.median(r["ce"] for r in rs),
        "row_acc_median": statistics.median(r["acc"] for r in rs),
        "rows_acc_ge_0.95": sum(r["acc"] >= 0.95 for r in rs) / len(rs),
        "rows_acc_ge_0.99": sum(r["acc"] >= 0.99 for r in rs) / len(rs),
        "rows_acc_lt_0.80": sum(r["acc"] < 0.80 for r in rs) / len(rs),
    }
json.dump({"ckpt": ckpt, "data": data, "summary": summary, "rows": rows_out}, open(out, "w"), indent=1)
print(f"\n{os.path.basename(ckpt)} on {os.path.basename(data)}  ({time.time()-t0:.0f}s)")
for grp, s in summary.items():
    print(f"  {grp:6s} rows={s['rows']:3d} sup_tok={s['sup_tokens']:,} token_CE={s['token_ce']:.3f} token_acc={s['token_acc']:.3f} "
          f"row_acc_median={s['row_acc_median']:.3f} rows>=0.95acc={s['rows_acc_ge_0.95']:.0%} "
          f"rows>=0.99={s['rows_acc_ge_0.99']:.0%} rows<0.80={s['rows_acc_lt_0.80']:.0%}", flush=True)
