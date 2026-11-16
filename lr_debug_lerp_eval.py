"""Learning-rate debug: held-out loss along the straight line between two checkpoints in weight space.

    theta(alpha) = theta_A + alpha * (theta_B - theta_A)        A = the start of a fine-tuning run (v3), B = its result

alpha = 1 is the fine-tuned checkpoint, alpha = 0 the start; alpha < 1 is the same update direction with a smaller step
(what a proportionally smaller learning rate would have given to first order), alpha > 1 a larger one. If held-out loss is
lowest well below alpha = 1 the run over-stepped; if it keeps falling beyond 1 it under-stepped. No training is needed, and a
good alpha is itself a checkpoint that can be served.

  python -u lr_debug_lerp_eval.py A_DIR B_DIR 0,0.25,0.5,0.75,1,1.5 RENDERED.jsonl TAG
writes <job tmp>/lrval_lerp_<TAG>_a<alpha>.json in the format of fit_eval.py (read by lr_debug_fit_report.py) and prints a table.
Serving venv (transformers); about 12 GB of GPU memory.
"""
import collections
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

SEQ = 4096
T = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
GROUPS = ["oracle_test", "r2_unseen", "replay_unseen", "train_new", "train_replay"]
a_dir, b_dir, alphas, data, tag = sys.argv[1:6]
alphas = [float(x) for x in alphas.split(",")]
rows = [json.loads(line) for line in open(data) if line.strip()]

t0 = time.time()
mb = AutoModelForCausalLM.from_pretrained(b_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
sd_b = {k: v.detach().clone() for k, v in mb.state_dict().items()}
del mb
model = AutoModelForCausalLM.from_pretrained(a_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
sd_a = {k: v.detach().clone() for k, v in model.state_dict().items()}
model = model.cuda().eval()
blend = [k for k, v in sd_a.items() if v.is_floating_point() and k in sd_b and sd_b[k].shape == v.shape]
skipped = [k for k in sd_a if k not in blend]
dist = sum(float((sd_b[k].float() - sd_a[k].float()).pow(2).sum()) for k in blend) ** 0.5
norm = sum(float(sd_a[k].float().pow(2).sum()) for k in blend) ** 0.5
print(f"loaded A={a_dir.split('/')[-1]} B={b_dir.split('/')[-1]} in {time.time() - t0:.0f}s; {len(blend)} tensors blended, {len(skipped)} left as A; "
      f"|B-A| / |A| = {dist / norm:.4f}", flush=True)


def evaluate():
    out = []
    with torch.inference_mode():
        for row in rows:
            ids, labels = row["ids"], row["labels"]
            n_sup = ce_sum = correct = 0
            for w in range(0, len(ids) - 1, SEQ):
                x, y = ids[w:w + SEQ + 1], labels[w:w + SEQ + 1]
                if len(x) < 2:
                    continue
                inp = torch.tensor(x[:-1], device="cuda")[None]
                tgt = torch.tensor(y[1:], device="cuda")
                mask = tgt != -100
                if not mask.any():
                    continue
                logits = model(input_ids=inp).logits[0]
                lg, tt = logits[mask].float(), tgt[mask]
                ce_sum += F.cross_entropy(lg, tt, reduction="sum").item()
                correct += (lg.argmax(-1) == tt).sum().item()
                n_sup += int(mask.sum().item())
                del logits, lg
            out.append({"i": row["i"], "group": row["group"], "tokens": len(ids), "sup": n_sup,
                        "ce": ce_sum / max(n_sup, 1), "acc": correct / max(n_sup, 1)})
    return out


table = []
target = dict(model.named_parameters())
target.update(dict(model.named_buffers()))
for alpha in alphas:
    t1 = time.time()
    with torch.no_grad():
        for k in blend:
            a = sd_a[k].float()
            target[k].copy_((a + alpha * (sd_b[k].float() - a)).to(target[k].dtype))
    res = evaluate()
    json.dump({"ckpt": f"{a_dir} -> {b_dir} at alpha {alpha}", "data": data, "rows": res},
              open(f"{T}/lrval_lerp_{tag}_a{alpha:.2f}.json", "w"), indent=1)
    agg = collections.OrderedDict()
    for g in GROUPS:
        rs = [r for r in res if r["group"] == g and r["sup"] > 0]
        sup = sum(r["sup"] for r in rs)
        if sup:
            agg[g] = (sum(r["ce"] * r["sup"] for r in rs) / sup, sum(r["acc"] * r["sup"] for r in rs) / sup)
    table.append((alpha, agg))
    print(f"alpha {alpha:4.2f}  " + "  ".join(f"{g} {v[0]:.3f} ({100 * v[1]:.1f})" for g, v in agg.items()) + f"   [{time.time() - t1:.0f}s]", flush=True)

print(f"\nLERP {tag}: token loss (accuracy %) along A -> B")
print(f"{'alpha':>6s}" + "".join(f"{g:>22s}" for g in GROUPS))
for alpha, agg in table:
    print(f"{alpha:6.2f}" + "".join((f"{agg[g][0]:13.3f} ({100 * agg[g][1]:5.1f})" if g in agg else f"{'':22s}") for g in GROUPS))
print("LERP done", tag, flush=True)
