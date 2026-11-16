"""In-training validation loss on pre-rendered rows (debug / model selection; active when OURO_VAL_FILE is set).

    OURO_VAL_FILE=<rendered.jsonl>   rows {"group", "ids", "labels"} as written by render_sft_rows.py / lr_debug_valset.py
    OURO_VAL_OUT=<out.jsonl>         one line per (step, group); default <dump_folder>/val_probe.jsonl
    OURO_VAL_EVERY=<n>               validate every n optimizer steps (default 134), at step 0 and at the last step

Each row is cut into training-length windows and run teacher-forced, exactly like training (same model object, same
autocast, no gradient). Reported per group, on supervised tokens only:
  ce_final / acc_final   token loss and top-1 accuracy of the LAST recurrence step (what the served model emits)
  ce_step[t]             token loss of every recurrence step (the Stage-I objective trains all of them)
  ce_mix                 the exit-weighted loss sum_t p(t) CE_t (the data term of the training loss)
  exit_pdf[t]            mean exit probability of each step

Terminal-Bench scores cost 6 to 23 hours per checkpoint and move by a few trials between identical runs; this costs about two
minutes and resolves differences of a percent, so a learning-rate or schedule change can be judged on held-out loss first.
"""
from __future__ import annotations

import collections
import json
import os
import time

import torch
import torch.nn.functional as F

from torchtitan.tools.logging import logger

_ROWS: dict[str, list] = {}


def _rows(path: str) -> list:
    if path not in _ROWS:
        _ROWS[path] = [json.loads(line) for line in open(path) if line.strip()]
    return _ROWS[path]


def maybe_validate(trainer, force: bool = False) -> None:
    path = os.environ.get("OURO_VAL_FILE")
    if not path:
        return
    every = int(os.environ.get("OURO_VAL_EVERY", "134"))
    step = int(trainer.step)
    if not force and step % every != 0 and step != trainer.config.training.steps:
        return
    out_path = os.environ.get("OURO_VAL_OUT") or os.path.join(trainer.config.dump_folder, "val_probe.jsonl")
    seq = int(trainer.config.training.seq_len)
    model = trainer.model_parts[0]
    device = trainer.device
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    t0 = time.time()
    lr = float(trainer.optimizers.optimizers[0].param_groups[0]["lr"])
    with torch.no_grad():
        for row in _rows(path):
            ids, labels, g = row["ids"], row["labels"], row.get("group", "all")
            for w in range(0, len(ids) - 1, seq):
                x, y = ids[w:w + seq + 1], labels[w:w + seq + 1]
                if len(x) < 2:
                    continue
                tgt = torch.tensor(y[1:], device=device)
                mask = tgt != -100
                if not bool(mask.any()):
                    continue
                inp = torch.tensor(x[:-1], device=device)[None]
                with trainer.maybe_enable_amp:
                    pred = model(inp)
                tt = tgt[mask]
                a = acc[g]
                if isinstance(pred, dict):
                    steps = pred["stacked_step_logits"][0][mask]                  # [n, V, T]
                    pdf = pred["stacked_exit_pdf"][0][mask].float()               # [n, T]
                    n_steps = steps.shape[-1]
                    for t in range(n_steps):
                        ce_t = F.cross_entropy(steps[..., t].float(), tt, reduction="none")
                        a[f"ce_step{t + 1}"] += float(ce_t.sum())
                        a["ce_mix"] += float((ce_t * pdf[:, t]).sum())
                        a[f"pdf{t + 1}"] += float(pdf[:, t].sum())
                        if t == n_steps - 1:
                            a["ce_final"] += float(ce_t.sum())
                            a["correct"] += float((steps[..., t].argmax(-1) == tt).sum())
                    a["n_steps"] = n_steps
                    del steps, pdf
                else:                                                             # "standard" stage: expected logits only
                    lg = pred[0][mask].float()
                    a["ce_final"] += float(F.cross_entropy(lg, tt, reduction="sum"))
                    a["correct"] += float((lg.argmax(-1) == tt).sum())
                    del lg
                a["n"] += int(mask.sum())
                a["rows"] += 1 if w == 0 else 0
                del pred
    took = time.time() - t0
    parts = []
    with open(out_path, "a") as f:
        for g, a in acc.items():
            n = max(a["n"], 1.0)
            rec = {"step": step, "lr": lr, "group": g, "rows": int(a["rows"]), "tokens": int(a["n"]),
                   "ce_final": a["ce_final"] / n, "acc_final": a["correct"] / n, "seconds": round(took, 1)}
            k = int(a.get("n_steps", 0))
            if k:
                rec["ce_step"] = [a[f"ce_step{t + 1}"] / n for t in range(k)]
                rec["exit_pdf"] = [a[f"pdf{t + 1}"] / n for t in range(k)]
                rec["ce_mix"] = a["ce_mix"] / n
            f.write(json.dumps(rec) + "\n")
            parts.append(f"{g} {rec['ce_final']:.3f}/{100 * rec['acc_final']:.1f}%")
    logger.info(f"[val] step {step} lr {lr:.2e} ({took:.0f}s): " + "  ".join(parts))
    torch.cuda.empty_cache()
