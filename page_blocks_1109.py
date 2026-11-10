"""Page blocks added 2026-11-09: the pooled-objective LiveCodeBench BO (all 342 problems) and the ep4 continuation (epochs 5 to 8
with per-epoch test and training-subset evaluations). Each function re-reads its result files on every page build."""
import glob
import json
import os

from page_blocks_1107 import TMP, _chip, _pct, _sm

# ---------------------------------------------------------------- Q12: pooled-objective BO on LiveCodeBench (all 342 problems)
POOLED_EVALS = "/home/jli199/boptim_scratch/bo_lcb_router_pooled/evaluations.jsonl"
LCB342_FULL_TOK_S = 6.25  # full depth pooled over the three slices


def _pooled342(label):
    """RESULT + GRADED lines of label, label_o120, label_o240 pooled over all 342 problems (wall-clock reports)."""
    n = passed = toks = wall = 0.0
    for suf in ("", "_o120", "_o240"):
        res = grd = None
        for rep in (f"{TMP}/wallclock_lcb.txt", f"{TMP}/wallclock_lcb_bo.txt"):
            if not os.path.exists(rep):
                continue
            for line in open(rep):
                if line.startswith(f"RESULT {label}{suf} "):
                    res = json.loads(line[len(f"RESULT {label}{suf} "):])
                if line.startswith(f"{label}{suf} GRADED "):
                    kv = dict(p.split("=", 1) for p in line.split()[2:] if "=" in p and not p.startswith("by_difficulty"))
                    grd = {"n": int(kv["n"]), "acc": float(kv["pass@1"])}
        if res and grd:
            n += grd["n"]
            passed += grd["acc"] * grd["n"]
            toks += res["mean_tokens"] * res["n"]
            wall += res["wall_s"]
    return {"n": int(n), "acc": passed / n, "tok_s": toks / wall} if n else None


def lcb_bo_pooled_block():
    evs = [json.loads(l) for l in open(POOLED_EVALS) if l.strip()] if os.path.exists(POOLED_EVALS) else []
    evs = [e for e in evs if e.get("status") == "ok"]
    best = min(evs, key=lambda e: e["value"]) if evs else None
    rows = []
    for k, e in enumerate(evs, 1):
        t = e["thresholds"][:3]
        rows.append(f'<tr{" class=\"best\"" if e is best else ""}><td class="n">{k}</td><td>({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})</td><td class="n">{e["pass1_pct"]:.1f}</td>'
                    f'<td class="n">{e["tok_s"]:.2f}</td><td class="n">{e["tok_s"] / LCB342_FULL_TOK_S:.2f}×</td><td class="n">{_pct(e.get("trunc_rate") or 0, 0)} %</td>'
                    f'<td class="n">{e["value"]:.2f}</td><td class="n">{e["wall_s"] / 3600:.1f} h · GPU {e["gpu"]}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="8">first three evaluations running (started 2026-11-09 19:32)</td></tr>')
    log = f"{TMP}/lcb_bo_pooled.log"
    finished = os.path.exists(log) and "Completed 1 experiment" in open(log, errors="replace").read()
    refs = [(d, _pooled342(k)) for k, d in (("full", "full depth"), ("gate0.5", "released gate q = 0.5"), ("gate0.3", "released gate q = 0.3"))]
    ref_txt = "; ".join(f'{d} {_pct(r["acc"])} % at {r["tok_s"]:.1f} tokens/s' for d, r in refs if r)
    status = f'{"finished" if finished else "running"}: {len(evs)} of 21 evaluations done'
    if best:
        t = best["thresholds"][:3]
        status += f'; best so far ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): {best["pass1_pct"]:.1f} % at {best["tok_s"]:.2f} tokens/s, objective {best["value"]:.2f}'
    return f'''<div class="sub-head"><h3>Second BO run: pooled objective over all 342 problems {_chip("good", "finished") if finished else _chip("info", "running")}</h3><p>The corrected protocol after the held-out failure above: every evaluation now serves all 342 LiveCodeBench problems under real decode-time early exit (batch 1, greedy, 1,024 tokens, sandbox grading), so there is no unseen slice to overfit against; the accuracy floor is the measured full-depth pass@1 over the 342 (20.5 %) minus one standard error at this size (2.2 points), i.e. 18.3 %; a failed generation is retried once on another card before it counts. gp_hedge over (q1, q2, q3), three evaluations in parallel on GPUs 0 to 2, 4 random points then 17 proposed, 3 to 6 hours each.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th class="n">#</th><th>thresholds (q1, q2, q3)</th><th class="n">pass@1 % (342)</th><th class="n">tokens / s</th><th class="n">speed-up vs full depth</th><th class="n">truncated</th><th class="n">objective</th><th class="n">cost</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">Reference points over the same 342 problems: {ref_txt}. Objective = −tokens/s + 2 × max(0, 18.3 − pass@1 %). {status}.</p>
  </div>
'''


# ---------------------------------------------------------------- Q11: continuing ep4 for more epochs
CONT_DUMP = "/home/jli199/boptim_scratch/ouro_tvary_r1_8ep"
CONT_SPE = 536


def _cont_losses():
    """Per-epoch training-loss mean/end of the continuation from its offline wandb file (epochs 5..8 = steps 1..2144)."""
    try:
        import extract_wandb_loss as ew
    except Exception:
        return {}
    files = sorted(glob.glob(f"{CONT_DUMP}/tb/*/wandb/offline-run-*/run-*.wandb"))
    if not files:
        return {}
    try:
        rows, _ = ew.read_history(files[-1])
    except Exception:
        return {}
    key = "loss_metrics/global_avg_loss"
    pts = sorted((int(r["_step"]), float(r[key])) for r in rows if r.get("_step") is not None and key in r)
    out = {}
    for k in range(5, 9):
        lo, hi = CONT_SPE * (k - 5), CONT_SPE * (k - 4)
        ep = [v for s, v in pts if lo < s <= hi]
        if ep:
            out[k] = {"mean": sum(ep) / len(ep), "end": sum(ep[-3:]) / len(ep[-3:]), "complete": any(s >= hi - 10 for s, _ in pts)}
    return out


def _tb_counts(root, name):
    n = ok = 0
    tasks = set()
    for f in glob.glob(f"{root}/{name}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n += 1
        if d.get("is_resolved"):
            ok += 1
            tasks.add(d.get("task_id"))
    return ok, n, len(tasks)


def cont_block():
    losses = _cont_losses()
    rows = ['<tr class="dim"><td>r1 (1.67 epochs, reference)</td><td class="n">890</td><td class="n">0.351 → 0.093</td><td class="n">0.093</td><td class="n">9/22 · 6 tasks</td><td class="n">7/25</td><td class="st"><span class="sm">best test checkpoint</span></td></tr>',
            '<tr class="dim"><td>epoch 4 (ep4, the starting point)</td><td class="n">2,144</td><td class="n">0.042</td><td class="n">0.033</td><td class="n">5/21 · 3 tasks</td><td class="n">8/25</td><td class="st"><span class="sm">99 % token accuracy; 33/75 on the whole training set</span></td></tr>']
    for k in range(5, 9):
        L = losses.get(k)
        t_ok, t_n, t_tasks = _tb_counts("/tmp/tb_runs", f"tvary_r1_8ep_ep{k}_eval")
        s_ok, s_n, _ = _tb_counts("/srv/scratch/jli199/tb_runs", f"train25_ep{k}_1800")
        done_epoch = bool(L and L["complete"])
        loss_mean = (f'{L["mean"]:.3f}' + ("" if L["complete"] else " (so far)")) if L else "—"
        loss_end = f'{L["end"]:.3f}' if done_epoch else "—"
        test = f'{t_ok}/{t_n} · {t_tasks} tasks' if t_n else ("running" if done_epoch else "—")
        sub = (f'{s_ok}/{s_n}' + ("" if s_n >= 25 else " so far")) if s_n else ("queued" if done_epoch else "—")
        staged = os.path.isdir(f"{CONT_DUMP}_ep{k}_clean")
        note = "" if not done_epoch else ("done" if t_n >= 22 and s_n >= 25 else ("staged, evaluating" if staged else "checkpoint saved"))
        rows.append(f'<tr><td>epoch {k}</td><td class="n">{2144 + CONT_SPE * (k - 4):,}</td><td class="n">{loss_mean}</td><td class="n">{loss_end}</td><td class="n">{test}</td><td class="n">{sub}</td><td class="st">{_sm(note)}</td></tr>')
    trained = os.path.exists(f"{CONT_DUMP}/TRAIN_DONE")
    if trained:
        all_done = all(_tb_counts("/srv/scratch/jli199/tb_runs", f"train25_ep{k}_1800")[1] >= 25 for k in range(5, 9))
        status = "training finished; evaluations " + ("done" if all_done else "in progress")
    else:
        status = "training on GPU 5 since 2026-11-09 19:33; the evaluation chains on GPUs 3 and 4 stage and score each epoch as it lands"
    return f'''<div class="sub-head"><h3>Continuing ep4: four more epochs on the same data {_chip("good", "trained") if trained else _chip("info", "running")}</h3><p>Test of the under-fitting hypothesis: ep4 is continued on the same shuffled 1:1 mixture (112 take-turns r1 rows + 112 v3 replay rows) for four more true epochs (536 packed steps each), weights-only load as in every continuation here, a checkpoint at every epoch boundary. Each checkpoint is scored on the 11-task test set (2 attempts, the protocol of the epoch 1 to 4 rows above) and on a fixed 25-task subset of the 75 training tasks (every third task; 1 attempt, 1,800 s), where ep4 solves 8 and r1 7. If the training-subset score keeps rising while the test score does not, the training-task gap is execution, not fit.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Checkpoint</th><th class="n">steps (cumulative)</th><th class="n">training loss, epoch mean</th><th class="n">loss at epoch end</th><th class="n">test pass@1 (11 tasks × 2)</th><th class="n">training subset (25 tasks)</th><th class="st">Status</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">{status}.</p>
  </div>
'''
