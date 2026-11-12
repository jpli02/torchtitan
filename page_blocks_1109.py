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
        su = f'{e["speedup"]:.2f}×' if e.get("speedup") else f'{e["tok_s"] / LCB342_FULL_TOK_S:.2f}×'
        flag = "" if e.get("cal_clean", True) else _sm(" shared card")
        rows.append(f'<tr{" class=\"best\"" if e is best else ""}><td class="n">{k}</td><td>({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})</td><td class="n">{e["pass1_pct"]:.1f}</td>'
                    f'<td class="n">{e["tok_s"]:.2f}{flag}</td><td class="n">{su}</td><td class="n">{_pct(e.get("trunc_rate") or 0, 0)} %</td>'
                    f'<td class="n">{e["value"]:.2f}</td><td class="n">{e["wall_s"] / 3600:.1f} h · GPU {e["gpu"]}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="8">evaluations running (restarted 2026-11-10 12:30 with calibrated speed; the four accuracies already measured are reused)</td></tr>')
    log = sorted(glob.glob(f"{TMP}/lcb_bo_pooled*.log"))[-1] if glob.glob(f"{TMP}/lcb_bo_pooled*.log") else ""
    finished = bool(log) and "Completed 1 experiment" in open(log, errors="replace").read()
    refs = [(d, _pooled342(k)) for k, d in (("full", "full depth"), ("gate0.5", "released gate q = 0.5"), ("gate0.3", "released gate q = 0.3"))]
    ref_txt = "; ".join(f'{d} {_pct(r["acc"])} % at {r["tok_s"]:.1f} tokens/s' for d, r in refs if r)
    running = bool(glob.glob("/proc/[0-9]*/cmdline")) and any(
        "bo_lcb_router_pooled" in open(c, errors="replace").read() and "main.py" in open(c, errors="replace").read()
        for c in glob.glob("/proc/[0-9]*/cmdline") if os.access(c, os.R_OK))
    status = f'{"finished" if finished else ("running" if running else "paused until a card is entirely free (other users hold cards 0 and 2 to 5; card 1 runs the epoch evaluations); resumes automatically")}: {len(evs)} of 20 evaluations done'
    if best:
        t = best["thresholds"][:3]
        status += f'; best so far ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): {best["pass1_pct"]:.1f} % at {best["tok_s"]:.2f} tokens/s, objective {best["value"]:.2f}'
    return f'''<div class="sub-head"><h3>Second BO run: pooled objective over all 342 problems {_chip("good", "finished") if finished else _chip("info", "running")}</h3><p>The corrected protocol after the held-out failure above: every evaluation serves all 342 LiveCodeBench problems under real decode-time early exit (batch 1, greedy, 1,024 tokens, sandbox grading), so there is no unseen slice to overfit against; the accuracy floor is the measured full-depth pass@1 over the 342 (20.5 %) minus one standard error at this size (2.2 points), i.e. 18.3 %. Speed is no longer read off the long generation: the first attempt lost 28 % of one router's tokens/s to a neighbour's job on the shared card, so each point's tokens/s now comes from a calibration run on a stratified set of 32 problems (every 11th of the 342: the first 30 problems under-read early-exit speed-ups by a third, being prefill-heavy) whose card is checked for other users' compute before and after and repeated if it was shared (flagged in the table if it stayed shared); the full-depth speed is calibrated right after it on the same card, and the speed term is that speed-up times 6.25 tokens/s so cards of different speed cancel out. Every point measured before the stratified set, and the warm-start points, were re-calibrated on it on 2026-11-11 before the search continued. A failed generation is retried once on another card. gp_hedge over (q1, q2, q3). Two further changes on 2026-11-11: the run is warm-started through the optimizer's resume state with 7 routers whose pooled accuracy was already known (full depth, the released gates at 0.7 / 0.5 / 0.3, and the three slice-run routers, with their held-out accuracies pooled over the 342 and their quiet-card speed-ups), so the surrogate starts from 12 points instead of 4 random ones; and the accuracy generation runs as three concurrent batch-1 processes on the card (batched decoding under true early exit changes the outputs, 23 of 24 problems diverged, so it cannot be used; batch 1 itself is exactly reproducible, 24 of 24 generations identical across runs), which cuts a point from 3 to 7 hours to roughly 1.5 to 3. Cards join as the Terminal-Bench evaluations release them.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th class="n">#</th><th>thresholds (q1, q2, q3)</th><th class="n">pass@1 % (342)</th><th class="n">tokens / s (calibrated)</th><th class="n">speed-up vs full depth</th><th class="n">truncated</th><th class="n">objective</th><th class="n">cost</th></tr></thead>
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
    """(resolved, results, resolved tasks) over the run's main root and its resume root (resume_NAME/NAME), one result per task."""
    seen = {}
    for pattern in (f"{root}/{name}/*/*/results.json", f"{root}/resume_{name}/{name}/*/*/results.json"):
        for f in sorted(glob.glob(pattern)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            key = (d.get("task_id"), f.split("/")[-2])
            if key not in seen:
                seen[key] = d
    n = len(seen)
    ok = sum(1 for d in seen.values() if d.get("is_resolved"))
    tasks = {d.get("task_id") for d in seen.values() if d.get("is_resolved")}
    return ok, n, len(tasks)


def _oom_turns(name):
    """Out-of-memory turns in an evaluation's serving log (a card another user half-filled cannot prefill 8k-token contexts)."""
    p = f"/home/jli199/terminal_bench_eval/logs/{name}_server.log"
    if not os.path.exists(p):
        return 0, 0
    oom = turns = 0
    for line in open(p, errors="replace"):
        if "OOM at step" in line:
            oom += 1
        elif "TURN prompt" in line:
            turns += 1
    return oom, turns


def cont_block():
    losses = _cont_losses()
    rows = ['<tr class="dim"><td>r1 (1.67 epochs, reference)</td><td class="n">890</td><td class="n">0.351 → 0.093</td><td class="n">0.093</td><td class="n">23/75 (31 %)</td><td class="n">7/25</td><td class="n">9/22 · 6 tasks</td><td class="st"><span class="sm">best test checkpoint</span></td></tr>',
            '<tr class="dim"><td>epoch 4 (ep4, the starting point)</td><td class="n">2,144</td><td class="n">0.042</td><td class="n">0.033</td><td class="n">33/75 (44 %)</td><td class="n">8/25</td><td class="n">5/21 · 3 tasks</td><td class="st"><span class="sm">99 % token accuracy</span></td></tr>']
    for k in range(5, 9):
        L = losses.get(k)
        t_ok, t_n, t_tasks = _tb_counts("/tmp/tb_runs", f"tvary_r1_8ep_ep{k}_eval")
        s_ok, s_n, _ = _tb_counts("/srv/scratch/jli199/tb_runs", f"train25_ep{k}_1800")
        f_ok, f_n, _ = _tb_counts("/srv/scratch/jli199/tb_runs", f"train75_ep{k}_1800")
        done_epoch = bool(L and L["complete"])
        loss_mean = (f'{L["mean"]:.3f}' + ("" if L["complete"] else " (so far)")) if L else "—"
        loss_end = f'{L["end"]:.3f}' if done_epoch else "—"
        t_oom, t_turns = _oom_turns(f"tvary_r1_8ep_ep{k}_eval")
        s_oom, s_turns = _oom_turns(f"train25_ep{k}_1800")
        f_oom, f_turns = _oom_turns(f"train75_ep{k}_1800")
        test = f'{t_ok}/{t_n} · {t_tasks} tasks' if t_n else ("not yet" if done_epoch else "—")
        sub = (f'{s_ok}/{s_n}' + ("" if s_n >= 25 else " so far")) if s_n else "—"
        full = (f'{f_ok}/{f_n}' + (f" ({100 * f_ok / f_n:.0f} %)" if f_n >= 75 else " so far")) if f_n else ("queued" if done_epoch else "—")
        if t_n and t_oom:
            test += _sm(f" {t_oom} of {t_turns} turns out of memory: card shared, rerun")
        if s_n and s_oom:
            sub += _sm(f" {s_oom} of {s_turns} turns out of memory: card shared, rerun")
        if f_n and f_oom:
            full += _sm(f" {f_oom} of {f_turns} turns out of memory: card shared, rerun")
        staged = bool(glob.glob(f"{CONT_DUMP}_ep{k}_clean/model*.safetensors"))
        note = "" if not done_epoch else ("done" if f_n >= 75 else ("evaluating on all 75 tasks" if f_n else ("staged, queued for a free card" if staged else "checkpoint saved, queued")))
        rows.append(f'<tr><td>epoch {k}</td><td class="n">{2144 + CONT_SPE * (k - 4):,}</td><td class="n">{loss_mean}</td><td class="n">{loss_end}</td><td class="n">{full}</td><td class="n">{sub}</td><td class="n">{test}</td><td class="st">{_sm(note)}</td></tr>')
    trained = os.path.exists(f"{CONT_DUMP}/TRAIN_DONE")
    if trained:
        all_done = all(_tb_counts("/srv/scratch/jli199/tb_runs", f"train75_ep{k}_1800")[1] >= 75 for k in range(5, 9))
        status = "training finished; evaluations " + ("done" if all_done else "in progress")
    else:
        status = "training on GPU 5 since 2026-11-09 19:33; the evaluation chains on GPUs 3 and 4 stage and score each epoch as it lands"
    if trained and not all_done:
        status += ("; per the user's ask the measure is now every epoch on ALL 75 training tasks (one card runs epochs 6 then 5; epochs 7 and 8 start on the first card another user releases, about 25 hours per epoch per card); the first test evaluations of epochs 6 and 7 are discarded: the evaluation chain had left out the 12k-token context cap that every "
                   "epoch 1 to 4 evaluation used, so contexts grew to 16k tokens and 13 % of the turns failed in the prefill; epoch 7's first training-subset "
                   "run is discarded too (its card was half-filled by another user's job and 99 % of its turns ran out of memory at the cap). Everything is "
                   "being re-run with the cap on the one card we hold entirely (GPU 1), epochs 6, 7, 5, 8 in that order, about 27 hours; epoch 6's "
                   "training-subset run was clean and stands")
    return f'''<div class="sub-head"><h3>Continuing ep4: four more epochs on the same data {_chip("good", "trained") if trained else _chip("info", "running")}</h3><p>Test of the under-fitting hypothesis: ep4 is continued on the same shuffled 1:1 mixture (112 take-turns r1 rows + 112 v3 replay rows) for four more true epochs (536 packed steps each), weights-only load as in every continuation here, a checkpoint at every epoch boundary. Each checkpoint is scored on the 11-task test set (2 attempts, the protocol of the epoch 1 to 4 rows above) and on a fixed 25-task subset of the 75 training tasks (every third task; 1 attempt, 1,800 s), where ep4 solves 8 and r1 7. If the training-subset score keeps rising while the test score does not, the training-task gap is execution, not fit.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Checkpoint</th><th class="n">steps (cumulative)</th><th class="n">training loss, epoch mean</th><th class="n">loss at epoch end</th><th class="n">all 75 training tasks</th><th class="n">training subset (25 tasks)</th><th class="n">test pass@1 (11 tasks × 2)</th><th class="st">Status</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">{status}.</p>
  </div>
'''
