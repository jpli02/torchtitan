"""2026-11-16 block: learning-rate debug of the Terminal-Bench fine-tuning (user's first priority).

Reads the artefacts of the lr_debug_* scripts from a durable copy in boptim_scratch/lr_debug (collected from the job tmp at
build time): per-checkpoint validation losses (lrval_<name>.json), validation curves written inside training
(valprobe_<tag>.jsonl), weight-space line searches (lrval_lerp_<tag>_a<alpha>.json) and the optimizer probe.
"""
import collections
import glob
import json
import os
import re
import shutil
import statistics

import page_blocks_1109 as pb2
from page_blocks_1107 import _chip, _sm

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
KEEP = "/home/jli199/boptim_scratch/lr_debug"
LOGS = "/home/jli199/terminal_bench_eval/logs"
GROUPS = ["oracle_test", "r2_unseen", "replay_unseen", "train_new"]
NAME = {"oracle_test": "test tasks, oracle runs", "r2_unseen": "training tasks, unseen runs",
        "replay_unseen": "earlier training rows, not replayed", "train_new": "rows trained on"}
SHORT = {"oracle_test": "test tasks", "r2_unseen": "unseen runs", "replay_unseen": "not replayed", "train_new": "trained on"}
HEAD = {"oracle_test": "test tasks<br>(oracle runs)", "r2_unseen": "training tasks<br>(unseen runs)", "replay_unseen": "earlier rows<br>(not replayed)",
        "train_new": "rows<br>trained on"}
COL = {"oracle_test": "var(--s1)", "r2_unseen": "var(--s2)", "replay_unseen": "var(--s3)", "train_new": "var(--base)"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _collect():
    """Copy the debug artefacts out of the job tmp (deleted with the job) into boptim_scratch."""
    os.makedirs(KEEP, exist_ok=True)
    for pat in ("lrval_*.json", "valprobe_*.jsonl", "opt_probe_*.jsonl", "lrval*_rendered.jsonl"):
        for p in glob.glob(f"{TMP}/{pat}"):
            q = os.path.join(KEEP, os.path.basename(p))
            if os.path.getsize(p) and (not os.path.exists(q) or os.path.getmtime(p) > os.path.getmtime(q)):
                shutil.copy2(p, q)


def _groups(path):
    """Token loss and accuracy per validation group of one fit file."""
    d = json.load(open(path))
    out = {}
    for g in GROUPS + ["train_replay"]:
        rs = [r for r in d["rows"] if r["group"] == g and r["sup"] > 0]
        sup = sum(r["sup"] for r in rs)
        if sup:
            out[g] = (sum(r["ce"] * r["sup"] for r in rs) / sup, sum(r["acc"] * r["sup"] for r in rs) / sup)
    return out


def _legend():
    return ('<div class="sm" style="display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0 2px">'
            + "".join(f'<span><svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" style="vertical-align:-1px;margin-right:5px">'
                      f'<circle cx="6" cy="6" r="4.5" style="fill:{COL[g]}"/></svg>{NAME[g]}</span>' for g in GROUPS) + '</div>')


def _line_chart(title, xlabel, series, xticks, ymax, xfmt=lambda v: f"{v:g}", w=450, h=250, mark_x=None):
    """One small chart: token loss (single axis) against xlabel; series = {group: [(x, y, tip)]} coloured by validation group."""
    ml, mr, mt, mb = 40, 122, 26, 36                                  # the right margin holds the direct labels at the line ends
    x0, x1 = min(xticks), max(xticks)
    px = lambda x: ml + (x - x0) / (x1 - x0) * (w - ml - mr)  # noqa: E731
    py = lambda y: mt + (1 - min(y, ymax) / ymax) * (h - mt - mb)  # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}" style="width:100%;max-width:{w}px;height:auto;display:block">',
         f'<text x="{ml}" y="14" style="fill:var(--text);font-size:12.5px;font-weight:600">{title}</text>']
    step = 0.25 if ymax <= 1.6 else 0.5
    k = 0
    while k * step <= ymax + 1e-9:
        y = k * step
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{py(y):.1f}" y2="{py(y):.1f}" style="stroke:var(--line);stroke-width:1"/>')
        o.append(f'<text x="{ml - 6}" y="{py(y) + 3.5:.1f}" text-anchor="end" style="fill:var(--text-3);font-size:10.5px">{y:g}</text>')
        k += 1
    for xt in xticks:
        o.append(f'<text x="{px(xt):.1f}" y="{h - mb + 15}" text-anchor="middle" style="fill:var(--text-3);font-size:10.5px">{xfmt(xt)}</text>')
    if mark_x is not None and x0 <= mark_x <= x1:
        o.append(f'<line x1="{px(mark_x):.1f}" x2="{px(mark_x):.1f}" y1="{mt}" y2="{h - mb}" style="stroke:var(--line-2);stroke-width:1;stroke-dasharray:3 3"/>')
    o.append(f'<text x="{(ml + w - mr) / 2:.1f}" y="{h - 4}" text-anchor="middle" style="fill:var(--text-2);font-size:11px">{xlabel}</text>')
    ends = []
    for g in GROUPS:
        pts = sorted(series.get(g) or [])
        if not pts:
            continue
        d = " ".join(f'{"M" if i == 0 else "L"}{px(x):.1f},{py(y):.1f}' for i, (x, y, _) in enumerate(pts))
        o.append(f'<path d="{d}" style="fill:none;stroke:{COL[g]};stroke-width:2"/>')
        for x, y, tip in pts:
            o.append(f'<g><title>{tip}</title><circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" style="fill:{COL[g]};stroke:var(--surface);stroke-width:1.5"/></g>')
        ends.append([py(pts[-1][1]), px(pts[-1][0]), g, pts[-1][1]])
    ends.sort()
    for i in range(1, len(ends)):                                   # keep the direct labels at the line ends apart
        if ends[i][0] - ends[i - 1][0] < 12:
            ends[i][0] = ends[i - 1][0] + 12
    for y, x, g, val in ends:
        o.append(f'<text x="{x + 8:.1f}" y="{y + 3.5:.1f}" style="fill:var(--text-2);font-size:10.5px">{SHORT[g]} {val:.2f}</text>')
    o.append('</svg>')
    return "".join(o)


def _grid(charts):
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:14px 18px;margin:6px 0 4px">'
            + "".join(f'<div style="min-width:0">{c}</div>' for c in charts) + '</div>')


def _probe_summary():
    p = f"{KEEP}/opt_probe_probe_ep4_8bit.jsonl"
    if not os.path.exists(p):
        return None
    by, quant = collections.defaultdict(list), set()
    for line in open(p):
        r = json.loads(line)
        if r.get("probe") == "start":
            quant = {t["name"] for t in r["tensors"] if t["quantised"]}
        elif r["name"] in quant:
            by[r["name"]].append(r)
    if not by:
        return None
    ratio = [statistics.median(r["u_real"] / r["u_exact"] for r in v if r["u_exact"] > 0) for v in by.values()]
    cos = [statistics.median(r["u_cos"] for r in v) for v in by.values()]
    moved = [v[-1]["d_real"] / v[-1]["d_exact"] for v in by.values() if v[-1]["d_exact"] > 0]
    return {"tensors": len(by), "steps": max(len(v) for v in by.values()), "ratio": (min(ratio), max(ratio)), "cos": (min(cos), max(cos)),
            "moved": (min(moved), max(moved)), "over": max(statistics.mean(r["share_over_1p5lr"] for r in v) for v in by.values())}


def _restart_losses():
    """Mean training loss of the epoch 4 -> 8 continuation: first bin, the worst bin at a high rate, the last bins."""
    p = f"{LOGS}/ouro_tvary_r1_8ep_train.log"
    if not os.path.exists(p):
        return None
    rows = [(int(m.group(1)), float(m.group(2))) for m in
            (re.search(r"step:\s*(\d+)\s+loss:\s*([0-9.eE+-]+)", ANSI.sub("", line)) for line in open(p, errors="replace")) if m]
    if not rows:
        return None
    mean = lambda lo, hi: statistics.mean([l for s, l in rows if lo < s <= hi] or [float("nan")])  # noqa: E731
    return {"start": mean(0, 134), "peak": max(mean(134 * k, 134 * (k + 1)) for k in range(2, 7)), "end": mean(1608, 2144)}


def _ckpt_table():
    test_static = {"v3": "8/22", "r1": "9/22", "ep1": "6/22", "ep2": "4/20", "ep4": "5/21"}
    label = {"v3": "v3 (start)", "r1": "take-turns r1", **{f"ep{k}": f"epoch {k}" for k in range(1, 9)}}
    steps = {"v3": "0", "r1": "896, decayed", "ep1": "536, mid-schedule", "ep2": "1,072, mid-schedule", "ep4": "2,144, decayed",
             "ep5": "+536, mid-schedule", "ep6": "+1,072, mid-schedule", "ep7": "+1,608, mid-schedule", "ep8": "+2,144, decayed"}
    rows, got = [], {}
    for n in ["v3", "r1", "ep1", "ep2", "ep4", "ep5", "ep6", "ep7", "ep8"]:
        p = f"{KEEP}/lrval_{n}.json"
        if not os.path.exists(p):
            continue
        s = _groups(p)
        got[n] = s
        test = test_static.get(n)
        if test is None:
            ok, cnt, _ = pb2._tb_counts("/tmp/tb_runs", f"tvary_r1_8ep_{n}_eval")
            test = f"{ok}/{cnt}" if cnt >= 20 else "—"
        cell = lambda g: f'<td class="n">{s[g][0]:.2f}{_sm(f" ({100 * s[g][1]:.1f})")}</td>' if g in s else '<td class="n">—</td>'  # noqa: E731
        rows.append(f'<tr><td>{label[n]}</td><td>{_sm(steps[n])}</td><td class="n">{test}</td>' + "".join(cell(g) for g in GROUPS) + '</tr>')
    return rows, got


def _lerp_section():
    files = glob.glob(f"{KEEP}/lrval_lerp_*_a*.json")
    by = collections.defaultdict(dict)
    for p in files:
        m = re.search(r"lrval_lerp_(.+)_a([0-9.]+)\.json$", os.path.basename(p))
        if m:
            by[m.group(1)][float(m.group(2))] = _groups(p)
    order = [t for t in ["r1", "ep1", "ep4", "ep8"] if t in by]
    if not order:
        return "", None
    title = {"r1": "v3 → take-turns r1 (896 steps)", "ep1": "v3 → epoch 1 (536 steps)", "ep4": "v3 → epoch 4 (2,144 steps)",
             "ep8": "v3 → epoch 8 (4,288 steps)"}
    charts, best = [], {}
    ymax = max(v[g][0] for t in order for v in by[t].values() for g in GROUPS if g in v) * 1.08
    for t in order:
        al = sorted(by[t])
        series = {g: [(a, by[t][a][g][0], f"{NAME[g]}: loss {by[t][a][g][0]:.3f}, accuracy {100 * by[t][a][g][1]:.1f} % at {a:g} of the step")
                      for a in al if g in by[t][a]] for g in GROUPS}
        charts.append(_line_chart(title[t], "share of the fine-tuning step applied (0 = v3, 1 = the checkpoint)", series, al, ymax, mark_x=1.0))
        best[t] = {g: min(al, key=lambda a: by[t][a][g][0]) for g in ("oracle_test", "r2_unseen") if all(g in by[t][a] for a in al)}
    return _grid(charts), {"best": best, "by": by, "order": order}


def _curve_section():
    arms = {}
    for p in sorted(glob.glob(f"{KEEP}/valprobe_lr*_const.jsonl")):
        tag = os.path.basename(p)[9:-6]
        by = collections.defaultdict(dict)
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                by[r["step"]][r["group"]] = r
        if len(by) >= 2:
            arms[tag] = dict(sorted(by.items()))
    if not arms:
        return "", None
    ymax = max(r["ce_final"] for by in arms.values() for d in by.values() for g, r in d.items() if g in GROUPS) * 1.08
    xmax = max(max(by) for by in arms.values())
    xt = [s for s in range(0, xmax + 1, 268)] or [0, xmax]
    if xt[-1] < xmax:
        xt.append(xmax)
    charts = []
    for tag, by in arms.items():
        lr = tag[2:].split("_")[0]
        series = {g: [(s, d[g]["ce_final"], f"{NAME[g]}: loss {d[g]['ce_final']:.3f}, accuracy {100 * d[g]['acc_final']:.1f} % after {s} steps at {lr}")
                      for s, d in by.items() if g in d] for g in GROUPS}
        done = max(by)
        charts.append(_line_chart(f"constant learning rate {lr}" + ("" if done >= 1072 else f" (running, step {done})"), "training steps from v3",
                                  series, xt, ymax))
    return _grid(charts), arms


def lr_debug_block():
    _collect()
    probe, rst = _probe_summary(), _restart_losses()
    rows, got = _ckpt_table()
    lerp_html, lerp = _lerp_section()
    curve_html, arms = _curve_section()
    checks = ['<tr><td>Is the configured schedule the one applied?</td><td>Yes. Rebuilt with the trainer&#39;s own optimizer and scheduler classes, the rate at each of the '
              '2,144 steps equals the formula (50 warm-up steps to 2e-5, cosine decay to 2e-6); the largest relative gap is 1e-7.</td></tr>']
    if probe:
        checks.append(f'<tr><td>Does the 8-bit AdamW take the steps exact AdamW would?</td><td>Yes on real gradients. Over {probe["steps"]} training steps from the epoch 4 '
                      f'checkpoint, on {probe["tensors"]} weight tensors, the applied step is {probe["ratio"][0]:.2f} to {probe["ratio"][1]:.2f} times the exact one in size, '
                      f'with direction cosine {probe["cos"][0]:.3f} to {probe["cos"][1]:.3f}; '
                      + (f'{100 * probe["over"]:.2f} % of elements' if probe["over"] >= 1e-4 else 'fewer than 1 element in 10,000')
                      + ' steps more than 1.5 times the rate. '
                      f'A synthetic test with gradient scales six orders of magnitude apart inside one 256-element block does make it over-step (13 to 27 times), but '
                      f'this model&#39;s gradients are not like that.</td></tr>')
    if rst:
        checks.append(f'<tr><td>Is 2e-5 too high for continuing a fitted model?</td><td>Yes. The epoch 4 → 8 run restarted the schedule; its training loss rose from '
                      f'{rst["start"]:.3f} to {rst["peak"]:.3f} while the rate was near 2e-5 and reached {rst["end"]:.3f} only after the decay. Checkpoints saved mid-schedule '
                      f'(epochs 1, 2, 3, 5, 6, 7) are snapshots at a high rate.</td></tr>')
    checks.append('<tr><td>What else is in the recipe?</td><td>One 4,096-token window per step with no gradient accumulation, AdamW betas 0.9 / 0.95, weight decay 0.1, '
                  'gradient clipping at norm 1.0 (active on about nine steps in ten, the norm is 2 to 6), a fresh optimizer for every run.</td></tr>')
    bullets = []
    if "v3" in got and "ep8" in got:
        a, b = got["v3"], got["ep8"]
        bullets.append(f'<li><b>The schedule and the optimizer do what they are configured to do; the learning rate is not mis-applied.</b> What the checkpoints show '
                       f'instead is memorisation: from v3 to epoch 8 the loss on the rows trained on falls from {a["train_new"][0]:.2f} to {b["train_new"][0]:.2f}, while '
                       f'the loss on oracle runs of the test tasks rises from {a["oracle_test"][0]:.2f} to {b["oracle_test"][0]:.2f} and the loss on earlier training rows '
                       f'that were not replayed doubles ({a["replay_unseen"][0]:.2f} to {b["replay_unseen"][0]:.2f}).</li>')
        if "r1" in got:
            r = got["r1"]
            bullets.append(f'<li><b>Held-out loss moves with the test score.</b> The two checkpoints with the lowest loss on test-task runs (v3 {a["oracle_test"][0]:.2f}, '
                           f'take-turns r1 {r["oracle_test"][0]:.2f}) are the two with the best test scores (8 and 9 of 22); every later checkpoint is worse on both. '
                           f'New runs of the training tasks gain only early (v3 {a["r2_unseen"][0]:.2f}, r1 {r["r2_unseen"][0]:.2f}) and are worse than v3 from epoch 4 on.</li>')
    if lerp:
        parts = []
        for t in lerp["order"]:
            bt = lerp["best"][t]
            if "oracle_test" in bt:
                parts.append(f'{ {"r1": "r1", "ep1": "epoch 1", "ep4": "epoch 4", "ep8": "epoch 8"}[t]}: lowest test-task loss at {bt["oracle_test"]:g} of the step, '
                             f'lowest unseen-run loss at {bt.get("r2_unseen", float("nan")):g}')
        bullets.append('<li><b>Step size along the direction each run moved.</b> ' + "; ".join(parts) + '. A value of 0 means v3 itself is best, 1 the checkpoint as trained.</li>')
    ref = None                                                       # the recipe's 2e-5 after 536 steps (epoch 1), on the same 35-row subset
    if os.path.exists(f"{KEEP}/lrval_lerp_ep1_a1.00.json"):
        ref = _groups(f"{KEEP}/lrval_lerp_ep1_a1.00.json")
    low = arms.get("lr5e-6_const") if arms else None
    if low and ref:
        s = min(low, key=lambda st: abs(st - 536))
        d = low[s]
        g4 = lambda src, key: " / ".join(f"{src[g][key] if isinstance(src[g], dict) else src[g][0]:.2f}" for g in GROUPS)  # noqa: E731
        better = all(d[g]["ce_final"] < ref[g][0] for g in ("oracle_test", "r2_unseen", "replay_unseen"))
        bullets.insert(1, f'<li><b>{"The rate itself is too high, not only the number of steps." if better else "A lower rate changes the held-out picture."}</b> '
                          f'A run from v3 at a constant 5e-6, a quarter of the recipe&#39;s peak, scores {g4(d, "ce_final")} after {s} steps on test tasks / unseen runs / '
                          f'earlier rows / rows trained on; the recipe at 2e-5 after 536 steps (epoch 1) scores {g4(ref, 0)} on the same rows. '
                          + (f'At a similar fit to the training rows the lower rate is better on all three held-out sets, and its test-task loss is below v3&#39;s '
                             f'({low[min(low)]["oracle_test"]["ce_final"]:.2f}) instead of above it.' if better and d["oracle_test"]["ce_final"] < low[min(low)]["oracle_test"]["ce_final"]
                             else '') + f' The run is at step {max(low)} of 1,072.</li>')
    elif arms:
        parts = []
        for tag, by in arms.items():
            lr = tag[2:].split("_")[0]
            s0 = by[min(by)]
            best = {g: min(by, key=lambda s: by[s][g]["ce_final"] if g in by[s] else 9e9) for g in ("oracle_test", "r2_unseen") if g in s0}
            parts.append(f'at {lr} the test-task loss is lowest at step {best.get("oracle_test")} ({by[best["oracle_test"]]["oracle_test"]["ce_final"]:.3f}, start '
                         f'{s0["oracle_test"]["ce_final"]:.3f}) and the unseen-run loss at step {best.get("r2_unseen")} '
                         f'({by[best["r2_unseen"]]["r2_unseen"]["ce_final"]:.3f}, start {s0["r2_unseen"]["ce_final"]:.3f}) of {max(by)} so far')
        bullets.append('<li><b>Learning rates compared on held-out loss during training.</b> ' + "; ".join(parts) + '.</li>')
    status = _chip("info", "running") if not arms or any(max(by) < 1072 for by in arms.values()) or len(arms) < 2 else _chip("good", "measured")
    html = f'''<div class="sub-head"><h3>Is the learning rate wrong? {status}</h3><p>The Terminal-Bench test score has not moved above the starting checkpoint with any amount of training, so the learning rate of the fine-tuning was checked from the mechanics up: is the configured rate the one applied, does the optimizer take the steps it should, and how does held-out loss respond to the rate. Terminal-Bench itself is a slow and noisy instrument for this (6 to 23 hours per checkpoint, 22 test trials), so the checkpoints and the new runs are compared on teacher-forced token loss of four fixed sets of trajectories: oracle runs of the 12 test tasks, runs of the training tasks that were not trained on, earlier training rows that the mix did not replay, and rows that were trained on.</p></div>
<div class="tablewrap">
    <ul>
      {chr(10).join("      " + b for b in bullets).strip()}
    </ul>
    <table>
      <thead><tr><th>Check</th><th>Result</th></tr></thead>
      <tbody>
        {chr(10).join("        " + c for c in checks).strip()}
      </tbody>
    </table>
    <p style="margin:18px 0 6px;font-weight:600">Held-out token loss of every checkpoint</p>
    <table>
      <thead><tr><th>Checkpoint</th><th>steps</th><th class="n">test set solved</th>{"".join(f'<th class="n">{HEAD[g]}</th>' for g in GROUPS)}</tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">Token loss (top-1 accuracy in %) on supervised tokens, last recurrence step, 4,096-token windows as in training: 12 oracle runs of test tasks (7,903 tokens), 30 unseen runs of training tasks (71,539), 20 earlier rows (67,985), 12 trained-on rows (25,560). Steps count from v3; a leading + counts from epoch 4. "Mid-schedule" checkpoints were saved while the rate was still high, "decayed" ones at the end of a schedule. Results: <code>{KEEP}</code>.</p>'''
    if lerp_html:
        html += f'''
    <p style="margin:18px 0 2px;font-weight:600">Was the step too large? Loss on the line from v3 to each checkpoint</p>
    {_legend()}
    {lerp_html}
    <p class="src">Each point is a model whose weights are v3 plus a share of the difference to the fine-tuned checkpoint (dashed line: the checkpoint itself; 1.5 extrapolates beyond it). A smaller share is, to first order, what a proportionally lower learning rate would have produced. Same four sets as above on a 35-row subset.</p>'''
    if curve_html:
        html += f'''
    <p style="margin:18px 0 2px;font-weight:600">Held-out loss during training at a constant learning rate</p>
    {_legend()}
    {curve_html}
    <p class="src">Runs from v3 on the same data and order as the 4-epoch run, 50 warm-up steps and then a constant rate, validated every 134 steps (a quarter epoch) inside the trainer on the 35-row subset; a constant rate makes every point of a run comparable with every other. The recipe&#39;s 2e-5 has no curve of its own yet; its reference points on the same rows are the "1" positions of the charts above (epoch 1 = 536 steps, r1 = 896 steps with decay, epoch 4 = 2,144 steps).</p>'''
    return html + '\n  </div>\n'
