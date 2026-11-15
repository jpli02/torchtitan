"""Page block added 2026-11-14: the router search on BFCL function calling (search set 575 of the 1,150 non-live items, real
decode-time early exit, compute counted as executed loops). Table of every evaluation, a Pareto view against the released gate,
and the held-out validation when it exists. Re-reads the result files on every build."""
import json
import os

import page_blocks_1114 as pb4
from page_blocks_1107 import TMP, _chip

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
EVALS = f"{WORK}/evaluations.jsonl"
HELD = f"{WORK}/heldout_validation.json"
TOL, PENALTY, N_DESIGN, N_BO = 1.7, 15.0, 6, 20  # as in boptim_lcb/bfcl_bo_run.py
GATES = {(1.0, 1.0, 1.0): "full depth", (0.7, 0.7, 0.7): "gate 0.7", (0.5, 0.5, 0.5): "gate 0.5", (0.3, 0.3, 0.3): "gate 0.3",
         (0.25, 0.25, 0.25): "gate 0.25", (0.2, 0.2, 0.2): "gate 0.2"}


def _evals():
    out, seen = [], set()
    if not os.path.exists(EVALS):
        return out
    for line in open(EVALS):
        try:
            e = json.loads(line)
        except Exception:
            continue
        key = tuple(round(v, 3) for v in e.get("thresholds", [])[:3])
        if e.get("status") != "ok" or key in seen:
            continue
        seen.add(key)
        e["key"] = key
        out.append(e)
    return out


def _fmt(key):
    return f"({key[0]:.2f}, {key[1]:.2f}, {key[2]:.2f})"


def bfcl_bo_block():
    evs = _evals()
    log = f"{TMP}/bfcl_bo.log"
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    finished = "BO finished with code 0" in text
    full = next((e for e in evs if e["key"] == (1.0, 1.0, 1.0)), None)
    floor = full["acc_pct"] - TOL if full else None

    def val(e):
        return 100.0 * e["exec_loops"] / 4.0 + PENALTY * max(0.0, (floor if floor is not None else -1e9) - e["acc_pct"])

    base = [e for e in evs if e["key"] in GATES]
    props = [e for e in evs if e["key"] not in GATES]
    for k, e in enumerate(props, 1):
        e["id"] = k
    # the claim is "baseline accuracy at less compute": among routers that hold the floor, the fewest executed loops wins
    holds = (lambda e: floor is not None and e["acc_pct"] >= floor)
    best_base = min((e for e in base if e is not full and holds(e)), key=lambda e: e["exec_loops"], default=None)
    best_prop = min((e for e in props if holds(e)), key=lambda e: e["exec_loops"], default=None)
    rows = []
    for e in sorted(base, key=lambda e: -e["key"][0]) + props:
        name = f'<b>{"released " if e is not full else ""}{GATES[e["key"]]}</b>' if e["key"] in GATES else f'BO #{e["id"]} {_fmt(e["key"])}'
        below = floor is not None and e["acc_pct"] < floor
        cls = ' class="best"' if e is best_prop and best_base is not None and e["exec_loops"] < best_base["exec_loops"] else ""
        rows.append(f'<tr{cls}><td>{name}</td><td class="n">{e["acc_pct"]:.1f}{" ↓" if below else ""}</td><td class="n">{e.get("overall_unweighted_pct", 0):.1f}</td>'
                    f'<td class="n">{e["exec_loops"]:.2f}</td><td class="n">{100 * e["exec_loops"] / 4:.0f} %</td><td class="n">{4 / e["exec_loops"]:.2f}×</td>'
                    f'<td class="n">{val(e):.1f}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="7">the full-depth baseline is running</td></tr>')
    done_b, done_p = len(base), len(props)
    status = f'{done_b} of {N_DESIGN} baseline points and {done_p} of {N_BO} proposals evaluated'
    if full:
        status += f'; full depth scores {full["acc_pct"]:.1f} % on the search set, so the floor is {floor:.1f} %'
    if best_base:
        status += (f'; the cheapest released-gate point that holds the floor so far is {GATES[best_base["key"]]}: {best_base["acc_pct"]:.1f} % at '
                   f'{best_base["exec_loops"]:.2f} loops')
    if best_prop:
        status += (f'; the cheapest proposal that holds the floor so far is #{best_prop["id"]} {_fmt(best_prop["key"])}: {best_prop["acc_pct"]:.1f} % at '
                   f'{best_prop["exec_loops"]:.2f} loops')
        if best_base:
            d = 100 * (1 - best_prop["exec_loops"] / best_base["exec_loops"])
            status += f' ({abs(d):.0f} % {"fewer" if d > 0 else "more"} loops than that gate point, on the search set)'
    elif props:
        status += '; no proposal holds the floor yet'
    chip = _chip("good", "finished") if finished else _chip("info", "running")

    chart = ""
    if full and len(evs) >= 4:
        ymin = floor - 6
        def lab(e):
            """Direct label above the square, or below and to the left when the square sits on the dashed floor line."""
            return (-8, 16, "end") if floor - 0.9 < e["acc_pct"] < floor + 0.3 else (0, -10, "middle")

        pts_b = [{"x": 4 / e["exec_loops"], "y": e["acc_pct"], "name": GATES[e["key"]], "lab": lab(e),
                  "tip": f'{GATES[e["key"]]}: {e["acc_pct"]:.1f} % at {e["exec_loops"]:.2f} loops ({4 / e["exec_loops"]:.2f}× less decode compute)'}
                 for e in sorted(base, key=lambda e: -e["key"][0]) if e["acc_pct"] >= ymin]
        # proposals with identical outcomes (thresholds that no token falls between) are drawn as one point
        groups = {}
        for e in props:
            if e["acc_pct"] >= ymin:
                groups.setdefault((round(e["exec_loops"], 4), round(e["acc_pct"], 2)), []).append(e)
        pts_p, best_pid = [], None
        for g in groups.values():
            pid = ", ".join(str(e["id"]) for e in g)
            if best_prop is not None and any(e is best_prop for e in g):
                best_pid = pid
            pts_p.append({"x": 4 / g[0]["exec_loops"], "y": g[0]["acc_pct"], "id": pid,
                          "tip": "; ".join(f'#{e["id"]} {_fmt(e["key"])}' for e in g)
                                 + f': {g[0]["acc_pct"]:.1f} % at {g[0]["exec_loops"]:.2f} loops ({4 / g[0]["exec_loops"]:.2f}×), objective {val(g[0]):.1f}'})
        # proposal labels go below and to the right of their circle; the gate labels sit above their squares
        label_dx = {p["id"]: (9, 15, "start") for p in pts_p}
        hidden = sum(1 for e in evs if e["acc_pct"] < ymin)
        g5, g3 = next((e for e in base if e["key"] == (0.5, 0.5, 0.5)), None), next((e for e in base if e["key"] == (0.3, 0.3, 0.3)), None)
        noise_note = ""
        if g5 and g3 and g5["acc_pct"] < g3["acc_pct"]:
            noise_note = (f'. The curve is not smooth: gate 0.5 scores below gate 0.3 by {round((g3["acc_pct"] - g5["acc_pct"]) * 5.75)} items of 575, '
                          'which is the size of the sampling noise a single point carries here')
        xs = [p["x"] for p in pts_b + pts_p]
        ys = [p["y"] for p in pts_b + pts_p]
        x1 = max(2.0, 0.5 * (int(max(xs) / 0.5) + 1))
        y0, y1 = int(min(min(ys), floor) - 1), int(max(ys) + 2)
        xt = [1 + 0.5 * i for i in range(int((x1 - 1) / 0.5) + 1)]
        yt = list(range(y0, y1 + 1, 1 if y1 - y0 <= 8 else 2))
        svg = pb4._svg((0.9, x1), (y0, y1), xt, yt, pts_p, pts_b, [], round(floor, 2), best_pid, label_dx,
                       "Pareto view of the BFCL router search: accuracy against decode compute saved, proposals and the released-gate curve",
                       (pb4.ML + 22, pb4.H - pb4.MB - 78), xlabel="decode compute saved: 4 / executed loops (higher is cheaper)", ylabel="accuracy on the 575 search items, %",
                       floor_right=True, best_label="cheapest proposal that holds the floor")
        chart = (f'<p style="margin:18px 0 6px;font-weight:600">Accuracy against decode compute, search set</p>\n    {svg}\n'
                 f'    <p class="src">Squares: full depth and the released gate at each threshold, joined in threshold order. Circles: the optimiser&#39;s proposals '
                 f'(hover for thresholds). A proposal beats the gate only if it lies above and to the right of the gate curve{noise_note}'
                 + (f'; {hidden} point{"s" if hidden != 1 else ""} below {ymin:.0f} % not drawn' if hidden else "") + '.</p>')

    held = ""
    if os.path.exists(HELD):
        try:
            h = json.load(open(HELD))
        except Exception:
            h = None
        if h and h.get("rows"):
            hr = []
            for r in h["rows"]:
                cells = "".join(f'<td class="n">{r[k]:.1f}</td><td class="n">{r[k + "_loops"]:.2f}</td>' if r.get(k) is not None else '<td class="n">…</td><td class="n">…</td>'
                                for k in ("search", "heldout", "live"))
                hr.append(f'<tr><td>{r["name"]}</td>{cells}</tr>')
            held = f'''<p style="margin:18px 0 6px;font-weight:600">Held-out validation: the 575 unseen non-live items and the 1,350 live items</p>
    <table>
      <thead><tr><th>router</th><th class="n">search %</th><th class="n">loops</th><th class="n">held-out %</th><th class="n">loops</th><th class="n">live %</th><th class="n">loops</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in hr).strip()}
      </tbody>
    </table>
    <p class="src">{h.get("note", "")}</p>'''

    return f'''<div class="sub-head"><h3>BO on BFCL function calling, the benchmark where depth matters {chip}</h3><p>The third search, on the benchmark the probes singled out (ICML 2025; depth 1 / 2 / 4 score 0 / 51 / 78 %, and real early exit gives a smooth curve, not a cliff). It applies what the two LiveCodeBench runs taught. <b>Search and held-out sets are disjoint</b>: the optimiser sees 575 of the 1,150 non-live items (every second item of each category); the other 575 and the 1,350 live items are kept for validation. <b>Compute is counted, not timed</b>: the model reports the loops it executed for every decoded token, which is deterministic, where tokens per second on this shared machine moved by 10 to 30 % between identical runs. <b>The released gate is measured first</b> on the same items (full depth, then thresholds 0.7, 0.5, 0.3, 0.25, 0.2) and seeds the optimiser. Objective, minimised: 100 × executed loops / 4, plus 15 for every accuracy point below the floor (full depth minus 1.7 points, one standard error at this size); full depth scores 100 and a router that halves decode compute for free scores 50. Along the gate&#39;s own curve one accuracy point buys about 0.14 loops; the penalty lets a router give up a point only for more than 0.6 loops, so the optimum sits at the floor. Every evaluation runs real early exit at batch 1 and is graded by the official AST checker in a sandbox; accuracy is the share of items passed. <b>The search box comes from the gate curve</b>: accuracy holds at threshold 0.3, slips at 0.25 and collapses at 0.2, and the mean can go below two loops only if tokens may leave after loop 1, so the optimiser searches q1 in 0.2 to 0.6 and q2, q3 in 0.05 to 0.5. Optimiser: gp_hedge, two proposals per iteration, ten iterations (four grid points, then sixteen model-based proposals); the four gate points inside the box seed it.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>router (q1, q2, q3)</th><th class="n">accuracy % (575)</th><th class="n">leaderboard-style %</th><th class="n">executed loops</th><th class="n">decode compute</th><th class="n">compute saved</th><th class="n">objective</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">↓ marks accuracy below the floor. "Leaderboard-style" is BFCL&#39;s unweighted mean over its four categories. Status: {status}. Trajectories: <code>{EVALS}</code>.</p>
    {chart}
    {held}
  </div>
'''
