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
TOL, PENALTY, N_DESIGN, N_BO = 1.7, 15.0, 6, 20  # as in boptim_lcb/bfcl_bo_run.py; N_BO proposals per search
GATE = f"{WORK}/gate_probs.jsonl"
GATES = {(1.0, 1.0, 1.0): "full depth", (0.7, 0.7, 0.7): "gate 0.7", (0.5, 0.5, 0.5): "gate 0.5", (0.3, 0.3, 0.3): "gate 0.3",
         (0.25, 0.25, 0.25): "gate 0.25", (0.2, 0.2, 0.2): "gate 0.2"}
_GP = []


def _evals():
    out, seen = [], set()
    if not os.path.exists(EVALS):
        return out
    for line in open(EVALS):
        try:
            e = json.loads(line)
        except Exception:
            continue
        key = tuple(round(v, 4) for v in e.get("thresholds", [])[:3])  # four decimals: near 0.25 the third one already matters
        if e.get("status") != "ok" or key in seen:
            continue
        seen.add(key)
        e["key"] = key
        out.append(e)
    return out


def _fmt(key):
    return f"({key[0]:.3f}, {key[1]:.3f}, {key[2]:.3f})"


def _gate():
    """[tokens, 3] cumulative exit probabilities after loops 1, 2, 3 for every answer token of the search set (bfcl_gate_dump.py)."""
    if not _GP and os.path.exists(GATE):
        import numpy as np
        rows = []
        for line in open(GATE):
            rows.extend(json.loads(line)["P"])
        _GP.append(np.asarray(rows, float))
    return _GP[0] if _GP else None


def _exit_shares(thr):
    """Share of the answer tokens that leave after loops 1, 2, 3, 4 under thresholds (q1, q2, q3), from the gate statistics."""
    P = _gate()
    if P is None:
        return None
    e1 = P[:, 0] >= thr[0]
    e2 = ~e1 & (P[:, 1] >= thr[1])
    e3 = ~e1 & ~e2 & (P[:, 2] >= thr[2])
    s = [float(e1.mean()), float(e2.mean()), float(e3.mean())]
    return s + [max(0.0, 1.0 - sum(s))]


def _unseen_reading(rows, key, n_items, label):
    """What an unseen set says about the searched routers: each against the released gate's own curve at the same executed loops
    (straight line between the two gate points around it), in items. Empty until the set has full depth, two gate points and a router."""
    have = [r for r in rows if r.get(key) is not None]
    full = next((r for r in have if r["name"] == "full depth"), None)
    gates = sorted((r for r in have if r["name"].startswith("released gate")), key=lambda r: r[key + "_loops"])
    routers = [r for r in have if r["name"].startswith("BO #")]
    if not full or len(gates) < 2 or not routers:
        return ""
    floor = full[key] - TOL
    curve = gates + [full]
    parts, diffs = [], []
    for r in routers:
        L = r[key + "_loops"]
        lo = max((g for g in curve if g[key + "_loops"] <= L), key=lambda g: g[key + "_loops"], default=None)
        hi = min((g for g in curve if g[key + "_loops"] >= L), key=lambda g: g[key + "_loops"], default=None)
        if lo is None or hi is None:
            continue
        span = hi[key + "_loops"] - lo[key + "_loops"]
        line = lo[key] if span <= 0 else lo[key] + (L - lo[key + "_loops"]) / span * (hi[key] - lo[key])
        d = round((r[key] - line) * n_items / 100.0)
        diffs.append(d)
        parts.append(f'{r["name"].split(" (")[0]} {r[key]:.1f} % at {L:.2f} loops, {abs(d)} item{"s" if abs(d) != 1 else ""} {"above" if d > 0 else "below"} the gate curve' if d else
                     f'{r["name"].split(" (")[0]} {r[key]:.1f} % at {L:.2f} loops, on the gate curve')
    if not parts:
        return ""
    ok_gates = [g for g in gates if g[key] >= floor]
    ok_routers = [r for r in routers if r[key] >= floor]
    cheapest = min(ok_gates + ok_routers, key=lambda r: r[key + "_loops"], default=None)
    noise = max(2, round(0.006 * n_items))  # about half a point: what a paired difference between two close routers moves by chance
    verdict = ("The advantage seen on the search set did not carry over: on these unseen items the searched routers sit on the released gate&#39;s own curve."
               if all(d <= noise for d in diffs) else "Some of the advantage seen on the search set carries over to these unseen items.")
    return (f'<p class="src"><b>Result on {label}.</b> Full depth scores {full[key]:.1f} %, which puts the floor at {floor:.1f} %. ' + "; ".join(parts) + ". "
            + (f'The cheapest router that holds this floor is {cheapest["name"].split(" (")[0]} ({cheapest[key]:.1f} % at {cheapest[key + "_loops"]:.2f} loops). ' if cheapest else "")
            + verdict + '</p>')


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
    # same tie-break as the validation driver: loops to three decimals, then accuracy, then order of evaluation
    best_prop = min((e for e in props if holds(e)), key=lambda e: (round(e["exec_loops"], 3), -e["acc_pct"]), default=None)
    log2 = f"{TMP}/bfcl_bo2.log"
    text2 = open(log2, errors="replace").read() if os.path.exists(log2) else ""
    # the second search (exit shares) only shows once it has results of its own; it was stopped before its first evaluation
    # on 2026-11-15, after the held-out result of the first search
    started2, finished2 = len(props) > N_BO, "BO2 finished with code 0" in text2
    rows = []
    for e in sorted(base, key=lambda e: -e["key"][0]) + props:
        if e.get("id") == N_BO + 1:
            rows.append('<tr class="dim"><td colspan="7">second search, in exit shares (thresholds placed between the gate&#39;s discrete values)</td></tr>')
        name = f'<b>{"released " if e is not full else ""}{GATES[e["key"]]}</b>' if e["key"] in GATES else f'BO #{e["id"]} {_fmt(e["key"])}'
        below = floor is not None and e["acc_pct"] < floor
        cls = ' class="best"' if e is best_prop and best_base is not None and e["exec_loops"] < best_base["exec_loops"] else ""
        sh = _exit_shares(e["thresholds"][:3])
        shares = " / ".join(f"{100 * v:.0f}" for v in sh) if sh else "—"
        rows.append(f'<tr{cls}><td>{name}</td><td class="n">{e["acc_pct"]:.1f}{" ↓" if below else ""}</td><td class="n">{e.get("overall_unweighted_pct", 0):.1f}</td>'
                    f'<td class="n">{e["exec_loops"]:.2f}</td><td class="n">{4 / e["exec_loops"]:.2f}×</td><td class="n">{shares}</td>'
                    f'<td class="n">{val(e):.1f}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="7">the full-depth baseline is running</td></tr>')
    done_b, done_p = len(base), len(props)
    status = f'{done_b} of {N_DESIGN} baseline points and {min(done_p, N_BO)} of {N_BO} proposals evaluated'
    if started2:
        status += f' in the first search, {max(0, done_p - N_BO)} of {N_BO} in the second'
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
    chip = (_chip("good", "finished") if finished2 else _chip("info", "second search running")) if started2 else (
        _chip("good", "search finished") if finished else _chip("info", "running"))
    second = ""
    if finished and started2:
        second = ('<p class="src"><b>Second search, in exit shares.</b> The gate statistics below show why a cube of thresholds is a poor search space: most '
                  'of it changes nothing, and inside the narrow band where loop-1 exits are decided the gate offers only a few discrete settings. The second '
                  'search therefore works in the shares themselves: f1, the share of all answer tokens released after loop 1 (0 to 25 %); f2, the share of '
                  'the remaining tokens released after loop 2 (70 to 100 %); f3, the share of the still remaining tokens released after loop 3 (50 to 100 %, '
                  'the rest run all four loops, which no router of the first search allowed). Each proposal is turned into thresholds placed midway between '
                  'two neighbouring values of the gate. Same objective, same search set; every router evaluated so far seeds the optimiser. '
                  + ("It runs after the held-out part of the validation." if not started2 else ("It is finished." if finished2 else "It is running.")) + '</p>')
    reading = ""
    if finished and best_prop and best_base:
        d = 100 * (1 - best_prop["exec_loops"] / best_base["exec_loops"])
        items = round((best_prop["acc_pct"] - best_base["acc_pct"]) * 5.75)
        n_hold = sum(1 for e in props if holds(e))
        n_cheaper = sum(1 for e in props if holds(e) and e["exec_loops"] < best_base["exec_loops"])
        reading = (f'<p class="src"><b>Reading of the search.</b> {n_hold} of the {len(props)} proposals hold the accuracy floor and {n_cheaper} of those need fewer '
                   f'loops than {GATES[best_base["key"]]}, the gate&#39;s cheapest point above the floor. The cheapest, #{best_prop["id"]} {_fmt(best_prop["key"])}, runs '
                   f'{best_prop["exec_loops"]:.2f} loops against {best_base["exec_loops"]:.2f}: {abs(d):.0f} % {"fewer" if d > 0 else "more"} loops, with '
                   f'{abs(items)} item{"s" if abs(items) != 1 else ""} of 575 {"fewer" if items < 0 else "more"} answered correctly. That gap in accuracy is inside the '
                   f'noise the gate curve itself shows, so the search set cannot tell a better router from a fit to these 575 items; the held-out table decides.</p>')

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
        pts_p.sort(key=lambda p: p["id"] == best_pid)  # the best point is drawn last, on top of its neighbours
        xl, yl = "decode compute saved: 4 / executed loops (higher is cheaper)", "accuracy on the 575 search items, %"
        svg = pb4._svg((0.9, x1), (y0, y1), xt, yt, pts_p, pts_b, [], round(floor, 2), best_pid, label_dx,
                       "Pareto view of the BFCL router search: accuracy against decode compute saved, proposals and the released-gate curve",
                       (pb4.ML + 22, pb4.H - pb4.MB - 78), xlabel=xl, ylabel=yl, floor_right=True, best_label="cheapest proposal that holds the floor",
                       label_props=len(pts_p) <= 4)
        chart = (f'<p style="margin:18px 0 6px;font-weight:600">Accuracy against decode compute, search set</p>\n    {svg}\n'
                 f'    <p class="src">Squares: full depth and the released gate at each threshold, joined in threshold order. Circles: the optimiser&#39;s proposals '
                 f'(hover for thresholds). A proposal beats the gate only if it lies above and to the right of the gate curve{noise_note}'
                 + (f'; {hidden} point{"s" if hidden != 1 else ""} below {ymin:.0f} % not drawn' if hidden else "") + '.</p>')
        # zoom on the region the search works in: the proposals near the floor and the gate points next to them
        # window: from just left of the gate's cheapest point above the floor to just right of the next cheaper gate point
        near = []
        if best_base is not None:
            gx = 4 / best_base["exec_loops"]
            nxt = min((4 / e["exec_loops"] for e in base if 4 / e["exec_loops"] > gx + 1e-9), default=gx + 0.2)
            zx0, zx1 = round(0.05 * int((gx - 0.08) / 0.05), 2), round(0.05 * (int((nxt + 0.04) / 0.05) + 1), 2)
            near = [p for p in pts_p if zx0 <= p["x"] <= zx1 and p["y"] >= floor - 1.5]
        if len(near) >= 3:
            zb = [p for p in pts_b if zx0 <= p["x"] <= zx1 and p["y"] >= floor - 1.5]
            zys = [p["y"] for p in near + zb]
            zy0, zy1 = int(min(min(zys), floor)), int(max(zys)) + 1
            zxt = [round(zx0 + 0.05 * i, 2) for i in range(int(round((zx1 - zx0) / 0.05)) + 1)]
            zyt = [zy0 + 0.5 * i for i in range(int((zy1 - zy0) / 0.5) + 1)]
            zoom = pb4._svg((zx0, zx1), (zy0, zy1), zxt, zyt, [p for p in near if zx0 <= p["x"] <= zx1], zb, [], round(floor, 2), best_pid, label_dx,
                            "Zoom of the BFCL router search around the accuracy floor: proposals against the nearest released-gate points",
                            (pb4.W - pb4.MR - 238, pb4.MT + 16), xlabel=xl, ylabel=yl, floor_right=False, best_label="cheapest proposal that holds the floor")
            chart += (f'\n    <p style="margin:18px 0 6px;font-weight:600">The same, zoomed on the region around the floor</p>\n    {zoom}\n'
                      f'    <p class="src">Numbers next to the circles are the proposal numbers of the table; proposals with identical outcomes share one circle. '
                      f'One accuracy point on this axis is 5.75 items.</p>')

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
    <p class="src">{h.get("note", "")}</p>
    {_unseen_reading(h["rows"], "heldout", 575, "the 575 unseen non-live items")}
    {_unseen_reading(h["rows"], "live", 1350, "the 1,350 live items")}'''

    return f'''<div class="sub-head"><h3>BO on BFCL function calling, the benchmark where depth matters {chip}</h3><p>The third search, on the benchmark the probes singled out (ICML 2025; depth 1 / 2 / 4 score 0 / 51 / 78 %, and real early exit gives a smooth curve, not a cliff). It applies what the two LiveCodeBench runs taught. <b>Search and held-out sets are disjoint</b>: the optimiser sees 575 of the 1,150 non-live items (every second item of each category); the other 575 and the 1,350 live items are kept for validation. <b>Compute is counted, not timed</b>: the model reports the loops it executed for every decoded token, which is deterministic, where tokens per second on this shared machine moved by 10 to 30 % between identical runs. <b>The released gate is measured first</b> on the same items (full depth, then thresholds 0.7, 0.5, 0.3, 0.25, 0.2) and seeds the optimiser. Objective, minimised: 100 × executed loops / 4, plus 15 for every accuracy point below the floor (full depth minus 1.7 points, one standard error at this size); full depth scores 100 and a router that halves decode compute for free scores 50. Along the gate&#39;s own curve one accuracy point buys about 0.14 loops; the penalty lets a router give up a point only for more than 0.6 loops, so the optimum sits at the floor. Every evaluation runs real early exit at batch 1 and is graded by the official AST checker in a sandbox; accuracy is the share of items passed. <b>The search box comes from the gate curve</b>: accuracy holds at threshold 0.3, slips at 0.25 and collapses at 0.2, and the mean can go below two loops only if tokens may leave after loop 1, so the optimiser searches q1 in 0.2 to 0.6 and q2, q3 in 0.05 to 0.5. Optimiser: gp_hedge, two proposals per iteration, ten iterations (four grid points, then sixteen model-based proposals); the four gate points inside the box seed it.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>router (q1, q2, q3)</th><th class="n">accuracy % (575)</th><th class="n">leaderboard-style %</th><th class="n">executed loops</th><th class="n">compute saved</th><th class="n">tokens leaving after loop 1 / 2 / 3 / 4, %</th><th class="n">objective</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">↓ marks accuracy below the floor. "Leaderboard-style" is BFCL&#39;s unweighted mean over its four categories. The exit shares are computed from the gate statistics below, not measured in the run; they reproduce the measured loops within 0.03. Status: {status}. Trajectories: <code>{EVALS}</code>.</p>
    {reading}
    {second}
    {chart}
    {held}
  </div>
'''


def _fill_sentence(evs):
    """What the exit-lab runs add to the answer bullet: the cost below two loops is the cache of early leavers; the correction."""
    plain = {e["key"]: e for e in evs if e["key"] in GATES}
    corr = {e["key"]: e for e in _load_ok(f"{WORK}/lab_search/evaluations.jsonl")
            if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf")}
    k = (0.2, 0.2, 0.2)
    if k not in plain or k not in corr:
        return " Finer thresholds on this gate are unlikely to help."
    s = (f' Below two loops the cost is not which tokens leave but the cache entries they leave behind for the tokens that keep going: correcting those entries '
         f'lifts the gate at 0.2 from {plain[k]["acc_pct"]:.1f} to {corr[k]["acc_pct"]:.1f} % on the search items at the same {corr[k]["exec_loops"]:.2f} loops')
    hp = {e["key"]: e for e in _load_ok(f"{WORK}/heldout/evaluations.jsonl") if not e.get("extra_cfg")}
    hc = {e["key"]: e for e in _load_ok(f"{WORK}/lab_heldout/evaluations.jsonl")
          if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf")}
    if k in hp and k in hc:
        s += f', and from {hp[k]["acc_pct"]:.1f} to {hc[k]["acc_pct"]:.1f} % on unseen items'
    return s + ". Exiting on the head&#39;s confidence in place of the gate does not help."


def _fill_answer_bullet():
    """The positive result, when it exists: a searched router with corrected cache entries that holds the held-out floor with
    fewer loops than every plain released-gate point that holds it."""
    hp = [e for e in _load_ok(f"{WORK}/heldout/evaluations.jsonl") if not e.get("extra_cfg") and e["key"] in GATES]
    design = {(0.3, 0.3, 0.3), (0.25, 0.25, 0.25), (0.2443, 0.2443, 0.2443), (0.2, 0.2, 0.2)}
    hc = [e for e in _load_ok(f"{WORK}/lab_heldout/evaluations.jsonl")
          if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf") and e["key"] not in design]
    full = next((e for e in hp if e["key"] == (1.0, 1.0, 1.0)), None)
    if not full or not hc:
        return ""
    floor = full["acc_pct"] - TOL
    for e in hc:
        e["eff"] = e["exec_loops"] + _fill_overhead(e["thresholds"][:3])
    ok = [e for e in hc if e["acc_pct"] >= floor]
    gp = [g for g in hp if g["key"] != (1.0, 1.0, 1.0) and g["acc_pct"] >= floor]
    if not ok or not gp:
        return ""
    best, gbest = min(ok, key=lambda e: e["eff"]), min(gp, key=lambda g: g["exec_loops"])
    if best["eff"] >= gbest["exec_loops"]:
        return ""
    same = min((g for g in gp if g["acc_pct"] >= best["acc_pct"] - 0.05), key=lambda g: g["exec_loops"], default=None)
    s = (f'\n      <li><b>On BFCL a searched router keeps baseline accuracy at {best["eff"]:.2f} of 4 loops on unseen items, once early exits leave corrected cache '
         f'entries behind.</b> On the 575 held-out items it scores {best["acc_pct"]:.1f} % (full depth {full["acc_pct"]:.1f} %, floor {floor:.1f} %) at '
         f'{4 / best["eff"]:.2f}× less decode compute than full depth; the plain released gate needs {gbest["exec_loops"]:.2f} loops to hold that floor')
    if same is not None:
        s += f' and {same["exec_loops"]:.2f} loops for the same accuracy ({same["acc_pct"]:.1f} % at threshold {same["key"][0]:g})'
    import math
    n_ok, n_all = len(ok), len(hc)
    need = math.ceil(floor * 5.75 - 1e-9)
    short = sorted(need - round(e["acc_pct"] * 5.75) for e in hc if e["acc_pct"] < floor)
    miss = ("" if not short else f'; the other{"s miss" if len(short) > 1 else " misses"} it by '
            + (f'{short[0]} item{"s" if short[0] != 1 else ""}' if len(set(short)) == 1 else f'{short[0]} to {short[-1]} items'))
    s += (f'. {n_ok} of the {n_all} searched routers tested on unseen items hold{"s" if n_ok == 1 else ""} the floor{miss}. Two things were needed together: a linear correction of the cache '
          f'entries a token leaves behind when it exits early, fitted once on full-depth answers, and a search over how many tokens leave after each loop.</li>')
    return s


def bfcl_answer_bullet():
    """One answer bullet for the router question, written from the search and held-out results (empty until both exist)."""
    evs = _evals()
    full = next((e for e in evs if e["key"] == (1.0, 1.0, 1.0)), None)
    if not full or not os.path.exists(HELD):
        return ""
    try:
        rows = json.load(open(HELD))["rows"]
    except Exception:
        return ""
    floor = full["acc_pct"] - TOL
    gates = [e for e in evs if e["key"] in GATES and e is not full and e["acc_pct"] >= floor]
    props = [e for e in evs if e["key"] not in GATES and e["acc_pct"] >= floor]
    hfull = next((r for r in rows if r["name"] == "full depth" and r.get("heldout") is not None), None)
    hrouters = [r for r in rows if r["name"].startswith("BO #") and r.get("heldout") is not None]
    hgates = [r for r in rows if r["name"].startswith("released gate") and r.get("heldout") is not None]
    if not gates or not props or not hfull or not hrouters or not hgates:
        return ""
    g = min(gates, key=lambda e: e["exec_loops"])
    p = min(props, key=lambda e: (round(e["exec_loops"], 3), -e["acc_pct"]))
    hfloor = hfull["heldout"] - TOL
    hg = min((r for r in hgates if r["heldout"] >= hfloor), key=lambda r: r["heldout_loops"], default=None)
    hp = min((r for r in hrouters if r["heldout"] >= hfloor), key=lambda r: r["heldout_loops"], default=None)
    if hg is None:
        return ""
    wins = hp is not None and hp["heldout_loops"] < hg["heldout_loops"]
    head = ("On BFCL a searched router keeps its advantage on unseen items." if wins else
            "Searching thresholds alone does not beat the released gate on BFCL.")
    unseen = (f'on 575 unseen items the cheapest router above the floor is {"the searched " + hp["name"].split(" (")[0] if wins else hg["name"]} '
              f'({(hp if wins else hg)["heldout"]:.1f} % at {(hp if wins else hg)["heldout_loops"]:.2f} loops'
              + ("" if wins or hp is None else f'; the best searched router needs {hp["heldout_loops"]:.2f} loops for {hp["heldout"]:.1f} %') + ')')
    return (_fill_answer_bullet() + f'\n      <li><b>{head}</b> Function calling is the benchmark where depth matters most (depth 1 / 2 / 4: 0 / 51 / 78 %), and the released gate already '
            f'keeps full-depth accuracy at {g["exec_loops"]:.2f} of 4 loops. On the 575 search items the best searched router held the accuracy floor at '
            f'{p["exec_loops"]:.2f} loops; {unseen}. The reason is in the gate: its cumulative exit probabilities sit near 0.25, 0.5 and 0.75 and take only a few '
            f'distinct values, so a threshold chooses how many tokens leave after each loop, not which ones.{_fill_sentence(evs)}</li>')


# ---------------------------------------------------------------- the gate's exit probabilities on BFCL answers
def _hist_svg(P):
    """Three stacked histograms (after loops 1, 2, 3) of the cumulative exit probability, one shared x axis and one y scale.
    One hue: each panel is a single series and its heading names it. Bars are thin with a surface gap; the grid is hairline."""
    import numpy as np
    W, ML, MR, MT, PH, GAP, MB = 760, 58, 26, 30, 104, 40, 46
    H = MT + 3 * PH + 2 * GAP + MB
    pw = W - ML - MR
    n = len(P)
    edges = np.linspace(0.0, 1.0, 101)
    hists = [np.histogram(P[:, s], bins=edges)[0] / n * 100.0 for s in range(3)]
    ymax = 10.0 * int(max(h.max() for h in hists) / 10.0 + 1)
    yt = [0, ymax / 2, ymax]
    bw = pw / 100.0
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Histograms of the cumulative exit probability of BFCL answer tokens after loops 1, 2 and 3: '
         f'each piles up just below a quarter, a half and three quarters" style="width:100%;height:auto;max-width:{W}px;display:block;font-family:inherit">']
    for s in range(3):
        top = MT + s * (PH + GAP)
        base = top + PH

        def Y(v, base=base):
            return base - v / ymax * PH

        for v in yt:
            o.append(f'<line x1="{ML}" x2="{W - MR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" style="stroke:var(--line);stroke-width:1"/>')
            o.append(f'<text x="{ML - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" style="fill:var(--text-2);font-size:12px">{v:g}</text>')
        for xv in (0.25, 0.5, 0.75):
            o.append(f'<line x1="{ML + xv * pw:.1f}" x2="{ML + xv * pw:.1f}" y1="{top}" y2="{base}" style="stroke:var(--line);stroke-width:1"/>')
        o.append(f'<line x1="{ML}" x2="{W - MR}" y1="{base}" y2="{base}" style="stroke:var(--line-2);stroke-width:1.2"/>')
        h = hists[s]
        for i, v in enumerate(h):
            x = ML + i * bw
            if v > 0:
                hh = max(1.0, v / ymax * PH)
                r = min(2.2, hh)
                x0, x1, y0 = x + 1.0, x + bw - 1.0, base - hh
                o.append(f'<path d="M{x0:.1f},{base} L{x0:.1f},{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} L{x1 - r:.1f},{y0:.1f} '
                         f'Q{x1:.1f},{y0:.1f} {x1:.1f},{y0 + r:.1f} L{x1:.1f},{base} Z" style="fill:var(--s1)"/>')
            # hover target: the whole column of the bin, wider and taller than the bar
            o.append(f'<g><title>after loop {s + 1}: {v:.1f} % of the answer tokens have a cumulative exit probability between {edges[i]:.2f} and {edges[i + 1]:.2f}'
                     f'</title><rect x="{x:.1f}" y="{top}" width="{bw:.1f}" height="{PH}" style="fill:transparent"/></g>')
        # heading and one direct label: where the mass sits
        k = int(np.argmax(np.convolve(h, np.ones(3), mode="valid")))
        mass, lo, hi = h[k:k + 3].sum(), edges[k], edges[k + 3]
        o.append(f'<text x="{ML + 6}" y="{top - 9}" style="fill:var(--text);font-size:12.5px;font-weight:600">after loop {s + 1}</text>')
        peak_x = ML + (k + 1.5) * bw
        right = peak_x < ML + 0.55 * pw
        o.append(f'<text x="{peak_x + (16 if right else -16):.1f}" y="{top + 16}" text-anchor="{"start" if right else "end"}" '
                 f'style="fill:var(--text-2);font-size:12px">{mass:.0f} % of the tokens between {lo:.2f} and {hi:.2f}</text>')
    for xv in (0, 0.25, 0.5, 0.75, 1):
        o.append(f'<text x="{ML + xv * pw:.1f}" y="{H - MB + 17}" text-anchor="middle" style="fill:var(--text-2);font-size:12px">{xv:g}</text>')
    o.append(f'<text x="{ML + pw / 2:.0f}" y="{H - 10}" text-anchor="middle" style="fill:var(--text-2);font-size:12.5px">cumulative exit probability, the number a threshold is compared with</text>')
    o.append(f'<text transform="translate(15 {MT + (3 * PH + 2 * GAP) / 2:.0f}) rotate(-90)" text-anchor="middle" style="fill:var(--text-2);font-size:12.5px">share of the answer tokens per 0.01, %</text>')
    o.append("</svg>")
    return "\n".join(o)


def bfcl_gate_block():
    P = _gate()
    if P is None:
        return ""
    import numpy as np
    n = len(P)
    mx = [float(P[:, s].max()) for s in range(3)]
    # the discrete loop-1 values around the spike: value, tokens at it, tokens at or above it
    u, c = np.unique(np.round(P[:, 0], 4), return_counts=True)
    tail = np.cumsum(c[::-1])[::-1] / n
    levels = [(float(u[i]), 100.0 * c[i] / n, 100.0 * tail[i]) for i in range(len(u)) if 0.2 <= 100.0 * tail[i] <= 36.0 and 100.0 * c[i] / n >= 0.25]
    lrows = "".join(f'<tr><td class="n">{v:.4f}</td><td class="n">{at:.1f}</td><td class="n">{above:.1f}</td></tr>' for v, at, above in levels)
    # how well the statistics predict the loops measured under real early exit
    errs = []
    for e in _evals():
        sh = _exit_shares(e["thresholds"][:3])
        errs.append(abs(sum((i + 1) * v for i, v in enumerate(sh)) - e["exec_loops"]))
    g3, g25 = _exit_shares([0.3, 0.3, 0.3]), _exit_shares([0.25, 0.25, 0.25])
    dead3 = 100.0 * float((P[:, 2] >= 0.47).mean())
    above27 = 100.0 * float((P[:, 0] >= 0.27).mean())
    evs = _evals()
    full = next((e for e in evs if e["key"] == (1.0, 1.0, 1.0)), None)
    used = ""
    if full:
        f1s = sorted({round(100.0 * _exit_shares(e["thresholds"][:3])[0], 1) for e in evs
                      if e["key"] not in GATES and e["acc_pct"] >= full["acc_pct"] - TOL} - {0.0}, reverse=True)
        if f1s:
            used = f' The routers of the searches that hold the accuracy floor release {", ".join(f"{v:g}" for v in f1s)} % after loop 1, or nothing.'
    return f'''<div class="sub-head"><h3>Why thresholds act in steps: the gate&#39;s exit probabilities sit near 0.25, 0.5 and 0.75</h3><p>What an exit threshold is compared with, measured directly: for each of the {n:,} answer tokens of the 575 search items (the model&#39;s own full-depth answers, one forward pass each), the cumulative exit probability after loops 1, 2 and 3. They do not spread over 0 to 1. After loop 1 no token goes above {mx[0]:.2f}, after loop 2 none above {mx[1]:.2f}, after loop 3 none above {mx[2]:.2f}, and most tokens sit just under a quarter, a half and three quarters: when the gate is willing to stop, it spreads its exit probability almost evenly over the four loops.</p></div>
<div class="tablewrap" style="padding:14px 16px">
    {_hist_svg(P)}
    <p class="src">Hover a bar for its value. Three consequences for a router. <b>A single threshold acts in steps</b>: the released gate at 0.3 cannot release any token after loop 1 ({100 * g3[0]:.0f} / {100 * g3[1]:.0f} / {100 * g3[2]:.0f} / {100 * g3[3]:.0f} % of the tokens leave after loops 1 / 2 / 3 / 4), at 0.25 it releases {100 * g25[0]:.0f} % after loop 1, and a threshold above 0.27 releases {"practically nothing" if above27 < 0.05 else f"at most {above27:.1f} %"} there. <b>Loop-1 exits are decided among near-identical values</b>: the gate&#39;s output is a 16-bit number, so around 0.25 the probability takes only the values in the table below, each shared by up to 6 % of all tokens; a threshold can only cut between two of them, and the tokens on either side of a cut differ by one step of the gate&#39;s 16-bit output. <b>Thresholds on loop 3 below 0.47 do nothing</b>: {dead3:.1f} % of the tokens are above that value after loop 3, so the first search, which kept q3 under 0.5, let almost no token run the fourth loop. These statistics reproduce the executed loops measured under real early exit for all {len(errs)} routers evaluated so far, with a mean error of {sum(errs) / max(1, len(errs)):.3f} loops (largest {max(errs) if errs else 0:.3f}), so the compute of any router is known before it is run; only its accuracy has to be measured.</p>
    <p style="margin:16px 0 6px;font-weight:600">The settings the gate offers for loop-1 exits</p>
    <table style="max-width:560px">
      <thead><tr><th class="n">probability after loop 1</th><th class="n">tokens at this value, %</th><th class="n">released if the threshold is at or below it, %</th></tr></thead>
      <tbody>
        {lrows}
      </tbody>
    </table>
    <p class="src">Read down the last column: these are the only loop-1 exit shares a threshold can produce in this range. The released gate at 0.25 releases {100 * g25[0]:.1f} %.{used}</p>
  </div>
'''


# ---------------------------------------------------------------- router search with the corrected cache entries (third search)
FILL = "/home/jli199/boptim_scratch/bo_bfcl_router_fill"
FILL_DESIGN = {(0.3, 0.3, 0.3): "gate 0.3", (0.25, 0.25, 0.25): "gate 0.25", (0.2443, 0.2443, 0.2443): "gate 0.2443", (0.2, 0.2, 0.2): "gate 0.2"}


def _fill_overhead(thr):
    """Loops' worth of compute the corrected cache entries add per answer token for a router: one (dim + 1) x dim map per head
    for the key and for the value of every skipped (loop, layer) slot, against 4 h^2 + 3 h m parameters per layer of a loop
    (h = 2048, m = 5632, 16 heads of 128): 3.1 % of one loop for a token leaving after loop 1, 2.1 % after loop 2, 1.0 % after 3."""
    sh = _exit_shares(thr)
    if not sh:
        return 0.0
    h, m, heads, dim = 2048, 5632, 16, 128
    slot = 2 * heads * (dim + 1) * dim / (4 * h * h + 3 * h * m)       # one skipped loop, relative to one loop
    return slot * (3 * sh[0] + 2 * sh[1] + 1 * sh[2])


def _fill_unseen_table(key, n_items, label, names):
    """Validation of corrected routers on an unseen set against the plain released gate measured on the same items."""
    plain = [e for e in _load_ok(f"{WORK}/{key}/evaluations.jsonl") + _load_ok(f"{WORK}/lab_{key}/evaluations.jsonl") if not e.get("extra_cfg")]
    seen, pl = set(), []
    for e in plain:
        if e["key"] in GATES and e["key"] not in seen:
            seen.add(e["key"])
            pl.append(e)
    corr = [e for e in _load_ok(f"{WORK}/lab_{key}/evaluations.jsonl")
            if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf")]
    full = next((e for e in pl if e["key"] == (1.0, 1.0, 1.0)), None)
    if not corr or not full or len(pl) < 3:
        return ""
    floor = full["acc_pct"] - TOL
    curve_pts = sorted(pl, key=lambda e: e["exec_loops"])

    def curve(L):
        lo = max((g for g in curve_pts if g["exec_loops"] <= L), key=lambda g: g["exec_loops"], default=curve_pts[0])
        hi = min((g for g in curve_pts if g["exec_loops"] >= L), key=lambda g: g["exec_loops"], default=curve_pts[-1])
        span = hi["exec_loops"] - lo["exec_loops"]
        return lo["acc_pct"] if span <= 0 else lo["acc_pct"] + (L - lo["exec_loops"]) / span * (hi["acc_pct"] - lo["acc_pct"])

    rows = [f'<tr class="dim"><td>plain {"released " if g["key"] != (1.0, 1.0, 1.0) else ""}{GATES[g["key"]]}</td><td class="n">{g["acc_pct"]:.1f}{" ↓" if g["acc_pct"] < floor else ""}</td>'
            f'<td class="n">{g["exec_loops"]:.2f}</td><td class="n">{4 / g["exec_loops"]:.2f}×</td><td class="n">—</td></tr>' for g in sorted(pl, key=lambda g: -g["exec_loops"])]
    best_c = None
    for e in sorted(corr, key=lambda e: -(e["exec_loops"])):
        e["eff"] = e["exec_loops"] + _fill_overhead(e["thresholds"][:3])
        d = round((e["acc_pct"] - curve(e["eff"])) * n_items / 100.0)
        nm = names.get(e["key"]) or (GATES[e["key"]] if e["key"] in GATES else _fmt(e["key"]))
        if e["acc_pct"] >= floor and (best_c is None or e["eff"] < best_c["eff"]):
            best_c = e
        rows.append(f'<tr><td>{nm}, corrected {_fmt(e["key"])}</td><td class="n">{e["acc_pct"]:.1f}{" ↓" if e["acc_pct"] < floor else ""}</td>'
                    f'<td class="n">{e["eff"]:.2f}</td><td class="n">{4 / e["eff"]:.2f}×</td><td class="n">{"+" if d > 0 else ""}{d}</td></tr>')
    best_p = min((g for g in pl if g["key"] != (1.0, 1.0, 1.0) and g["acc_pct"] >= floor), key=lambda g: g["exec_loops"], default=None)
    verdict = ""
    if best_c and best_p:
        dl = 100 * (1 - best_c["eff"] / best_p["exec_loops"])
        verdict = (f' The cheapest corrected router that holds this floor runs {best_c["eff"]:.2f} loops ({best_c["acc_pct"]:.1f} %), the cheapest plain gate point '
                   f'{best_p["exec_loops"]:.2f} ({GATES[best_p["key"]]}, {best_p["acc_pct"]:.1f} %): {abs(dl):.0f} % {"fewer" if dl > 0 else "more"} loops.')
    return f'''<p style="margin:18px 0 6px;font-weight:600">Validation on {label}</p>
    <table>
      <thead><tr><th>router</th><th class="n">accuracy %</th><th class="n">loops (with the correction)</th><th class="n">compute saved</th><th class="n">items vs plain gate curve</th></tr></thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    <p class="src">Full depth scores {full["acc_pct"]:.1f} % here, so the floor is {floor:.1f} % (↓ marks accuracy below it). None of these items was used to fit the correction or to choose thresholds.{verdict}</p>'''


def bfcl_fill_search_block():
    evs = [e for e in _load_ok(f"{FILL}/evaluations.jsonl") if (e.get("extra_cfg") or {}).get("decode_fill_maps")]
    plain = _evals()
    full = next((e for e in plain if e["key"] == (1.0, 1.0, 1.0)), None)
    gates = sorted((e for e in plain if e["key"] in GATES), key=lambda e: e["exec_loops"])
    if not evs or not full or len(gates) < 2:
        return ""
    floor = full["acc_pct"] - TOL
    log = f"{TMP}/bfcl_bo3.log"
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    finished = "BO3 finished with code 0" in text

    def curve(L):  # plain released-gate accuracy at L executed loops (straight line between its two points around L)
        lo = max((g for g in gates if g["exec_loops"] <= L), key=lambda g: g["exec_loops"], default=gates[0])
        hi = min((g for g in gates if g["exec_loops"] >= L), key=lambda g: g["exec_loops"], default=gates[-1])
        span = hi["exec_loops"] - lo["exec_loops"]
        return lo["acc_pct"] if span <= 0 else lo["acc_pct"] + (L - lo["exec_loops"]) / span * (hi["acc_pct"] - lo["acc_pct"])

    seen, uniq = set(), []
    for e in evs:
        if e["key"] not in seen:
            seen.add(e["key"])
            uniq.append(e)
    k = 0
    for e in uniq:
        if e["key"] in FILL_DESIGN:
            e["name"] = f'<b>{FILL_DESIGN[e["key"]]}, corrected</b>'
            e["short"] = FILL_DESIGN[e["key"]]
        else:
            k += 1
            e["id"] = k
            e["name"] = f'BO #{k} {_fmt(e["key"])}'
            e["short"] = f"#{k}"
    for e in uniq:  # loops including the cost of the corrected entries themselves
        e["eff"] = e["exec_loops"] + _fill_overhead(e["thresholds"][:3])
    holds = [e for e in uniq if e["acc_pct"] >= floor]
    best = min(holds, key=lambda e: (round(e["eff"], 3), -e["acc_pct"]), default=None)
    best_gate = min((g for g in gates if g["key"] != (1.0, 1.0, 1.0) and g["acc_pct"] >= floor), key=lambda g: g["exec_loops"], default=None)
    rows = []
    for e in sorted((e for e in uniq if e["key"] in FILL_DESIGN), key=lambda e: -e["key"][0]) + [e for e in uniq if e["key"] not in FILL_DESIGN]:
        sh = _exit_shares(e["thresholds"][:3])
        shares = " / ".join(f"{100 * v:.0f}" for v in sh) if sh else "—"
        d = round((e["acc_pct"] - curve(e["eff"])) * 5.75)
        rows.append(f'<tr{" class=\"best\"" if e is best else ""}><td>{e["name"]}</td><td class="n">{e["acc_pct"]:.1f}{" ↓" if e["acc_pct"] < floor else ""}</td>'
                    f'<td class="n">{e["exec_loops"]:.2f}</td><td class="n">{e["eff"]:.2f}</td><td class="n">{4 / e["eff"]:.2f}×</td><td class="n">{shares}</td>'
                    f'<td class="n">{"+" if d > 0 else ""}{d}</td></tr>')
    n_bo = sum(1 for e in uniq if e["key"] not in FILL_DESIGN)
    status = f'{sum(1 for e in uniq if e["key"] in FILL_DESIGN)} of {len(FILL_DESIGN)} design points and {n_bo} of {N_BO} proposals evaluated'
    if best and best_gate:
        dl = 100 * (1 - best["eff"] / best_gate["exec_loops"])
        status += (f'; the cheapest router that holds the floor so far is {best["short"] if best["key"] in FILL_DESIGN else "BO " + best["short"]} at {best["eff"]:.2f} loops with the correction&#39;s cost, '
                   f'{abs(dl):.0f} % {"fewer" if dl > 0 else "more"} than the plain {GATES[best_gate["key"]]} ({best_gate["exec_loops"]:.2f} loops)')
    # chart: plain gate curve (squares) against the corrected routers (circles)
    ymin = floor - 6
    pts_b = [{"x": 4 / g["exec_loops"], "y": g["acc_pct"], "name": GATES[g["key"]], "lab": ((-8, 16, "end") if floor - 0.9 < g["acc_pct"] < floor + 0.3 else (0, -10, "middle")),
              "tip": f'plain {GATES[g["key"]]}: {g["acc_pct"]:.1f} % at {g["exec_loops"]:.2f} loops'} for g in sorted(gates, key=lambda g: -g["exec_loops"]) if g["acc_pct"] >= ymin]
    for e in uniq:  # chart labels: "g0.25" = the gate pattern at 0.25 with the correction, "#3" = the third proposal
        e["pid"] = ("g" + FILL_DESIGN[e["key"]].split()[1]) if e["key"] in FILL_DESIGN else e["short"]
    pts_p = [{"x": 4 / e["eff"], "y": e["acc_pct"], "id": e["pid"],
              "tip": f'{e["short"]} corrected {_fmt(e["key"])}: {e["acc_pct"]:.1f} % at {e["eff"]:.2f} loops including the correction'} for e in uniq if e["acc_pct"] >= ymin]
    best_id = best["pid"] if best else None
    pts_p.sort(key=lambda p: p["id"] == best_id)
    xs, ys = [p["x"] for p in pts_b + pts_p], [p["y"] for p in pts_b + pts_p]
    x1 = max(2.0, 0.5 * (int((max(xs) + 0.15) / 0.5) + 1))                 # room to the right of the last point for its label
    y0, y1 = int(min(min(ys), floor) - 1), int(max(ys) + 2)
    design_ids = {e["pid"] for e in uniq if e["key"] in FILL_DESIGN}
    svg = pb4._svg((0.9, x1), (y0, y1), [1 + 0.5 * i for i in range(int((x1 - 1) / 0.5) + 1)], list(range(y0, y1 + 1, 1 if y1 - y0 <= 8 else 2)),
                   pts_p, pts_b, [], round(floor, 2), best_id, {p["id"]: ((9, -9, "start") if p["id"] == best_id else (9, 15, "start")) for p in pts_p},
                   "Accuracy against decode compute on the search set: routers with corrected cache entries against the plain released-gate curve",
                   (pb4.ML + 22, pb4.H - pb4.MB - 78), xlabel="decode compute saved: 4 / loops including the correction (higher is cheaper)",
                   ylabel="accuracy on the 575 search items, %", floor_right=True, best_label="cheapest corrected router that holds the floor", id_prefix="",
                   legend_names=("plain released gate and full depth", "corrected router that holds the accuracy floor", "corrected router below the floor"),
                   label_only=design_ids | ({best_id} if best_id else set()))
    chip = _chip("good", "finished") if finished else _chip("info", "running")
    names = {e["key"]: (e["short"] if e["key"] in FILL_DESIGN else "BO " + e["short"]) for e in uniq}
    unseen = (_fill_unseen_table("heldout", 575, "the 575 unseen non-live items", names)
              + _fill_unseen_table("live", 1350, "the 1,350 live items", names))
    return f'''<div class="sub-head"><h3>Router search with corrected cache entries {chip}</h3><p>The third search on BFCL changes what an early exit leaves behind, not only when a token leaves. A token that leaves early no longer leaves plain copies of its keys and values in the cache slots of the loops it skips, but corrected ones: one linear map per layer, head and loop pair, fitted once on full-depth answers (the evidence is in the block above). The released gate still decides the exits. With that, the region below two loops, where the plain gate collapses, is worth searching. Search variables are the exit shares: f1, the share of all answer tokens released after loop 1 (0 to 70 %); f2, the share of the remaining tokens released after loop 2 (60 to 100 %); f3, the share of the still remaining tokens released after loop 3 (50 to 100 %). Each proposal becomes thresholds placed between two neighbouring values of the gate. Same objective, floor ({floor:.1f} %), search items and checker as the first search; the released-gate patterns with the correction seed the optimiser, four grid points follow, then sixteen model-based proposals.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>router (corrected cache entries)</th><th class="n">accuracy %</th><th class="n">executed loops</th><th class="n">with the correction</th><th class="n">compute saved</th><th class="n">leaving after loop 1/2/3/4, %</th><th class="n">items vs plain gate curve</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">↓ marks accuracy below the floor. "With the correction" adds what the corrected entries cost to compute: one small matrix product per skipped cache slot, 3.1 % of one loop for a token that leaves after loop 1, 2.1 % after loop 2, 1.0 % after loop 3. Compute saved, the chart and the last column use that figure; the last column compares each router with the plain released gate at the same loops (straight line between its two points around it). Status: {status}. Trajectories: <code>{FILL}/evaluations.jsonl</code>.</p>
    <p style="margin:18px 0 6px;font-weight:600">Accuracy against decode compute, search set</p>
    {svg}
    <p class="src">Squares: the plain released gate (no correction), joined in threshold order. Circles: routers with corrected cache entries (hover for thresholds); labelled are the gate patterns with the correction ("g0.25" is the pattern at 0.25) and the cheapest router that holds the floor. The plain gate falls off to the right of threshold 0.25; the corrected routers hold up further.</p>
    {unseen}
  </div>
'''


# ---------------------------------------------------------------- what limits real early exit below two loops (exit lab)
def _lab_label(gate, conf, maps, skip):
    parts = []
    if gate and min(gate[:3]) < 0.99:
        parts.append(f"gate {gate[0]:g}" if len(set(round(v, 4) for v in gate[:3])) == 1 else "gate " + _fmt(gate))
    if conf:
        if conf[0] >= 1.0:
            parts.append(f"confidence ≥ {conf[1]:g} from loop 2 on")
        elif len(set(conf)) == 1:
            parts.append(f"confidence ≥ {conf[0]:g}")
        else:
            parts.append("confidence ≥ " + " / ".join(f"{v:g}" for v in conf))
    if skip and any(skip):
        parts.append("no cache entry" + (" in the lower 32 layers" if skip[0] == 32 else "") + (" after loop-1 exits" if not any(skip[1:]) else " for any skipped loop"))
    if maps:
        parts.append("corrected cache entries")
    return ", ".join(parts) if parts else "full depth"


def _load_ok(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("status") == "ok":
                e["key"] = tuple(round(v, 4) for v in e["thresholds"][:3])
                out.append(e)
    return out


def _fill_heldout_rows():
    """Held-out check of the corrected cache entries: the released gate's thresholds, plain against corrected, on the 575 unseen items."""
    plain = {e["key"]: e for e in _load_ok(f"{WORK}/heldout/evaluations.jsonl") if not e.get("extra_cfg")}
    corr = {e["key"]: e for e in _load_ok(f"{WORK}/lab_heldout/evaluations.jsonl")
            if (e.get("extra_cfg") or {}).get("decode_fill_maps") and not (e.get("extra_cfg") or {}).get("decode_exit_conf")}
    rows, gains = [], []
    for k in sorted(corr, key=lambda k: -k[0]):
        if k not in plain or k not in GATES:
            continue
        p, c = plain[k], corr[k]
        d = round((c["acc_pct"] - p["acc_pct"]) * 5.75)
        gains.append(d)
        rows.append(f'<tr><td>released {GATES[k]}</td><td class="n">{p["acc_pct"]:.1f}</td><td class="n">{p["exec_loops"]:.2f}</td><td class="n">{c["acc_pct"]:.1f}</td>'
                    f'<td class="n">{c["exec_loops"] + _fill_overhead(c["thresholds"][:3]):.2f}</td><td class="n">{"+" if d > 0 else ""}{d}</td></tr>')
    if not rows:
        return '<p class="src">Held-out check of the corrected cache entries: running.</p>'
    full = plain.get((1.0, 1.0, 1.0))
    verdict = ("The gain carries over to unseen items." if max(gains) >= 5 and min(gains) >= -2 else
               "On unseen items the gain is within noise." if max(gains) >= 0 else "On unseen items the correction does not help.")
    return f'''<p style="margin:18px 0 6px;font-weight:600">4. Held-out check: the same thresholds on the 575 unseen items, plain against corrected cache entries</p>
    <table style="max-width:820px">
      <thead><tr><th>router</th><th class="n">plain %</th><th class="n">loops</th><th class="n">corrected %</th><th class="n">loops</th><th class="n">items gained</th></tr></thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    <p class="src">{verdict}{f" Full depth scores {full['acc_pct']:.1f} % on this set, so its floor is {full['acc_pct'] - TOL:.1f} %." if full else ""} None of these items was used to fit the correction or to choose thresholds. The loops of the corrected routers include what the correction itself costs (about 0.02 to 0.03 loops per token here).</p>'''


def bfcl_lab_block():
    sig_p, tf_p, fit_p, lab_p = (f"{WORK}/signal_analysis.json", f"{WORK}/tf_exit_lab.jsonl", f"{WORK}/kv_fill_fit.json", f"{WORK}/lab_search/evaluations.jsonl")
    if not (os.path.exists(sig_p) and os.path.exists(tf_p) and os.path.exists(fit_p)):
        return ""
    sig = json.load(open(sig_p))
    fit = json.load(open(fit_p))["report"]
    tf = {}
    for line in open(tf_p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        parts = r["router"].split("|")
        opt = dict(p.split("=", 1) for p in parts[1:])
        r["label"] = _lab_label([float(v) for v in parts[0].split(",")], [float(v) for v in opt["conf"].split(",")] if "conf" in opt else None,
                                opt.get("maps"), [int(v) for v in opt["skip"].split(",")] if "skip" in opt else None)
        tf[r["label"]] = r
    order = ["gate 0.3", "gate 0.25", "gate 0.2", "gate 0.2, corrected cache entries", "confidence ≥ 0.99", "confidence ≥ 0.99, corrected cache entries",
             "confidence ≥ 0.99, no cache entry in the lower 32 layers after loop-1 exits", "confidence ≥ 0.99, no cache entry after loop-1 exits",
             "confidence ≥ 0.95 from loop 2 on"]
    trows = "".join(
        f'<tr><td>{lab}</td><td class="n">{r["loops"]:.2f}</td><td class="n">{r["wrong_pct"]:.2f}</td><td class="n">{r["wrong_selection_pct"]:.2f}</td>'
        f'<td class="n">{r["wrong_cache_pct"]:.2f}</td><td class="n">{r["by_exit_loop"]["1"]["wrong_pct"]:.2f}</td><td class="n">{r["by_exit_loop"]["4"]["wrong_pct"]:.1f}</td></tr>'
        for lab in order for r in [tf.get(lab)] if r)
    n_items = next(iter(tf.values()))["items"] if tf else 0
    n_tok = next(iter(tf.values()))["tokens"] if tf else 0
    plain, corr, none = tf.get("confidence ≥ 0.99"), tf.get("confidence ≥ 0.99, corrected cache entries"), tf.get("confidence ≥ 0.99, no cache entry after loop-1 exits")
    red = f'it removes about {100 * (1 - corr["wrong_cache_pct"] / plain["wrong_cache_pct"]):.0f} % of the cache-caused errors in the forced test' if plain and corr else "it helps in the forced test"
    worse = (f'Giving the deeper loops no entry at all leaves {none["wrong_cache_pct"] / plain["wrong_cache_pct"]:.0f} times as many cache-caused errors as the copy, so they do need one.'
             if plain and none else "")
    full_e = next((e for e in _evals() if e["key"] == (1.0, 1.0, 1.0)), None)
    full_acc = f'{full_e["acc_pct"]:.0f}' if full_e else "83"
    lrows = "".join(
        f'<tr><td class="n">{l}</td><td class="n">{fit["keys 1->2"]["copy"][l]:.2f}</td><td class="n">{fit["keys 1->2"]["affine"][l]:.2f}</td>'
        f'<td class="n">{fit["values 1->2"]["copy"][l]:.2f}</td><td class="n">{fit["values 1->2"]["affine"][l]:.2f}</td>'
        f'<td class="n">{fit["keys 2->3"]["copy"][l]:.2f}</td><td class="n">{fit["keys 3->4"]["copy"][l]:.2f}</td></tr>' for l in (0, 4, 8, 16, 24, 32, 40, 47))
    # real generation on the search set: lab routers against the released gate's own curve
    evs = _evals()
    gates = sorted((e for e in evs if e["key"] in GATES), key=lambda e: e["exec_loops"])
    real = ""
    if os.path.exists(lab_p) and len(gates) >= 2:
        lab_rows = []
        for line in open(lab_p):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("status") != "ok":
                continue
            x = e.get("extra_cfg") or {}
            lab = _lab_label(e["thresholds"][:3], x.get("decode_exit_conf"), x.get("decode_fill_maps"), x.get("decode_fill_skip"))
            L = e["exec_loops"] + (_fill_overhead(e["thresholds"][:3]) if x.get("decode_fill_maps") and not x.get("decode_exit_conf") else 0.0)
            lo = max((g for g in gates if g["exec_loops"] <= L), key=lambda g: g["exec_loops"], default=gates[0])
            hi = min((g for g in gates if g["exec_loops"] >= L), key=lambda g: g["exec_loops"], default=gates[-1])
            span = hi["exec_loops"] - lo["exec_loops"]
            line_acc = lo["acc_pct"] if span <= 0 else lo["acc_pct"] + (L - lo["exec_loops"]) / span * (hi["acc_pct"] - lo["acc_pct"])
            d = round((e["acc_pct"] - line_acc) * 5.75)
            lab_rows.append((not x.get("decode_exit_conf"), d, f'<tr><td>{lab}</td><td class="n">{e["acc_pct"]:.1f}</td><td class="n">{L:.2f}</td>'
                             f'<td class="n">{line_acc:.1f}</td><td class="n">{"+" if d > 0 else ""}{d}</td></tr>'))
        base = "".join(f'<tr class="dim"><td>{"released " if g["key"] != (1.0, 1.0, 1.0) else ""}{GATES[g["key"]]}</td><td class="n">{g["acc_pct"]:.1f}</td>'
                       f'<td class="n">{g["exec_loops"]:.2f}</td><td class="n">—</td><td class="n">—</td></tr>' for g in sorted(gates, key=lambda g: -g["exec_loops"]))
        rr = [r for _, _, r in sorted(lab_rows, key=lambda t: (not t[0], -t[1]))]            # gate routers first, best first
        gate_gain = [d for is_gate, d, _ in lab_rows if is_gate]
        conf_gain = [d for is_gate, d, _ in lab_rows if not is_gate]
        helps = bool(gate_gain) and max(gate_gain) >= 6
        title = ("3. In real generation the corrected cache entries lift the gate&#39;s own curve; confidence exits do not help" if helps
                 else "3. In real generation neither idea beats the released gate")
        held = _fill_heldout_rows()
        if rr:
            real = f'''<p style="margin:18px 0 6px;font-weight:600">{title} (575 search items, official checker)</p>
    <table>
      <thead><tr><th>router</th><th class="n">accuracy %</th><th class="n">executed loops</th><th class="n">plain gate curve at the same loops, %</th><th class="n">items above (+) or below (−) it</th></tr></thead>
      <tbody>
        {base}{"".join(rr)}
      </tbody>
    </table>
    <p class="src">"Plain gate curve at the same loops" is the straight line between the two released-gate points around the router (below the last point, that point itself). {"With corrected cache entries the released gate&#39;s own routers gain up to " + str(max(gate_gain)) + " items of 575 at the same thresholds: the entries left by early leavers were costing accuracy, and a linear correction gives part of it back. " if helps else ""}{"Confidence routers stay " + str(-max(conf_gain)) + " to " + str(-min(conf_gain)) + " items below the gate curve. " if conf_gain and max(conf_gain) < 0 else ""}The confidence router from loop 2 on agreed with the full-depth tokens more often than the gate at 0.3 in the forced test above, yet scores lower: agreeing with the full-depth output is not the same as being right, since full depth itself is right on {full_acc} % of the items. The correction was fitted on the answers of a fifth of these search items, so the clean test is the held-out set.</p>
    {held}'''
    return f'''<div class="sub-head"><h3>What limits real early exit below two loops: the cache left behind by tokens that leave after loop 1</h3><p>Function-call answers look like the ideal case for early exit. With an exact cache, the head&#39;s prediction after loop 1 already equals the full-depth token for {100 * sig["safe_share"][0]:.1f} % of the {sig["tokens"]:,} answer tokens, and after loop 2 for {100 * sig["safe_share"][1]:.1f} %; a router that always left at the first loop that is already right would run {sig["oracle_loops"]:.2f} loops. Real early exit does not get there: the released gate loses accuracy as soon as it releases tokens after loop 1 (threshold 0.25 and below). The measurements below say why, what does not fix it, and what gives part of it back.</p></div>
<div class="tablewrap" style="padding:14px 16px">
    <p style="margin:0 0 6px;font-weight:600">1. Under real early exit the errors land on the tokens that keep going</p>
    <table>
      <thead><tr><th style="min-width:250px">router</th><th class="n">loops</th><th class="n">wrong, %</th><th class="n">let out wrong, %</th><th class="n">from the cache, %</th><th class="n">loop-1 leavers wrong, %</th><th class="n">4-loop tokens wrong, %</th></tr></thead>
      <tbody>
        {trows}
      </tbody>
    </table>
    <p class="src">Forced test on {n_items} search items ({n_tok:,} answer tokens): the model is fed its own full-depth answer token by token and exits exactly as in generation; each prediction is compared with the full-depth token. "Let out wrong" means the head was already wrong after that loop with an exact cache; "from the cache" means it is right with an exact cache and wrong here. Tokens that leave after loop 1 are almost always right. The cost is paid later, by the tokens that need more loops: they attend to the cache entries the early leavers left behind.</p>
    <p style="margin:18px 0 6px;font-weight:600">2. Why: a loop-1 cache entry is a poor stand-in for the entry a deeper loop would have computed</p>
    <table style="max-width:820px">
      <thead><tr><th class="n">layer</th><th class="n">keys 1→2 copied</th><th class="n">corrected</th><th class="n">values 1→2 copied</th><th class="n">corrected</th><th class="n">keys 2→3 copied</th><th class="n">keys 3→4 copied</th></tr></thead>
      <tbody>
        {lrows}
      </tbody>
    </table>
    <p class="src">Error of the stand-in relative to the spread of the true entries over tokens (1.00 = no better than a constant), held-out answer tokens. When a token leaves after loop 1, the slots of loops 2 to 4 are filled with copies of its loop-1 keys and values. In the lower half of the layers such a copy is worse than a constant, because loop 1 computed it from a residual stream that the deeper loops never see in that form. Copies between deeper loops are close, which is why exits after loop 2 are harmless. "Corrected" is a linear map per layer and head, fitted on other answers: {red}. {worse} A stand-in built from the finished loop-1 state is exact in the first layer and no better in the upper ones. The entries of confident tokens are harder to predict than the others, so a confident prediction does not mean a settled state.</p>
    {real}
  </div>
'''
