"""Page block added 2026-11-14: Pareto views (pass@1 against speed-up) of the two LiveCodeBench BO runs, each with the released-gate
baselines measured on the same problems: (1) the first run, searched on the 120-problem slice; (2) the pooled run over all 342.
Pure inline SVG that uses the page's colour tokens, so it follows the light and dark themes. Re-reads the result files on every build."""
import json
import os

import page_blocks_1109 as pb2
from page_blocks_1107 import TMP

SUB_EVALS = "/home/jli199/boptim_scratch/bo_lcb_router/evaluations.jsonl"
POOLED_EVALS = "/home/jli199/boptim_scratch/bo_lcb_router_pooled/evaluations.jsonl"
RECAL = "/home/jli199/boptim_scratch/bo_lcb_router_pooled/recal_speeds.json"
W, H, ML, MR, MT, MB = 760, 440, 58, 26, 18, 52


def _wallclock_slice0():
    """label -> (pass@1 %, tok/s) for the wall-clock runs on problems 0..119 (labels without a slice suffix)."""
    res, grd = {}, {}
    p = f"{TMP}/wallclock_lcb.txt"
    if not os.path.exists(p):
        return {}
    for line in open(p):
        if line.startswith("RESULT "):
            lab = line.split()[1]
            try:
                res[lab] = json.loads(line[len("RESULT " + lab) + 1:])
            except Exception:
                pass
        elif " GRADED " in line:
            lab = line.split()[0]
            kv = dict(x.split("=", 1) for x in line.split()[2:] if "=" in x and not x.startswith("by_difficulty"))
            grd[lab] = float(kv["pass@1"])
    return {lab: (100 * grd[lab], res[lab]["tok_s"]) for lab in res if lab in grd and "_o" not in lab}


def _frontier(pts):
    """Non-dominated points (maximise x and y), sorted by x."""
    out = []
    for p in pts:
        if not any((q["x"] >= p["x"] and q["y"] >= p["y"]) and (q["x"] > p["x"] or q["y"] > p["y"]) for q in pts):
            out.append(p)
    return sorted(out, key=lambda p: (p["x"], -p["y"]))


def _svg(xr, yr, xt, yt, bo, base, extra, floor, best_id, label_dx, aria, legend_xy):
    x0, x1 = xr
    y0, y1 = yr
    pw, ph = W - ML - MR, H - MT - MB

    def X(v):
        return ML + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return MT + (y1 - v) / (y1 - y0) * ph

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{aria}" style="width:100%;height:auto;max-width:{W}px;display:block;font-family:inherit">']
    # grid and axes
    for v in yt:
        o.append(f'<line x1="{ML}" x2="{W - MR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" style="stroke:var(--line);stroke-width:1"/>')
        o.append(f'<text x="{ML - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" style="fill:var(--text-2);font-size:12px">{v:g}</text>')
    for v in xt:
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{MT}" y2="{H - MB}" style="stroke:var(--line);stroke-width:1"/>')
        o.append(f'<text x="{X(v):.1f}" y="{H - MB + 17}" text-anchor="middle" style="fill:var(--text-2);font-size:12px">{v:g}×</text>')
    o.append(f'<line x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}" style="stroke:var(--line-2);stroke-width:1.2"/>')
    o.append(f'<line x1="{ML}" x2="{ML}" y1="{MT}" y2="{H - MB}" style="stroke:var(--line-2);stroke-width:1.2"/>')
    o.append(f'<text x="{ML + pw / 2:.0f}" y="{H - 10}" text-anchor="middle" style="fill:var(--text-2);font-size:12.5px">speed-up over full depth (higher is faster)</text>')
    o.append(f'<text transform="translate(15 {MT + ph / 2:.0f}) rotate(-90)" text-anchor="middle" style="fill:var(--text-2);font-size:12.5px">pass@1, %</text>')
    # accuracy floor of the objective
    o.append(f'<line x1="{ML}" x2="{W - MR}" y1="{Y(floor):.1f}" y2="{Y(floor):.1f}" style="stroke:var(--text-3);stroke-width:1.2;stroke-dasharray:5 4"/>')
    o.append(f'<text x="{ML + 7}" y="{Y(floor) + 14:.1f}" text-anchor="start" style="fill:var(--text-3);font-size:11.5px">accuracy floor of the objective, {floor:g} %</text>')
    # released-gate curve (in threshold order) and the BO frontier
    o.append('<polyline points="' + " ".join(f'{X(p["x"]):.1f},{Y(p["y"]):.1f}' for p in base if p.get("curve", True))
             + '" style="fill:none;stroke:var(--base);stroke-width:1.6"/>')
    fr = _frontier(bo)
    if fr:
        steps = []
        for i, p in enumerate(fr):
            if i:
                steps.append(f'{X(p["x"]):.1f},{Y(fr[i - 1]["y"]):.1f}')
            steps.append(f'{X(p["x"]):.1f},{Y(p["y"]):.1f}')
        o.append('<polyline points="' + " ".join(steps) + '" style="fill:none;stroke:var(--s1);stroke-width:1.4;stroke-opacity:.55"/>')
    fr_ids = {p["id"] for p in fr}
    # BO proposals
    for p in bo:
        cx, cy = X(p["x"]), Y(p["y"])
        feas = p["y"] >= floor
        if p["id"] == best_id:
            style = "fill:var(--s2);stroke:var(--surface);stroke-width:1.6"
            r = 7.5
        elif feas:
            style = "fill:var(--s1);stroke:var(--surface);stroke-width:1.5" if p.get("clean", True) else "fill:var(--surface);stroke:var(--s1);stroke-width:2.2"
            r = 5.5
        else:
            style = "fill:var(--surface);stroke:var(--s1);stroke-width:1.4;stroke-opacity:.6"
            r = 4.5
        o.append(f'<g><title>{p["tip"]}</title><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" style="{style}"/></g>')
        if p["id"] in fr_ids or p["id"] == best_id:
            dx, dy, anchor = label_dx.get(p["id"], (8, -7, "start"))
            o.append(f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}" style="fill:var(--text);font-size:11.5px">#{p["id"]}</text>')
    # first-run routers re-scored (warm start), if any
    for p in extra:
        cx, cy = X(p["x"]), Y(p["y"])
        d = 6
        o.append(f'<g><title>{p["tip"]}</title><path d="M{cx:.1f},{cy - d:.1f} L{cx + d:.1f},{cy:.1f} L{cx:.1f},{cy + d:.1f} L{cx - d:.1f},{cy:.1f} Z" style="fill:var(--surface);stroke:var(--s3);stroke-width:1.8"/></g>')
    # baselines on top, with direct labels
    for p in base:
        cx, cy = X(p["x"]), Y(p["y"])
        o.append(f'<g><title>{p["tip"]}</title><rect x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" style="fill:var(--base);stroke:var(--surface);stroke-width:1.5"/></g>')
        dx, dy, anchor = p.get("lab", (0, -10, "middle"))
        o.append(f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}" style="fill:var(--text);font-size:11.5px;font-weight:600">{p["name"]}</text>')
    # legend
    lx, ly = legend_xy
    items = [("rect", "var(--base)", "released gate and full depth (baseline)"), ("dot", "var(--s1)", "BO proposal that holds the accuracy floor"),
             ("ring", "var(--s1)", "BO proposal below the floor" + (" or with a shared-card speed reading" if any(not p.get("clean", True) for p in bo) else "")),
             ("best", "var(--s2)", "the run&#39;s best by its objective")]
    if extra:
        items.append(("diamond", "var(--s3)", "first-run routers, re-scored on all 342 (warm start)"))
    for i, (kind, col, text) in enumerate(items):
        yy = ly + i * 18
        if kind == "rect":
            o.append(f'<rect x="{lx - 5}" y="{yy - 5}" width="10" height="10" style="fill:{col}"/>')
        elif kind == "dot":
            o.append(f'<circle cx="{lx}" cy="{yy}" r="5" style="fill:{col}"/>')
        elif kind == "ring":
            o.append(f'<circle cx="{lx}" cy="{yy}" r="4.5" style="fill:var(--surface);stroke:{col};stroke-width:1.6"/>')
        elif kind == "best":
            o.append(f'<circle cx="{lx}" cy="{yy}" r="6.5" style="fill:{col}"/>')
        else:
            o.append(f'<path d="M{lx},{yy - 6} L{lx + 6},{yy} L{lx},{yy + 6} L{lx - 6},{yy} Z" style="fill:var(--surface);stroke:{col};stroke-width:1.8"/>')
        o.append(f'<text x="{lx + 13}" y="{yy + 4}" style="fill:var(--text-2);font-size:11.5px">{text}</text>')
    o.append("</svg>")
    return "\n".join(o)


def _subset_chart():
    wc = _wallclock_slice0()
    if "full" not in wc or not os.path.exists(SUB_EVALS):
        return "", None
    full = wc["full"][1]
    evs = [json.loads(l) for l in open(SUB_EVALS) if l.strip()]
    evs = [e for e in evs if e.get("status") == "ok"]
    bo = []
    for k, e in enumerate(evs, 1):
        t = e["thresholds"][:3]
        bo.append({"id": k, "x": e["tok_s"] / full, "y": e["pass1_pct"], "value": e["value"],
                   "tip": f'#{k} thresholds ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): {e["pass1_pct"]:.1f} % at {e["tok_s"]:.2f} tokens/s ({e["tok_s"] / full:.2f}×)'})
    best = min(bo, key=lambda p: p["value"])
    names = [("full", "full depth", (0, -11, "middle")), ("gate0.7", "gate 0.7", (0, -11, "middle")), ("gate0.5", "gate 0.5", (-9, 15, "end")),
             ("gate0.3", "gate 0.3", (-10, 4, "end")), ("gate0.2", "gate 0.2", (0, 20, "middle"))]
    base = [{"name": n, "x": wc[k][1] / full, "y": wc[k][0], "lab": lab, "tip": f'{n}: {wc[k][0]:.1f} % at {wc[k][1]:.2f} tokens/s ({wc[k][1] / full:.2f}×)'}
            for k, n, lab in names if k in wc]
    svg = _svg((0.9, 2.6), (8, 26), [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], [8, 10, 12, 14, 16, 18, 20, 22, 24, 26], bo, base, [], 21.5, best["id"],
               {18: (8, -8, "start"), 16: (9, 4, "start"), 21: (-9, -7, "end"), 20: (9, 4, "start"), 7: (8, -8, "start"), 8: (8, 12, "start"), 2: (-8, -8, "end")},
               "Pareto view of the first BO run on the 120-problem slice: pass@1 against speed-up for 21 proposals and the released-gate baselines", (ML + 22, H - MB - 96))
    return svg, {"bo": bo, "base": base, "best": best}


def _full_chart():
    if not (os.path.exists(POOLED_EVALS) and os.path.exists(RECAL)):
        return "", None
    recal = {tuple(round(v, 3) for v in d["x"]): d for d in json.load(open(RECAL)).values()}
    evs = [json.loads(l) for l in open(POOLED_EVALS) if l.strip()]
    evs = [e for e in evs if e.get("status") == "ok" and e.get("speedup")]
    bo = []
    for k, e in enumerate(evs, 1):
        t = e["thresholds"][:3]
        bo.append({"id": k, "x": e["speedup"], "y": e["pass1_pct"], "value": e["value"], "clean": bool(e.get("cal_clean", True)),
                   "tip": f'#{k} thresholds ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): {e["pass1_pct"]:.1f} % at {e["speedup"]:.2f}×'
                          + ("" if e.get("cal_clean", True) else " (speed read on a shared card)")})
    best = min(bo, key=lambda p: p["value"])
    base = []
    for key, lab, name, pos in (((1.0, 1.0, 1.0), "full", "full depth", (0, -11, "middle")), ((0.7, 0.7, 0.7), "gate0.7", "gate 0.7", (-9, 15, "end")),
                                ((0.5, 0.5, 0.5), "gate0.5", "gate 0.5", (-10, 17, "end")), ((0.3, 0.3, 0.3), "gate0.3", "gate 0.3", (4, 19, "middle"))):
        acc, sp = pb2._pooled342(lab), recal.get(key)
        if acc and sp:
            base.append({"name": name, "x": sp["speedup"], "y": 100 * acc["acc"], "lab": pos,
                         "tip": f'{name}: {100 * acc["acc"]:.1f} % at {sp["speedup"]:.2f}× (calibration); {acc["tok_s"] / pb2._pooled342("full")["tok_s"]:.2f}× by wall clock over the 342'})
    base.sort(key=lambda p: -p["y"] if p["name"] != "full depth" else -1e9)
    order = {"full depth": 0, "gate 0.7": 1, "gate 0.5": 2, "gate 0.3": 3}
    base.sort(key=lambda p: order[p["name"]])
    extra = []
    for key, name, acc in (((0.49, 0.37, 0.314), "first-run #18 (0.49, 0.37, 0.31)", 19.0), ((0.408, 0.269, 0.313), "first-run #15 (0.41, 0.27, 0.31)", 17.8),
                           ((0.529, 0.058, 0.348), "first-run #20 (0.53, 0.06, 0.35)", 16.7)):
        sp = recal.get(key)
        if sp:
            extra.append({"x": sp["speedup"], "y": acc, "tip": f'{name}: {acc:.1f} % over all 342 at {sp["speedup"]:.2f}×'})
    svg = _svg((0.85, 2.0), (16, 21), [1.0, 1.2, 1.4, 1.6, 1.8, 2.0], [16, 17, 18, 19, 20, 21], bo, base, extra, 18.3, best["id"],
               {}, "Pareto view of the pooled BO run over all 342 problems: pass@1 against calibrated speed-up for 22 proposals and the released-gate baselines",
               (ML + 22, H - MB - 96))
    return svg, {"bo": bo, "base": base, "best": best, "extra": extra}


def pareto_block():
    s1, d1 = _subset_chart()
    s2, d2 = _full_chart()
    if not s1 and not s2:
        return ""
    cap1 = cap2 = ""
    if d1:
        above = [p for p in d1["bo"] if p["y"] >= 21.5 and p["x"] > max(b["x"] for b in d1["base"] if b["y"] >= 21.5)]
        b = d1["best"]
        cap1 = (f'First run: 21 proposals scored on the 120-problem slice (speed = each evaluation&#39;s own generation throughput over the slice, relative to full depth). '
                f'{len(above)} proposals sit to the right of every baseline that holds the 21.5 % floor, and the run&#39;s best, #{b["id"]}, reads {b["y"]:.1f} % at {b["x"]:.2f}×. '
                f'This is the picture that did not transfer: pass@1 on 120 problems has a standard error of 3.8 points, and on the 222 unseen problems the same routers fell '
                f'to or below the released gate (held-out table above).')
    if d2:
        b = d2["best"]
        g3 = next((x for x in d2["base"] if x["name"] == "gate 0.3"), None)
        cap2 = (f'Second run: 22 proposals scored on all 342 problems; speed-ups come from the stratified calibration, the same protocol for proposals and baselines. '
                f'The best by the objective, #{b["id"]}, reads {b["y"]:.1f} % at {b["x"]:.2f}×' + (f' against {g3["y"]:.1f} % at {g3["x"]:.2f}× for gate 0.3' if g3 else "")
                + '. No proposal lies clearly above and to the right of the gate curve: the calibrated speed-ups carry about ±10 % noise (gate 0.7 reads 0.97× here and 1.23× by wall clock), '
                'and the accuracy differences among the fast proposals are within four problems of 342. Hover a point for its thresholds.')
    return f'''<div class="sub-head"><h3>Pareto views of the two BO runs: proposals against the released gate</h3><p>Each chart plots every router the optimiser proposed as pass@1 against speed-up, with full depth and the released gate at its thresholds as the baseline curve, on the same problems as the proposals. Up and to the right is better; the thin blue step line is the proposals&#39; own Pareto front, and the dashed line is the accuracy floor the objective enforced.</p></div>
<div class="tablewrap" style="padding:14px 16px">
    <p style="margin:0 0 6px;font-weight:600">1. Subset: search on the first 120 problems</p>
    {s1}
    <p class="src">{cap1}</p>
    <p style="margin:18px 0 6px;font-weight:600">2. Full set: search on all 342 problems</p>
    {s2}
    <p class="src">{cap2}</p>
  </div>
'''
