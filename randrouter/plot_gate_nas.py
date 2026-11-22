"""Progress figure of the gate NAS (Stage-II gate trained from scratch, searched by BO) on BFCL and LiveCodeBench.

Left: accuracy against executed loops per token for every gate the search evaluated (numbered in order), next to the released
model's own measured points and the accuracy floor. Right: the search's best value so far against the released gate's best
value (lower is better). Reads the live evaluation files, so it can be rerun while the searches go on. No plotting library is
installed here, so the figure is written as an HTML page of inline SVG and rendered to PNG with headless Chrome.

  python randrouter/plot_gate_nas.py [--out /home/jli199/boptim_scratch/randrouter/gate_nas_progress]   (writes .html and .png)
"""
import argparse
import json
import os
import subprocess

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
BLUE, GREY = "#2a78d6", "#8a8984"
FONT = "font-family: Inter, 'Helvetica Neue', Arial, sans-serif"

BFCL_REF = [(1.401, 76.87, "gate 0.2"), (1.911, 80.87, "gate 0.25"), (2.123, 82.43, "gate 0.3"), (2.854, 80.70, "gate 0.5"),
            (3.237, 82.09, "gate 0.7"), (4.0, 82.78, "full depth")]
LCB_REF = [(1.0, 5.6, "depth 1"), (2.0, 15.8, "depth 2"), (2.208, 17.8, "gate 0.3"), (4.0, 20.5, "full depth")]


def bfcl_value(loops, acc):
    return 100 * loops / 4 + 5.0 * max(0.0, 81.08 - acc)


def lcb_value(loops, p1):
    return -6.25 * 4 / loops + 2.0 * max(0.0, 18.3 - p1)


def load(path, acc_key):
    out = []
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("status") not in ("ok", "shortcut"):
                continue
            loops = r.get("exec_loops") or r.get("loops_eff")
            out.append((loops, r.get(acc_key, r.get("pass1_pct")), r["value"]))
    return out


class Panel:
    W, H, L, R, T, B = 600, 400, 64, 120, 56, 52

    def __init__(self, title, sub, xr, yr, xlabel, ylabel, xticks, yticks):
        self.title, self.sub, self.xr, self.yr = title, sub, xr, yr
        self.xlabel, self.ylabel, self.xticks, self.yticks = xlabel, ylabel, xticks, yticks
        self.parts = []

    def x(self, v):
        return self.L + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.W - self.L - self.R)

    def y(self, v):
        return self.H - self.B - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.H - self.T - self.B)

    def text(self, x, y, s, size=11, color=INK2, anchor="start", weight="normal"):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')

    def frame(self):
        p = self.parts
        for t in self.yticks:
            p.append(f'<line x1="{self.L}" x2="{self.W - self.R}" y1="{self.y(t):.1f}" y2="{self.y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
            self.text(self.L - 8, self.y(t) + 4, f"{t:g}", 10, INK2, "end")
        for t in self.xticks:
            p.append(f'<line x1="{self.x(t):.1f}" x2="{self.x(t):.1f}" y1="{self.T}" y2="{self.H - self.B}" stroke="{GRID}" stroke-width="1"/>')
            self.text(self.x(t), self.H - self.B + 16, f"{t:g}", 10, INK2, "middle")
        p.append(f'<line x1="{self.L}" x2="{self.W - self.R}" y1="{self.H - self.B}" y2="{self.H - self.B}" stroke="{AXIS}" stroke-width="1"/>')
        self.text(self.L, 20, self.title, 13, INK, "start", "600")
        self.text(self.L, 38, self.sub, 11, INK2)
        self.text((self.L + self.W - self.R) / 2, self.H - 12, self.xlabel, 11, INK2, "middle")
        self.parts.append(f'<text x="16" y="{(self.T + self.H - self.B) / 2:.1f}" font-size="11" fill="{INK2}" text-anchor="middle" '
                          f'transform="rotate(-90 16 {(self.T + self.H - self.B) / 2:.1f})">{self.ylabel}</text>')

    def hline(self, v, label, dashed=False, width=1.5):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(f'<line x1="{self.L}" x2="{self.W - self.R}" y1="{self.y(v):.1f}" y2="{self.y(v):.1f}" stroke="{GREY}" stroke-width="{width}"{dash}/>')
        self.text(self.W - self.R + 6, self.y(v) + 4, label, 10, INK2)

    def svg(self):
        return f'<svg viewBox="0 0 {self.W} {self.H}" width="{self.W}" height="{self.H}" style="{FONT}">' + "".join(self.parts) + "</svg>"


def scatter_panel(evals, ref, floor, title, sub, ylabel, yr, yticks, label_offsets=None, legend_at=None, point_offsets=None):
    p = Panel(title, sub, (0.9, 4.1), yr, "executed loops per token (compute)", ylabel, [1, 2, 3, 4], yticks)
    p.frame()
    p.hline(floor, f"floor {floor:g} %", dashed=True, width=1)
    pts = " ".join(f"{p.x(x):.1f},{p.y(y):.1f}" for x, y, _ in ref)
    p.parts.append(f'<polyline points="{pts}" fill="none" stroke="{GREY}" stroke-width="2"/>')
    occupied = []    # boxes (x0, y0, x1, y1) already used by marks and words, so that the point labels can be placed clear of them

    def tbox(x, y, s, anchor, size=10):
        w = 0.58 * size * len(s)
        x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
        return (x0, y - size, x0 + w, y + 2)

    def hit(b):
        return any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in occupied)

    for x, y, name in ref:
        p.parts.append(f'<circle cx="{p.x(x):.1f}" cy="{p.y(y):.1f}" r="4.5" fill="{GREY}" stroke="{SURFACE}" stroke-width="1.5"/>')
        occupied.append((p.x(x) - 5, p.y(y) - 5, p.x(x) + 5, p.y(y) + 5))
        dx, dy = (label_offsets or {}).get(name, (7, -7))
        p.text(p.x(x) + dx, p.y(y) + dy, name, 10, INK2)
        occupied.append(tbox(p.x(x) + dx, p.y(y) + dy, name, "start"))
    occupied.append(tbox(p.W - p.R + 6, p.y(floor) + 4, f"floor {floor:g} %", "start"))
    best_i = min(range(len(evals)), key=lambda k: evals[k][2]) + 1 if evals else None
    for i, (x, y, v) in enumerate(evals, 1):
        p.parts.append(f'<circle cx="{p.x(x):.1f}" cy="{p.y(y):.1f}" r="6" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>')
        occupied.append((p.x(x) - 6, p.y(y) - 6, p.x(x) + 6, p.y(y) + 6))
    lx, ly = legend_at if legend_at else (p.L + 8, p.H - p.B - 34)
    occupied.append((lx - 8, ly - 10, lx + 250, ly + 26))
    # labels on the few points the reading rests on: the best so far and the latest (all of them while there are few), each at the
    # first of a few positions around its point that collides with no mark, word or the legend
    show = set(range(1, len(evals) + 1)) if len(evals) <= 3 else {best_i, len(evals)}
    cands = [(10, 4, "start"), (10, 18, "start"), (10, -9, "start"), (-10, -9, "end"), (-10, 18, "end"), (0, -12, "middle"), (0, 22, "middle"),
             (-10, 4, "end"), (24, 28, "start"), (-24, 28, "end")]
    for i in sorted(show, key=lambda k: (k != best_i, k)):
        x, y, v = evals[i - 1]
        tag = f"#{i}" + (" best" if i == best_i and len(evals) > 1 else "") + (" latest" if i == len(evals) and i != best_i and len(evals) > 3 else "")
        if point_offsets and i in point_offsets:
            dx, dy = point_offsets[i]
            anchor = "end" if dx < 0 else "start"
        else:
            dx, dy, anchor = cands[0]
            for c in cands:
                b = tbox(p.x(x) + c[0], p.y(y) + c[1], tag, c[2])
                if b[0] >= p.L and b[2] <= p.W - p.R + 40 and b[1] >= p.T and b[3] <= p.H - p.B and not hit(b):
                    dx, dy, anchor = c
                    break
        p.text(p.x(x) + dx, p.y(y) + dy, tag, 10, INK, anchor, "600")
        occupied.append(tbox(p.x(x) + dx, p.y(y) + dy, tag, anchor))
    # legend
    p.parts.append(f'<circle cx="{lx}" cy="{ly}" r="6" fill="{BLUE}"/>')
    p.text(lx + 12, ly + 4, "gate found by the GP search (#k = k-th evaluation)", 10, INK)
    p.parts.append(f'<line x1="{lx - 7}" x2="{lx + 7}" y1="{ly + 18}" y2="{ly + 18}" stroke="{GREY}" stroke-width="2"/>')
    p.parts.append(f'<circle cx="{lx}" cy="{ly + 18}" r="4.5" fill="{GREY}"/>')
    p.text(lx + 12, ly + 22, "released model, measured", 10, INK)
    return p.svg()


def best_panel(evals, ref_best, ref_label, full_val, title, sub, yr, yticks):
    p = Panel(title, sub, (0.5, 25.5), yr, "evaluation (budget 25)", "objective (lower is better)", [1, 5, 10, 15, 20, 25], yticks)
    p.frame()
    p.hline(ref_best, ref_label, width=2)
    p.hline(full_val, "full depth", dashed=True, width=1)
    for i, (_, _, v) in enumerate(evals, 1):
        p.parts.append(f'<circle cx="{p.x(i):.1f}" cy="{p.y(v):.1f}" r="3.5" fill="{BLUE}" fill-opacity="0.35"/>')
    best, cur = [], float("inf")
    for e in evals:
        cur = min(cur, e[2])
        best.append(cur)
    if best:
        d = f"M {p.x(1):.1f} {p.y(best[0]):.1f}"
        for i in range(1, len(best)):
            d += f" H {p.x(i + 1):.1f} V {p.y(best[i]):.1f}"
        d += f" H {p.x(len(best) + 0.5):.1f}"
        p.parts.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
        p.text(p.x(len(best) + 0.5) + 6, p.y(best[-1]) + 4, f"best {best[-1]:.1f}", 11, INK, "start", "600")
    lx, ly = p.x(14.5), p.T + 14     # top right: the curve stays left of it while the search is young
    p.parts.append(f'<line x1="{lx - 7}" x2="{lx + 7}" y1="{ly}" y2="{ly}" stroke="{BLUE}" stroke-width="2.5"/>')
    p.text(lx + 12, ly + 4, "GP search, best so far", 10, INK)
    p.parts.append(f'<circle cx="{lx}" cy="{ly + 18}" r="3.5" fill="{BLUE}" fill-opacity="0.35"/>')
    p.text(lx + 12, ly + 22, "each evaluation", 10, INK)
    return p.svg()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/randrouter/gate_nas_progress")
    a = ap.parse_args()
    bf = load("/home/jli199/boptim_scratch/bo_bfcl_gate_nas_gp_s1/evaluations.jsonl", "acc_pct")
    lc = load("/home/jli199/boptim_scratch/bo_lcb_gate_nas_gp_s1/evaluations.jsonl", "pass1_pct")
    bf_ref_best = min(bfcl_value(x, y) for x, y, _ in BFCL_REF)
    lc_ref = lcb_value(2.208, 17.8)
    bf_best = min((e[2] for e in bf), default=float("nan"))
    lc_best = min((e[2] for e in lc), default=float("nan"))
    panels = [
        scatter_panel(bf, BFCL_REF, 81.08, "BFCL v3 (575 search items)", f"{len(bf)} gates evaluated; accuracy against compute",
                      "accuracy (%)", (55, 86), [55, 60, 65, 70, 75, 80, 85],
                      {"gate 0.2": (-52, 4), "gate 0.25": (-58, -7), "gate 0.3": (-16, -10), "gate 0.5": (-14, 18), "gate 0.7": (-16, -10),
                       "full depth": (-58, -10)},
                      legend_at=(Panel.L + 230, Panel.H - Panel.B - 64)),
        best_panel(bf, bf_ref_best, f"released gate {bf_ref_best:.1f}", 100.0, "BFCL: search progress",
                   "objective = 100 x loops / 4 + 5 x points below the floor", (40, 135), [40, 60, 80, 100, 120]),
        scatter_panel(lc, LCB_REF, 18.3, "LiveCodeBench (342 problems)", f"{len(lc)} gates evaluated; pass@1 against compute",
                      "pass@1 (%)", (0, 23), [0, 5, 10, 15, 20],
                      {"depth 1": (8, 14), "depth 2": (-50, -6), "gate 0.3": (8, 14), "full depth": (-58, -10)},
                      legend_at=(Panel.L + 230, Panel.H - Panel.B - 64)),
        best_panel(lc, lc_ref, f"released gate {lc_ref:.1f}", -6.25, "LiveCodeBench: search progress",
                   "objective = -25 / loops + 2 x points below the floor", (-12, 0), [-12, -10, -8, -6, -4, -2, 0]),
    ]
    head = (f"Gate NAS with BO (exit gate trained from scratch, model frozen): no trained gate beats the released gate yet")
    sub = (f"BFCL best {bf_best:.1f} after {len(bf)} of 25 evaluations (released gate {bf_ref_best:.1f}); LiveCodeBench best {lc_best:.2f} "
           f"after {len(lc)} of 25 (released gate {lc_ref:.2f}). Lower objective is better.")
    note = ("Blue = one gate (architecture + Stage-II gamma + exit threshold) trained from a random initialisation and decoded with real "
            "early exit. Grey = the released model at its own thresholds (and fixed depth), measured the same way.")
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:{SURFACE};{FONT};color:{INK}}}'
            f'.wrap{{padding:20px 24px;width:1230px}} h1{{font-size:17px;margin:0 0 4px}} p{{margin:0 0 10px;color:{INK2};font-size:13px}}'
            f'.grid{{display:grid;grid-template-columns:600px 600px;gap:16px 24px}} .note{{font-size:12px;margin-top:8px}}</style></head>'
            f'<body><div class="wrap"><h1>{head}</h1><p>{sub}</p><div class="grid">{"".join(panels)}</div><p class="note">{note}</p></div></body></html>')
    open(a.out + ".html", "w").write(html)
    subprocess.run(["timeout", "60", "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--window-size=1280,980", f"--screenshot={a.out}.png", f"file://{a.out}.html"], capture_output=True)
    print("wrote", a.out + ".html", a.out + ".png", os.path.exists(a.out + ".png"))


if __name__ == "__main__":
    main()
