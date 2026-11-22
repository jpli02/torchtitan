"""Page block of 2026-11-21 (imported by page_update_1102.py): the gpt-6-luna threshold search on LiveCodeBench against the
Gaussian process (and why its lead is timing noise), and the gate NAS with BO (Stage-II gate trained from scratch per evaluation)
on LiveCodeBench and BFCL. Everything is read from the result files at build time, so a rebuild shows the searches' current state.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "randrouter"))
import plot_gate_nas as pg  # noqa: E402

LUNA = "/home/jli199/boptim_scratch/bo_lcb_router_pooled_luna6/evaluations.jsonl"
GNAS_LCB = "/home/jli199/boptim_scratch/bo_lcb_gate_nas_gp_s1/evaluations.jsonl"
GNAS_BFCL = "/home/jli199/boptim_scratch/bo_bfcl_gate_nas_gp_s1/evaluations.jsonl"
GNAS_LCB_LLM = "/home/jli199/boptim_scratch/bo_lcb_gate_nas_luna_s1/evaluations.jsonl"
GNAS_BFCL_LLM = "/home/jli199/boptim_scratch/bo_bfcl_gate_nas_luna_s1/evaluations.jsonl"
COLORS = {pg.SURFACE: "var(--surface)", pg.INK: "var(--text)", pg.INK2: "var(--text-2)", pg.GRID: "var(--line)", pg.AXIS: "var(--line-2)",
          pg.BLUE: "var(--s1)", pg.GREY: "var(--base)"}


def _sm(s):
    return f'<span style="font-size:11px;color:var(--text-3)">{s}</span>'


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def _themed(svg):
    for k, v in COLORS.items():
        svg = svg.replace(k, v)
    return svg.replace(f'width="{pg.Panel.W}" height="{pg.Panel.H}"', f'width="100%" style="max-width:{pg.Panel.W}px;height:auto;display:block"')


def _luna_block():
    rows = [r for r in _rows(LUNA) if r.get("status") == "ok"]
    if not rows:
        return ""
    # answers identical to the best evaluation (the runs' generations compared problem by problem)
    best = min(rows, key=lambda r: r["value"])

    def gens(r):
        p = os.path.join(r.get("run_dir", ""), "gen.jsonl")
        if not os.path.exists(p):
            return None
        return {g["question_id"]: g["gen"] for g in (json.loads(l) for l in open(p)) if "question_id" in g}

    gb = gens(best)
    trs = []
    for k, r in enumerate(rows, 1):
        t = r["thresholds"][:3]
        g = gens(r)
        same = sum(g[q] == gb[q] for q in gb if q in g) if (g is not None and gb is not None) else None
        same_cell = "—" if same is None else (f"{same} of {len(gb)}" + (" (the best)" if r is best else ""))
        cls = ' class="best"' if r is best else ""
        trs.append(f'<tr{cls}><td class="n">{k}</td><td>({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})</td><td class="n">{r["pass1_pct"]:.1f}</td>'
                   f'<td class="n">{r.get("exec_loops") or 0:.2f}</td><td class="n">{r["speedup"]:.2f}×{"" if r.get("cal_clean") else _sm("<br>shared card")}</td>'
                   f'<td class="n">{r["value"]:.2f}</td><td class="n">{same_cell}</td></tr>')
    return f'''<p style="margin:16px 0 6px;font-weight:600">1 · gpt-6-luna against the Gaussian process on the LiveCodeBench threshold search: the lead is timing</p>
    <p class="src">The LLM agent (gpt-6-luna through OpenRouter, the same 7 warm-start records and the same objective as the Gaussian-process run:
    value = −6.25 × measured speed-up + 2 × points of pass@1 below 18.3 %) ran 10 new evaluations on all 342 problems. Its best, {best["value"]:.2f}, is
    nominally below the Gaussian process's −11.10 after 22 evaluations. But four of its evaluations changed only q1 between 0.38 and 0.45, where the
    gate's cumulative exit probability after loop 1 never lands, and produced the same answers on every problem: they differ only in the measured
    speed-up of a shared card, and so in value by more than a full unit. The agent read that noise as signal and kept refining the same router.
    Searches since then count executed loops per token instead of timing them.</p>
    <table>
      <thead><tr><th class="n">#</th><th>router (q1, q2, q3)</th><th class="n">pass@1 %</th><th class="n">executed<br>loops</th>
      <th class="n">measured<br>speed-up</th><th class="n">objective</th><th class="n">answers identical<br>to the best run</th></tr></thead>
      <tbody>
        {chr(10).join(trs)}
      </tbody>
    </table>'''


def _gnas_table(rows, bench):
    trs = []
    best = min(rows, key=lambda r: r["value"]) if rows else None
    for k, r in enumerate(rows, 1):
        if bench == "bfcl":
            name = (r.get("extra_cfg") or {}).get("decode_router", "").rsplit("/", 1)[-1].replace(".pt", "")
            arch, g = (name.rsplit("_g", 1) + [""])[:2]
            gamma = g.split("_s")[0] if g else ""
            acc, loops = r.get("acc_pct"), r.get("exec_loops")
        else:
            arch, gamma = r.get("router_arch", ""), f'{r.get("gamma", 0):.4f}'
            acc, loops = r.get("pass1_pct"), r.get("loops_eff")
        note = _sm(" scored as " + r["shortcut"]) if r.get("shortcut") else ""
        cls = ' class="best"' if r is best else ""
        trs.append(f'<tr{cls}><td class="n">{k}</td><td>{arch}{note}</td><td class="n">{gamma}</td><td class="n">{r["thresholds"][0]:.3f}</td>'
                   f'<td class="n">{"—" if acc is None else f"{acc:.1f}"}</td><td class="n">{"—" if loops is None else f"{loops:.2f}"}</td>'
                   f'<td class="n">{r["value"]:.2f}</td></tr>')
    return "\n        ".join(trs)


def _gnas_block():
    bf = [r for r in _rows(GNAS_BFCL) if r.get("status") in ("ok", "shortcut")]
    lc = [r for r in _rows(GNAS_LCB) if r.get("status") in ("ok", "shortcut")]
    if not bf and not lc:
        return ""
    bfe = pg.load(GNAS_BFCL, "acc_pct")
    lce = pg.load(GNAS_LCB, "pass1_pct")
    bf_ref_best = min(pg.bfcl_value(x, y) for x, y, _ in pg.BFCL_REF)
    lc_ref = pg.lcb_value(2.208, 17.8)
    bf_best = min((e[2] for e in bfe), default=float("nan"))
    lc_best = min((e[2] for e in lce), default=float("nan"))
    beat_bf, beat_lc = bf_best < bf_ref_best, lc_best < lc_ref
    verdict = ("a trained gate beats the released gate on " + " and ".join(b for b, ok in (("BFCL", beat_bf), ("LiveCodeBench", beat_lc)) if ok)
               if (beat_bf or beat_lc) else "no trained gate beats the released gate yet")
    panels = [
        pg.scatter_panel(bfe, pg.BFCL_REF, 81.08, "BFCL v3 (575 search items)", f"{len(bfe)} gates evaluated; accuracy against compute",
                         "accuracy (%)", (55, 86), [55, 60, 65, 70, 75, 80, 85],
                         {"gate 0.2": (-52, 4), "gate 0.25": (-58, -7), "gate 0.3": (-16, -10), "gate 0.5": (-14, 18), "gate 0.7": (-16, -10),
                          "full depth": (-58, -10)}, legend_at=(pg.Panel.L + 230, pg.Panel.H - pg.Panel.B - 64)),
        pg.best_panel(bfe, bf_ref_best, f"released gate {bf_ref_best:.1f}", 100.0, "BFCL: search progress",
                      "objective = 100 x loops / 4 + 5 x points below the floor", (40, 135), [40, 60, 80, 100, 120]),
        pg.scatter_panel(lce, pg.LCB_REF, 18.3, "LiveCodeBench (342 problems)", f"{len(lce)} gates evaluated; pass@1 against compute",
                         "pass@1 (%)", (0, 23), [0, 5, 10, 15, 20],
                         {"depth 1": (8, 14), "depth 2": (-50, -6), "gate 0.3": (8, 14), "full depth": (-58, -10)},
                         legend_at=(pg.Panel.L + 230, pg.Panel.H - pg.Panel.B - 64)),
        pg.best_panel(lce, lc_ref, f"released gate {lc_ref:.1f}", -6.25, "LiveCodeBench: search progress",
                      "objective = -25 / loops + 2 x points below the floor", (-12, 0), [-12, -10, -8, -6, -4, -2, 0]),
    ]
    fig = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:12px 20px;margin:8px 0 4px">'
           + "".join(f"<div style=\"min-width:0\">{_themed(p)}</div>" for p in panels) + "</div>")
    llm = []
    for path, name in ((GNAS_BFCL_LLM, "BFCL"), (GNAS_LCB_LLM, "LiveCodeBench")):
        rr = [r for r in _rows(path) if r.get("status") in ("ok", "shortcut")]
        if rr:
            llm.append(f"{name} LLM agent: {len(rr)} evaluations, best {min(r['value'] for r in rr):.2f}")
    llm_txt = ("; ".join(llm) + ".") if llm else ("The LLM-agent arms (gpt-6-luna through the OpenAI API, same objective and budget) start on each card "
                                                 "after its Gaussian-process run.")
    return f'''<p style="margin:16px 0 6px;font-weight:600">2 · Gate NAS with BO: the gate's SFT searched like the HumanEval experiment, on BFCL and LiveCodeBench</p>
    <p class="src">Asked on 21 November: search the gate's training hyperparameters with BO, as in the HumanEval router-NAS experiment. Each evaluation
    trains an exit gate from a random initialisation with the Stage-II loss while the model stays frozen (about 10 seconds on cached states of the
    released model's own full-depth answers: 172 LiveCodeBench v1–v4 training problems, 600 BFCL live items; none is evaluated), then serves the
    released model with that gate under real decode-time early exit. Search space (as in HumanEval): gate depth 0 / 1 / 2, width 64 / 128 / 256,
    activation, input LayerNorm, Stage-II gamma in [0, 1], one exit threshold in [0.1, 1]; Gaussian process (skopt gp_hedge), 25 evaluations, no warm
    start, cost = executed loops. So far <b>{verdict}</b>: BFCL best {bf_best:.1f} after {len(bfe)} evaluations (released gate at its best threshold
    {bf_ref_best:.1f}), LiveCodeBench best {lc_best:.2f} after {len(lce)} (released gate at 0.3, measured with the exact loop counter, {lc_ref:.2f}).
    The Gaussian process starts from skopt's grid design, which in six dimensions samples only gamma 1/3 and 2/3 and thresholds 0.4 and 0.7: those gates
    exit too early, and the region of the released gate (gamma near Ouro's default 0.005) is a thin strip at the edge of the gamma axis that the search
    has to reach by itself. {llm_txt}</p>
    {fig}
    <p class="src">Left: accuracy against executed loops per token; blue = the gates the search evaluated (#k = k-th evaluation), grey = the released model
    at its own thresholds and at fixed depth, measured the same way. Right: best value so far (lower is better) against the released gate's best value.</p>
    <table>
      <thead><tr><th class="n">#</th><th>BFCL gate</th><th class="n">gamma</th><th class="n">threshold</th><th class="n">accuracy %</th>
      <th class="n">executed<br>loops</th><th class="n">objective</th></tr></thead>
      <tbody>
        {_gnas_table(bf, "bfcl")}
        <tr><td class="n">—</td><td>released gate (best threshold 0.25)</td><td class="n">—</td><td class="n">0.250</td><td class="n">80.9</td>
        <td class="n">1.91</td><td class="n">{bf_ref_best:.2f}</td></tr>
      </tbody>
    </table>
    <table>
      <thead><tr><th class="n">#</th><th>LiveCodeBench gate</th><th class="n">gamma</th><th class="n">threshold</th><th class="n">pass@1 %</th>
      <th class="n">executed<br>loops</th><th class="n">objective</th></tr></thead>
      <tbody>
        {_gnas_table(lc, "lcb")}
        <tr><td class="n">—</td><td>released gate</td><td class="n">—</td><td class="n">0.300</td><td class="n">17.8</td>
        <td class="n">2.21</td><td class="n">{lc_ref:.2f}</td></tr>
      </tbody>
    </table>
    <p class="src">BFCL objective = 100 × executed loops / 4 + 5 × points of accuracy below 81.08 % (full depth = 100). LiveCodeBench objective = −6.25 × 4 /
    executed loops + 2 × points of pass@1 below 18.3 % (full depth = −6.25); a LiveCodeBench gate that sends 95 % of the training tokens out after loop 1
    is scored with the measured fixed-depth-1 run, one that lets almost none leave with the measured full-depth run. Code:
    <code>boptim_lcb/ouro_lcb_gate_nas.py</code>, <code>ouro_bfcl_gate_nas.py</code>, serving directory with a swappable gate
    <code>boptim_lcb/make_gatelab_modeling.py</code> (the released gate loaded through it decodes identically).</p>'''


def gate_nas_block():
    parts = [_luna_block(), _gnas_block()]
    if not any(parts):
        return ""
    return f'''<div class="sub-head"><h3>LLM agent against BO on LiveCodeBench, and BO over the gate's training (2026-11-21)</h3><p>The gpt-6-luna threshold
    search ended nominally ahead of the Gaussian process, but on measurement noise: it evaluated one router four times. The gate searches that followed
    (the HumanEval experiment on Ouro-2.6B, BFCL and LiveCodeBench) are running; the numbers below are read at build time.</p></div>
  <div class="tablewrap">
    {"".join(p for p in parts if p)}
  </div>
'''
