"""2026-11-20 block: why the router searches cannot beat the released gate on LiveCodeBench, and what repairs it
(random-router training, user idea of 2026-11-20; calibrated / trained cache fill; Stage-II refit; proxy searches).

Reads the artefacts of the worktree's randrouter/ scripts under boptim_scratch/randrouter:
  tables/<system>.json             collect_tables.py   teacher-forced fidelity of routers, decoder's cache and exact cache
  tables/base_gate_signal.json     gate_signal.py      what the released gate knows after each loop
  fill_maps_lcb_n24.pt.report.json fit_fill_maps.py    relative error of copied / mapped keys and values per pair of loops
  gates_base_exact/frontier.json   stage2.py           refitted gates against the released one
  bo_tf/<name>/{baseline.json, evaluations.jsonl}       proxy searches (tf_bo.sh)
  real/<tag>/record.json           lcb_real_eval*.py   real execution on the 342 test problems
Every part is left out while its file does not exist.
"""
import glob
import json
import os

from page_blocks_1107 import _chip, _sm

R = "/home/jli199/boptim_scratch/randrouter"
SYSTEMS = [  # (table file, display name, note)
    ("base", "released model, plain copy fill", "what every search so far ran on"),
    ("fill24", "released model, calibrated fill", "affine maps fitted by regression on 24 training answers; no training"),
    ("r2kl125", "random-router training, 125 steps", "LoRA rank 32, distilled from the base model's exact-cache readouts; plain copy fill"),
    ("r3_kl_fill", "random-router training + calibrated fill", "LoRA rank 32 trained with the calibrated fill in place"),
    ("r3_maps", "trained fill maps, model untouched", "the maps trained under the random router; full depth cannot change"),
]
ROUTERS = [("0.5", "gate 0.5"), ("0.3", "gate 0.3"), ("0.25,0.3,0.3", "(0.25, 0.3, 0.3)"), ("0.245,0.3,0.3", "(0.245, 0.3, 0.3)"),
           ("0.2", "gate 0.2"), ("0.25,1.0,1.0", "(0.25, 1, 1): loop-1 exits only"), ("0.2,0.4,0.7", "(0.2, 0.4, 0.7)"),
           ("rank:0.5,0.5,0.5", "surest half leaves after every loop")]
# real execution of the released model with the plain copy fill, from the earlier searches (342 problems)
BASE_REAL = [("full depth", "4.00", 20.5), ("gate 0.7", "—", 20.5), ("gate 0.5", "—", 19.3), ("gate 0.3", "2.1", 18.1),
             ("(0.25, 0.3, 0.3), first proposal of both LLM optimizers", "2.02", 15.2)]


def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def _pct(v, d=2):
    return "—" if v is None else f"{100 * v:.{d}f}"


def _diagnosis():
    t = _load(f"{R}/tables/base.json")
    if not t:
        return ""
    rows = []
    for spec, disp in ROUTERS:
        r = t["routers"].get(spec)
        if not r:
            continue
        gap = 100 * (r["agree_exact"] - r["agree_real"])
        share1 = round(100 * r["share"][0])
        rows.append(f'<tr><td>{disp}</td><td class="n">{r["loops"]:.2f}</td><td class="n">{share1}</td><td class="n">{_pct(r["agree_real"])}</td>'
                    f'<td class="n">{_pct(r["agree_exact"])}</td><td class="n">{gap:.2f}</td></tr>')
    full = t["routers"].get("full")
    rep = _load(f"{R}/fill_maps_lcb_n24.pt.report.json")
    kv = ""
    if rep:
        def mean(name):
            v = rep["report"][name]
            return sum(v["copy"]) / len(v["copy"]), sum(v["affine"]) / len(v["affine"])
        k12, k23, k34 = mean("keys 1->2"), mean("keys 2->3"), mean("keys 3->4")
        v12, v23 = mean("values 1->2"), mean("values 2->3")
        kv = (f' The copies themselves show it: measured against the keys the skipped loop would have computed, a copied loop-1 key has a relative error of '
              f'{k12[0]:.2f} (a constant would score 1.0) and a copied loop-1 value {v12[0]:.2f}, while loop-2 keys stand in for loop-3 keys at {k23[0]:.2f} and '
              f'loop-3 for loop-4 at {k34[0]:.2f}. An affine map per layer and head brings the loop-1 errors to {k12[1]:.2f} (keys) and {v12[1]:.2f} (values).')
    sig = _load(f"{R}/tables/base_gate_signal.json")
    gs = ""
    if sig:
        l1 = sig["loops"][0]
        q = l1["cdf_quantiles_5_25_50_75_95"]
        gs = (f' The gate is not the problem: after loop 1 its cumulative exit probability already ranks the tokens (AUC {l1["auc_cdf"]:.2f} for '
              f'"the loop-1 readout is the full-depth token"; among the half it is surest about, {100 * l1["safe_among_top"]["50"]:.1f} % are right), but it is squeezed '
              f'under 0.25 (median {q[2]:.3f}, 95th percentile {q[4]:.3f}), so only thresholds in a window of about 0.01 below 0.25 use loop 1 at all.')
    return f'''<p style="margin:14px 0 6px;font-weight:600">1 · Diagnosis: a token that leaves after loop 1 is read correctly, but poisons the cache</p>
    <p class="src">The released model answers {t["n_seqs"]} LiveCodeBench problems that are not test problems at full depth; its own answers ({t["n_tokens"]:,} tokens) are then
    teacher-forced under a router, once with the decoder's cache (a token that left after loop s has copies of its loop-s keys and values in the slots of the loops it skipped)
    and once with the same exits but every token's exact keys and values. Agreement = share of answer tokens whose exit readout is the full-depth token
    (full depth against itself: {_pct(full["agree_real"]) if full else "—"} %, the rest is numerical noise).</p>
    <table>
      <thead><tr><th>Router (thresholds after loops 1, 2, 3)</th><th class="n">loops per token</th><th class="n">% leaving after loop 1</th>
      <th class="n">agreement %, decoder's cache</th><th class="n">agreement %, exact cache</th><th class="n">lost to the cache, points</th></tr></thead>
      <tbody>
        {chr(10).join(rows)}
      </tbody>
    </table>
    <p class="src">Reading. As long as no token leaves after loop 1 (gates 0.5 and 0.3) the cache costs a few tenths of a point. As soon as tokens leave after loop 1 the
    readouts of those tokens are still right (the exact-cache column does not move), yet the later tokens, which attend to the copied loop-1 keys and values in loops 2 to 4,
    lose 2 to 5 points.{kv}{gs} This is why no threshold search over the released gate gets below about 2.1 loops: the region below is closed by the cache, not by the choice of tokens.</p>'''


def _repairs():
    tabs = [(name, disp, note, _load(f"{R}/tables/{name}.json")) for name, disp, note in SYSTEMS]
    tabs = [x for x in tabs if x[3]]
    if len(tabs) < 2:
        return ""
    specs = [("full", "full depth"), ("0.3", "gate 0.3"), ("0.245,0.3,0.3", "(0.245, 0.3, 0.3)"), ("0.2", "gate 0.2"), ("rank:0.5,0.5,0.5", "surest half per loop")]
    head = "".join(f'<th class="n">{d}</th>' for _, d in specs)
    rows = []
    for name, disp, note, t in tabs:
        cells = []
        for spec, _ in specs:
            r = t["routers"].get(spec)
            cells.append(f'<td class="n">{_pct(r["agree_real"])}{_sm(" @ %.2f" % r["loops"]) if spec != "full" else ""}</td>' if r else '<td class="n">—</td>')
        rows.append(f'<tr><td>{disp}{_sm("<br>" + note)}</td>{"".join(cells)}</tr>')
    return f'''<p style="margin:16px 0 6px;font-weight:600">2 · Repairs: agreement with the base model's full-depth answers (decoder's cache), and the loops it takes</p>
    <table>
      <thead><tr><th>System</th>{head}</tr></thead>
      <tbody>
        {chr(10).join(rows)}
      </tbody>
    </table>
    <p class="src">"@" = loops per token under that router. Random-router training (the user's proposal): every answer token gets an exit loop from a router drawn at
    random (independent draws, or the tokens the gate is surest about at random shares), the cache is filled as the decoder fills it, and a LoRA update is trained so that the
    readout of every loop a token ran matches the released model's readout of the same loop with an exact cache. A first version with hard labels (cross-entropy to the full-depth
    token) also learned the cache but moved the model: full-depth agreement fell to 98.0 %. The full-depth column is the price of a repair: a system that changes the
    model pays it at every router; the two fill-only systems cannot.</p>'''


def _stage2():
    f = _load(f"{R}/gates_base_exact/frontier.json")
    if not f:
        return ""
    names = [("released", "released gate"), ("paper", "refit, the paper's Stage-II rule"), ("next", "refit, “will the next loop help”"), ("correct", "refit, “readout already right”")]
    specs = [s for s in ("rank:0.5,0.5,0.5", "rank:0.7,0.5,0.5", "rank:0.8,0.5,0.5", "rank:0.9,0.5,0.5") if s in f.get("released", {})]
    head = "".join(f'<th class="n">{round(100 * float(s.split(":")[1].split(",")[0]))} % leave after loop 1{_sm("<br>%.2f loops" % f["released"][s]["loops"])}</th>' for s in specs)
    rows = []
    for key, disp in names:
        if key not in f:
            continue
        rows.append(f'<tr><td>{disp}</td>' + "".join(f'<td class="n">{_pct(f[key][s]["agree_exact"])}</td>' if s in f[key] else '<td class="n">—</td>' for s in specs) + '</tr>')
    return f'''<p style="margin:16px 0 6px;font-weight:600">3 · Stage II again: refitting the gate alone changes nothing</p>
    <table>
      <thead><tr><th>Gate (a linear map of the state, as released)</th>{head}</tr></thead>
      <tbody>
        {chr(10).join(rows)}
      </tbody>
    </table>
    <p class="src">Agreement % with an exact cache when the tokens each gate is surest about leave (the same share for every gate, so the gates compare at equal loops). The three
    refits start from zero weights and are fitted with the language model frozen on 164 training answers. The released gate already ranks the tokens about as well as a linear
    read-out of these states can; the paper's rule, which gives the gate after loop 1 no target, is the weakest. The gain has to come from the cache, not from the gate.</p>'''


def _proxy():
    rows = []
    for name, disp in (("base_copy", "released model, plain copy"), ("fill24", "released model, calibrated fill"), ("r2kl125", "random-router training, 125 steps"),
                       ("r3_kl_fill", "random-router training + calibrated fill"), ("r3_maps", "trained fill maps")):
        cells = []
        for suffix in ("", "_strict"):
            b = _load(f"{R}/bo_tf/{name}{suffix}/baseline.json")
            p = f"{R}/bo_tf/{name}{suffix}/evaluations.jsonl"
            if not b or not os.path.exists(p):
                cells.append(None)
                continue
            ev = [json.loads(l) for l in open(p) if l.strip()]
            bk = min(b["single_knob"], key=lambda t: t["value"])
            be = min(ev, key=lambda r: r["value"])
            cells.append((bk, be, len(ev)))
        if not any(cells):
            continue

        def fmt(c):
            if not c:
                return '<td class="n">—</td><td class="n">—</td>'
            bk, be, n = c
            return (f'<td class="n">{bk["value"]:.2f}{_sm("<br>q = %g: %.2f loops, %.2f %%" % (bk["q"], bk["loops"], bk["agree_pct"]))}</td>'
                    f'<td class="n">{be["value"]:.2f}{_sm("<br>(%s): %.2f loops, %.2f %%" % (", ".join("%.2f" % v for v in be["thresholds"]), be["loops"], be["agree_pct"]))}</td>')
        rows.append(f'<tr><td>{disp}</td>{fmt(cells[0])}{fmt(cells[1])}</tr>')
    if not rows:
        return ""
    return f'''<p style="margin:16px 0 6px;font-weight:600">4 · The search on a proxy objective: 15 seconds per evaluation instead of 3 hours</p>
    <table>
      <thead><tr><th rowspan="2">System</th><th colspan="2" class="n">objective of the real searches</th><th colspan="2" class="n">no loss allowed (floor = gate 0.3 of the released model)</th></tr>
      <tr><th class="n">best single knob</th><th class="n">search, 50 evaluations</th><th class="n">best single knob</th><th class="n">search, 50 evaluations</th></tr></thead>
      <tbody>
        {chr(10).join(rows)}
      </tbody>
    </table>
    <p class="src">Proxy: one teacher-forced pass over the validation answers under the candidate thresholds with the decoder's cache; value = −6.25 × 4 / loops + penalty ×
    max(0, floor − agreement %), minimised. Left: the real searches' units (penalty 3.5 per agreement point below 98.45 %, the real 2.0 per pass@1 point at the observed slope).
    Right: penalty 20 below 98.22 %, i.e. the fastest router that loses nothing against the released gate at 0.3. Gaussian-process search (boptim, gp_hedge) warm-started with
    the single knob at seven thresholds. On the released model the proxy reproduces the real result: the search ends at about 2.2 loops with no loop-1 exit and
    −11.25, where the real Gaussian-process run ended at (1.00, 0.38, 0.30) and −11.10.</p>'''


def _real():
    recs = []
    for p in sorted(glob.glob(f"{R}/real/*/record.json")):
        r = _load(p)
        if r and r.get("n", 0) >= 300:
            recs.append(r)
    rows = [f'<tr><td>released model, plain copy</td><td>{d}</td><td class="n">{l}</td><td class="n">{a:.1f}</td></tr>' for d, l, a in BASE_REAL]
    for r in recs:
        sysname = "random-router training, 125 steps" if "r2_kl_s125" in r["ckpt"] else ("released model, calibrated fill" if "exitlab" in r["ckpt"] else os.path.basename(r["ckpt"]))
        thr = r["thresholds"][:3]
        d = "full depth" if min(thr) >= 1.0 else "(" + ", ".join(f"{v:g}" for v in thr) + ")"
        rows.append(f'<tr class="best"><td>{sysname}</td><td>{d}</td><td class="n">{r["exec_loops"]:.2f}</td><td class="n">{r["pass1_pct"]:.1f}</td></tr>')
    note = "" if recs else " The evaluations of the repaired systems are running."
    return f'''<p style="margin:16px 0 6px;font-weight:600">5 · Real execution on the 342 test problems</p>
    <table>
      <thead><tr><th>System</th><th>Router</th><th class="n">executed loops per token</th><th class="n">pass@1 %</th></tr></thead>
      <tbody>
        {chr(10).join(rows)}
      </tbody>
    </table>
    <p class="src">Greedy decoding with true early exit, official tests in a sandbox, as in the searches. pass@1 on 342 problems has a standard error of about 2.2 points.{note}</p>'''


def randrouter_block():
    parts = [_diagnosis(), _repairs(), _stage2(), _proxy(), _real()]
    if not any(parts):
        return ""
    return f'''<div class="sub-head"><h3>Why the searches cannot beat the released gate, and what repairs it (2026-11-20)</h3><p>Question from the user: change the router
    into a random one and retrain Stage II, to get past “the search cannot beat the baseline”. Short answer: the baseline is not protected by a good gate but by a bad cache.
    A token that leaves after loop 1 poisons the cache for every later token, so every router that would use loop 1 loses accuracy, whatever its thresholds. Training under a
    random router with the decoder's cache fill repairs most of that; refitting the gate (Stage II) alone does not.</p></div>
  <div class="tablewrap">
    {"".join(p for p in parts if p)}
  </div>
'''
