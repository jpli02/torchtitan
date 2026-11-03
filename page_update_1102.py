"""Rebuild the experiments page (Q12) from the version-48 source with: BigCodeBench true-exit results, the
LiveCodeBench direct-mode block, GSM-Symbolic rows, the 2025-benchmark citation table and updated settings.
Re-runnable as results land: always starts from TMP/boptim-experiments.v48.html (copied once from the
boptim_scratch source) and re-reads every result file.
Usage: python page_update_1102.py [--publish-copy]  -> writes TMP/boptim-experiments.html and updates the source."""
import json
import os
import shutil
import sys

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
SRC = "/home/jli199/boptim_scratch/boptim-experiments.html"
BASE = f"{TMP}/boptim-experiments.v48.html"
OUT = f"{TMP}/boptim-experiments.html"
if not os.path.exists(BASE):
    shutil.copy(SRC, BASE)
html = open(BASE).read()


def rep(old, new):
    global html
    assert html.count(old) == 1, (html.count(old), old[:90])
    html = html.replace(old, new)


def graded(name, label):
    p = f"{TMP}/bench_{name}_{label}.graded.json"
    return json.load(open(p)) if os.path.exists(p) else None


def result_line(name, label):
    p = f"{TMP}/bench_{name}.txt"
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith(f"RESULT {label} "):
            return json.loads(line[len(f"RESULT {label} "):])
    g = graded(name, label)  # the generation summary also sits in the graded file (used when the RESULT line was lost)
    return g.get("summary") if g else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def chip(kind, text):
    return f'<span class="chip {kind}">{text}</span>'


def sm(text):
    return f'<span class="sm">{text}</span>'


# ---------------------------------------------------------------- LiveCodeBench direct block
LCB = "lcb342"
lcb_pts = [("k4", "full depth, 4 loops", "4.00"), ("k2", "depth 2", "2.00"), ("q0.5", "released gate, q = 0.5", None),
           ("q0.3", "released gate, q = 0.3", None), ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)", None), ("k1", "depth 1", "1.00")]
k4g = graded(LCB, "k4")
k4acc = k4g["pass@1"] if k4g else None
rows = []
for label, disp, loops_fixed in lcb_pts:
    g, r = graded(LCB, label), result_line(LCB, label)
    cls = ' class="dim"' if label == "k4" else ""
    if g is None:
        rows.append(f'<tr{cls}><td>{disp}</td><td class="n">running</td><td class="n">—</td><td class="n">—</td><td class="n">—</td>'
                    f'<td class="n">—</td><td class="n">—</td><td class="st">{chip("info", "running")}</td></tr>')
        continue
    bd = g["by_difficulty"]
    loops = loops_fixed or (f"{r['mean_loops']:.2f}" if r else "—")
    toks = f"{r['mean_tokens']:.0f}" if r else "—"
    if label == "k4":
        reading = "baseline" + sm(f"{pct(r['trunc_rate'])} % hit the 1,024-token budget; {g['no_code']} answers without a code block") if r else "baseline"
    else:
        d = 100 * (g["pass@1"] - k4acc) if k4acc is not None else None
        if d is None:
            reading = ""
        elif abs(d) <= 2.5:
            reading = chip("good", f"{d:+.1f} pt, within noise")
        elif d < 0:
            reading = chip("crit" if d < -5 else "warn", f"{d:+.1f} pt")
        else:
            reading = chip("good", f"{d:+.1f} pt")
        if label == "k1":
            reading += sm("one loop: the floor the oracle bound cannot go below")
    best = ' class="best"' if (label != "k4" and k4acc is not None and abs(g["pass@1"] - k4acc) <= 0.025 and label.startswith(("q", "steps"))
                               and r and r.get("mean_loops", 4) < 3.0) else cls
    rows.append(f'<tr{best}><td>{disp}</td><td class="n">{pct(g["pass@1"])} %</td><td class="n">{bd.get("easy", "—")}</td>'
                f'<td class="n">{bd.get("medium", "—")}</td><td class="n">{bd.get("hard", "—")}</td><td class="n">{loops}</td>'
                f'<td class="n">{toks}</td><td class="st">{reading}</td></tr>')
orc_p = f"{TMP}/bench_{LCB}_oracle.json"
orc = json.load(open(orc_p)) if os.path.exists(orc_p) else None
orc_txt = (f"Oracle floor {orc['oracle_mean_loops']:.2f} loops; {pct(orc['uniform_step_agreement_with_final'][0], 0)} % of tokens settle at loop 1."
           if orc else "Oracle bound running.")
done = [l for l, _, _ in lcb_pts if graded(LCB, l)]
k4_note = ""
if k4g:
    bd = k4g["by_difficulty"]
    k4_note = (f" At full depth the direct answer passes {pct(k4acc)} % of the 342 problems (easy {bd.get('easy')}, medium {bd.get('medium')}, "
               f"hard {bd.get('hard')}), against 28.3 % on the 120 easiest problems in thinking mode where 83 of 120 answers overran a 3,072-token budget.")
lcb_block = f'''<div class="sub-head"><h3>LiveCodeBench, direct mode (ICLR 2025, releases 5 and 6, 342 problems, official tests)</h3><p>The most-cited coding benchmark with a 2025 venue. Same forced code-block answer as BigCodeBench (1,024-token budget, batch 6, gather-mode selection); every problem's full test set runs in a network-less container. 84 easy, 104 medium, 154 hard problems from September 2024 to April 2025.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Point</th><th class="n">pass@1 (342)</th><th class="n">easy (84)</th><th class="n">medium (104)</th><th class="n">hard (154)</th><th class="n">loops</th><th class="n">mean tokens</th><th class="st">Reading</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">Standard error about 2.5 points at n = 342. {orc_txt}{k4_note} Points not yet listed are running on two cards (gate and per-step points on one, depth 2 and the oracle on the other); each is about 2 hours of generation plus grading.</p>
  </div>
'''

# ---------------------------------------------------------------- BigCodeBench true-exit table
bcb_block = '''<div class="sub-head"><h3>BigCodeBench under real execution, batch 1, 150 full-set tasks</h3><p>Same decode-time early exit as the CRUXEval table below; direct mode, 768-token budget, graded in the official image.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Router</th><th class="n">pass@1</th><th class="n">tokens / s</th><th class="n">speed-up</th><th class="n">mean tokens</th><th class="st">Reading</th></tr></thead>
      <tbody>
        <tr class="dim"><td>full depth, 4 loops</td><td class="n">40.7 %</td><td class="n">6.43</td><td class="n">1.00×</td><td class="n">278</td><td class="st">baseline</td></tr>
        <tr class="best"><td>released gate, q = 0.5</td><td class="n">40.0 %</td><td class="n">8.79</td><td class="n">1.37×</td><td class="n">255</td><td class="st"><span class="chip good">no loss</span><span class="sm">−0.7 pt, inside the 4-point standard error</span></td></tr>
        <tr><td>released gate, q = 0.3</td><td class="n">37.3 %</td><td class="n">11.65</td><td class="n">1.81×</td><td class="n">245</td><td class="st"><span class="chip warn">−3.3 pt</span></td></tr>
      </tbody>
    </table>
    <p class="src">Standard error about 4 points at n = 150. Code-only answers of about 250 tokens make the skipped-loop cache approximation almost free: the gate at q = 0.5 keeps full-depth accuracy at 1.37× real speed, where CRUXEval needed q = 0.7 for no loss at 1.12×. The loop counter under true exit does not report the executed count (it reads 3.85 and 3.14 here against 2.82 and 2.09 for the same thresholds in gather mode), so speed is the measured tokens per second.</p>
  </div>
'''

# ---------------------------------------------------------------- LiveCodeBench true-exit (wall-clock) table
def wc_result(label):
    p = f"{TMP}/wallclock_lcb.txt"
    if not os.path.exists(p):
        return None
    for line in open(p):
        if line.startswith(f"RESULT {label} "):
            return json.loads(line[len(f"RESULT {label} "):])
    return None


def wc_graded(label):
    p = f"{TMP}/wcl_{label}.graded.json"
    return json.load(open(p)) if os.path.exists(p) else None


def wc_rows(pts, full_label):
    wfull, gfull = wc_result(full_label), wc_graded(full_label)
    rows = []
    for label, disp in pts:
        r, g = wc_result(label), wc_graded(label)
        cls = ' class="dim"' if label == full_label else ""
        if r is None or g is None:
            rows.append(f'<tr{cls}><td>{disp}</td><td class="n">running</td><td class="n">—</td><td class="n">—</td><td class="n">—</td><td class="n">—</td><td class="st">{chip("info", "running")}</td></tr>')
            continue
        bd = g["by_difficulty"]
        sp = f'{r["tok_s"] / wfull["tok_s"]:.2f}×' if wfull else "—"
        if label == full_label or gfull is None:
            reading = "baseline" if label == full_label else ""
        else:
            d = 100 * (g["pass@1"] - gfull["pass@1"])
            reading = chip("good", "no loss" if d >= 0 else f"{d:+.1f} pt, within noise") if d > -3.8 else chip("crit" if d < -7 else "warn", f"{d:+.1f} pt")
        rows.append(f'<tr{cls}><td>{disp}</td><td class="n">{pct(g["pass@1"])} %</td><td class="n">{bd.get("easy", "—")} / {bd.get("medium", "—")} / {bd.get("hard", "—")}</td>'
                    f'<td class="n">{r["tok_s"]:.2f}</td><td class="n">{sp}</td><td class="n">{pct(r["trunc_rate"], 0)} %</td><td class="st">{reading}</td></tr>')
    return rows


wc_pts = [("full", "full depth, 4 loops"), ("gate0.7", "released gate, q = 0.7"), ("gate0.5", "released gate, q = 0.5"), ("gate0.3", "released gate, q = 0.3")]
wc_pts2 = [("full_o120", "full depth, 4 loops"), ("gate0.7_o120", "released gate, q = 0.7"), ("gate0.5_o120", "released gate, q = 0.5"), ("gate0.3_o120", "released gate, q = 0.3")]
wrows = wc_rows(wc_pts, "full") + ['<tr class="dim"><td colspan="7"><b>problems 120 to 239, true exit (n = 120)</b></td></tr>'] + wc_rows(wc_pts2, "full_o120")
lcb_wc_block = f'''<div class="sub-head"><h3>LiveCodeBench under real execution, batch 1, first 120 problems</h3><p>Decode-time early exit as in the BigCodeBench and CRUXEval tables; direct mode, 1,024-token budget, graded against the official tests. This slice is harder than the full set (27 easy, 30 medium, 63 hard), so compare routers against each other.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Router</th><th class="n">pass@1 (120)</th><th class="n">easy / medium / hard</th><th class="n">tokens / s</th><th class="n">speed-up</th><th class="n">truncated</th><th class="st">Reading</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in wrows).strip()}
      </tbody>
    </table>
    <p class="src">Standard error about 3.8 points at n = 120. Answers average about 420 tokens, between BigCodeBench's 250 and CRUXEval's 900, so the cost of the skipped-loop cache approximation should sit between the two.</p>
  </div>
'''

# ---------------------------------------------------------------- citation table
cites = {}
cp = f"{TMP}/cites.tsv"
if os.path.exists(cp):
    for line in list(open(cp))[1:]:
        f = line.rstrip("\n").split("\t")
        cites[f[0]] = f[4]
s2 = {}  # Semantic Scholar counts (Google-Scholar scale) where the public API answered
import glob
for p in sorted(glob.glob(f"{TMP}/cites_s2*.tsv")):
    for line in list(open(p))[1:]:
        f = line.rstrip("\n").split("\t")
        if f[4] not in ("None", ""):
            s2[f[0]] = f[4]
sel = [
    ("LiveCodeBench", "LiveCodeBench", "ICLR 2025", "coding", chip("good", "adopted") + sm("direct mode; table above")),
    ("BigCodeBench", "BigCodeBench", "ICLR 2025", "coding", chip("good", "adopted") + sm("direct mode; Hard split is depth-insensitive")),
    ("GSM-Symbolic", "GSM-Symbolic", "ICLR 2025", "math", chip("info", "running") + sm("p1 and main subsets, 300 items each, GSM8K pipeline")),
    ("LiveBench", "LiveBench", "ICLR 2025", "mixed", chip("info", "not run") + sm("six categories, heavy graders; coding part derives from LiveCodeBench")),
    ("CRUXEval-X", "CRUXEval-X", "ACL 2025", "coding", chip("info", "not run") + sm("multilingual CRUXEval; the Python original is the headline here")),
    ("Humanity's Last Exam", "Humanity's Last Exam", "Nature 2026 (arXiv 2025)", "expert QA", chip("crit", "too hard") + sm("frontier models score under 30 %")),
    ("SuperGPQA", "SuperGPQA", "2025", "knowledge", chip("crit", "dropped") + sm("19 % with 59 % of answers truncated")),
    ("KernelBench", "KernelBench", "ICML 2025", "coding", chip("crit", "too hard") + sm("CUDA kernels; small models near zero")),
    ("CodeElo", "CodeElo", "2025", "coding", chip("crit", "too hard") + sm("Codeforces rating; small models near zero")),
    ("LiveCodeBench Pro", "LiveCodeBench Pro", "NeurIPS 2025", "coding", chip("crit", "too hard") + sm("olympiad level")),
    ("SWE-bench", "SWE-bench Verified / Pro", "ICLR 2024 / 2025", "agentic", chip("crit", "out of reach") + sm("count is the original SWE-bench paper; repository agents at 2.6B score about zero")),
    ("Terminal-Bench", "Terminal-Bench", "2025 (paper 2026)", "agentic", chip("info", "Part 2") + sm("pass@1 of 4 to 9 of 22 trials; the coarse, slow metric that motivated this section")),
    ("CodeMMLU", "CodeMMLU", "ICLR 2025", "coding, MC", chip("info", "low citation") + sm("skipped")),
    ("HumanEval Pro / MBPP Pro", "HumanEval Pro / MBPP Pro", "2025", "coding", chip("info", "low citation") + sm("skipped")),
]
crow = []
for key, disp, venue, dom, status in sel:
    c = cites.get(key, "—")
    crow.append(f'<tr><td>{disp}</td><td>{venue}</td><td>{dom}</td><td class="n">{s2.get(key, "—")}</td><td class="n">{c}</td><td class="st">{status}</td></tr>')


def both(key):
    return f"{s2[key]} / {cites.get(key, '—')}" if key in s2 else f"— / {cites.get(key, '—')}"


cite_block = f'''<div class="sub-head"><h3>Benchmark selection: 2025 venues ranked by citations</h3><p>Criterion for the paper's headline benchmark: a 2025 venue, highly cited, preferably coding, and takeable by a 2.6B model in greedy mode within a few hours per router point. Two counts on 3 November 2026: Semantic Scholar (Google-Scholar scale; its public API answered for some papers only) and OpenAlex (about a tenth of that, same ordering). For reference the 2024-venue benchmarks already in use: MMLU-Pro {both("MMLU-Pro")}, GSM8K {both("GSM8K")}, CRUXEval {both("CRUXEval")}, HumanEval {both("HumanEval (Codex)")}.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Benchmark</th><th>Venue</th><th>Domain</th><th class="n">Semantic Scholar</th><th class="n">OpenAlex</th><th class="st">Status for Ouro-2.6B</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in crow).strip()}
      </tbody>
    </table>
    <p class="src">LiveCodeBench is by far the most-cited 2025-venue coding benchmark ({s2.get("LiveCodeBench", "—")} on Semantic Scholar) and BigCodeBench the second; both run end to end here in direct mode. GSM-Symbolic ({s2.get("GSM-Symbolic", "—")}) is the most-cited 2025-venue benchmark outside coding that the model can take; KernelBench ({s2.get("KernelBench", "—")}) is well cited but out of reach for a 2.6B model. Scripts: <code>bench_cites.py</code> (OpenAlex), <code>bench_cites_s2.py</code> (Semantic Scholar), <code>lcb_pipeline.sh</code>, <code>bcb_pipeline.sh</code>, <code>prep_gsm_symbolic.py</code>.</p>
  </div>
'''

# ---------------------------------------------------------------- GSM-Symbolic + LCB rows in the uniform-depth table
def gs_cell(name, label):
    r = result_line(name, label)
    return (pct(r["acc"]) if r else "—"), r


gs_rows = ""
for name, disp in [("gsmsym_p1", "GSM-Symbolic p1, one extra clause (300)"), ("gsmsym_main", "GSM-Symbolic main (300)")]:
    k4, r4 = gs_cell(name, "k4")
    k2, _ = gs_cell(name, "k2")
    q3, r3 = gs_cell(name, "q0.3")
    ps, rp = gs_cell(name, "steps0.1-0.5-0.5")
    if r4 is None:
        note = chip("info", "running") + sm("ICLR 2025 GSM8K variant; same 2,048-token pipeline")
    else:
        extra = []
        if r3:
            extra.append(f"gate q = 0.3: {q3} % at {r3['mean_loops']:.2f} loops")
        if rp:
            extra.append(f"per-step (0.1, 0.5, 0.5): {ps} % at {rp['mean_loops']:.2f}")
        note = chip("info", "ICLR 2025") + sm(f"{pct(r4['trunc_rate'])} % truncated at 2,048 at K = 4" + ("; " + "; ".join(extra) if extra else ""))
    gs_rows += f'\n        <tr><td>{disp}</td><td class="n">—</td><td class="n">{k2}</td><td class="n">—</td><td class="n">{k4}</td><td class="st">{note}</td></tr>'


def lcb_cell(label):
    g = graded(LCB, label)
    return pct(g["pass@1"]) if g else "running"


bd4 = k4g["by_difficulty"] if k4g else {}
lcb_row = (f'<tr><td>LiveCodeBench v5+v6, direct mode (342)</td><td class="n">{lcb_cell("k1")}</td><td class="n">{lcb_cell("k2")}</td><td class="n">—</td>'
           f'<td class="n">{lcb_cell("k4")}</td><td class="st">{chip("good", "unsaturated")}{sm("forced code-block answer, 1,024-token budget; easy / medium / hard at K = 4: " + " / ".join(str(bd4.get(k, "—")) for k in ("easy", "medium", "hard")) + "; thinking mode on the 120 easiest was budget-bound (28.3 %, 83 of 120 cut off at 3,072 tokens)")}</td></tr>')

rep('<tr class="dim"><td>LiveCodeBench easy+medium (120)</td><td class="n">—</td><td class="n">27.5</td><td class="n">—</td><td class="n">28.3</td><td class="st"><span class="chip crit">budget-bound</span><span class="sm">83 of 120 cut off at 3,072 tokens with no code</span></td></tr>',
    lcb_row + gs_rows)

# ---------------------------------------------------------------- the other edits
rep('Real-execution points (full depth, gate 0.5, gate 0.3 on 150 tasks) are running; short outputs should make the skipped-loop cache approximation cheaper than on CRUXEval.',
    'Under real decode-time early exit (next table) the gate at q = 0.5 keeps this accuracy at 1.37× real speed.')
rep('<div class="sub-head"><h3>Real execution: decode-time early exit, batch 1, CRUXEval items 0 to 99</h3>',
    bcb_block + lcb_block + lcb_wc_block + '<div class="sub-head"><h3>Real execution: decode-time early exit, batch 1, CRUXEval items 0 to 99</h3>')
rep('Throughput 80 / 34 / 19 tokens per second at depths 1 / 2 / 4, batched.</p>\n  </div>',
    'Throughput 80 / 34 / 19 tokens per second at depths 1 / 2 / 4, batched.</p>\n  </div>\n' + cite_block.rstrip("\n"))
rep('Q12 · Router headroom: CRUXEval, GSM8K, MATH-500, LiveCodeBench', 'Q12 · Router headroom: CRUXEval, GSM8K, MATH-500, BigCodeBench, LiveCodeBench')
rep('LiveCodeBench releases 5 and 6 (342 problems, 60 easy + 60 medium used); MMLU-Pro and SuperGPQA 300-item stratified subsets queued; GPQA is gated on the Hub',
    'LiveCodeBench releases 5 and 6 (all 342 problems in direct mode; 60 easy + 60 medium in the earlier thinking-mode run); BigCodeBench Hard (148) and a 300-task full-set sample; GSM-Symbolic p1 and main (300 each, 3 instances per template); MMLU-Pro and SuperGPQA 300-item stratified subsets; GPQA is gated on the Hub')
rep('<dt>Pending</dt><dd>held-out CRUXEval (items 200 to 499) and GSM8K (items 300 to 599) for full depth, gate q = 0.3 and the per-step rule; MMLU-Pro and SuperGPQA through the same pipeline; wall-clock with per-step thresholds in the decode-time exit server</dd>',
    '<dt>Pending</dt><dd>LiveCodeBench depth 2, gate and per-step points plus the oracle bound; GSM-Symbolic p1 and main through the headroom pipeline; a LiveCodeBench real-execution run once the gather points are in; the masked skipped-slot cache rule as the next router design axis</dd>')
rep('before it is trusted as the inner loop.</li>',
    'before it is trusted as the inner loop.</li>\n      <li><b>Real exit is cheap when answers are short.</b> On BigCodeBench, where the direct answer is about 250 tokens of code, the released gate at q = 0.5 keeps full-depth accuracy at 1.37× real speed (40.0 vs 40.7 % on 150 tasks) and q = 0.3 costs 3 points at 1.81×; CRUXEval, with 900-token chains of thought, needed q = 0.7 for no loss at 1.12×. The price of the skipped-loop cache approximation grows with the length of the generation that attends to it, which makes short-answer coding benchmarks the friendliest ground for a searched router.</li>')

# ---------------------------------------------------------------- Q11: training-task pass@1 (fit check on the tasks the checkpoints trained on)
import glob as _glob


def tb_counts(name):
    n = ok = to = 0
    for f in _glob.glob(f"/srv/scratch/jli199/tb_runs/{name}/*/*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n += 1
        ok += bool(d.get("is_resolved"))
        to += d.get("failure_mode") == "agent_timeout"
    return ok, n, to


def tt_row(name, disp, saw, budget, reading, cls=""):
    ok, n, to = tb_counts(name)
    cell = f"{ok}/{n}" + (f" ({100 * ok / n:.0f} %)" if n else "")
    return f'<tr{cls}><td>{disp}</td><td>{saw}</td><td class="n">{cell if n else "running"}</td><td class="n">{to if n else "—"}</td><td class="n">{budget}</td><td class="st">{reading}</td></tr>'


r22 = tb_counts("train22_r1_1800")
r22_reading = (chip("info", "running") + sm("1,800-second budget on the 22 tasks ep4 solved")) if r22[1] < 22 else (
    chip("good", f"{100 * r22[0] / r22[1]:.0f} % at the full budget") + sm("given the time, r1 executes its training tasks too"))
if r22[1] and r22[1] < 22:
    r22_reading = chip("info", f"{r22[0]}/{r22[1]} so far") + sm("1,800-second budget on the 22 tasks ep4 solved; still running")
tt_rows = [
    tt_row("train75_v3", "v3", "never", "900 s", "baseline" + sm("solves 27 % of tasks it never trained on"), ' class="dim"'),
    tt_row("train75_ep4", "ep4, 4 true epochs on their trajectories", "yes, memorised (99 % token accuracy)", "900 s",
           chip("good", "executes what it memorised") + sm("while its test pass@1 fell 9/22 → 5/21: over-fitting, not under-fitting")),
    tt_row("train75_r1", "take-turns r1, 1.67 epochs (best test checkpoint)", "yes", "900 s",
           chip("crit", "turns 2× longer") + sm("64 s per turn vs 30 s for v3 at the same 5.5 tokens per second: the take-turns data made the student verbose, and 900 s is not enough")),
    tt_row("train22_r1_1800", "take-turns r1 on the 22 tasks ep4 solved", "yes", "1,800 s", r22_reading),
]
tt_block = f'''
<div class="sub-head"><h3>Training-task pass@1: do the checkpoints solve the tasks they trained on?</h3><p>One attempt per task on the 75 variant tasks whose resolved take-turns trajectories trained r1 and ep4; stock terminus-2 with the serving fixes, one server per card. The 12-hour run cap stopped the 900-second arms at 48 to 53 tasks.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Checkpoint</th><th>Trained on these tasks?</th><th class="n">solved</th><th class="n">timeouts</th><th class="n">budget</th><th class="st">Reading</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in tt_rows).strip()}
      </tbody>
    </table>
    <p class="src">Memorising the trajectories lifts the training-task pass rate from 27 to 42 % and lowers the test score, which settles the under-fitting question the other way round. The r1 row at 900 s measures verbosity, not fit: the checkpoint writes about twice the tokens per turn, so its fit is read from the 1,800-second row. Run dirs: <code>/srv/scratch/jli199/tb_runs/train75_*</code>, <code>train22_r1_1800</code>; counter <code>tb_progress.py</code>.</p>
  </div>'''
_i = html.find('<p class="src">Reading it: the model can absorb the generated trajectories completely')
assert _i > 0
_j = html.find('</p>\n  </div>', _i) + len('</p>\n  </div>')
html = html[:_j] + tt_block + html[_j:]

open(OUT, "w").write(html)
shutil.copy(OUT, SRC)
print(f"wrote {OUT} ({len(html)} bytes); LCB points graded: {done}; oracle: {'yes' if orc else 'no'}; gsmsym_p1 k4: {'yes' if result_line('gsmsym_p1', 'k4') else 'no'}")
