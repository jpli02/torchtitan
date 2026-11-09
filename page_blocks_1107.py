"""Page blocks added 2026-11-07 for page_update_1102.py: the token-accuracy distribution table (Q11), the Dory1.0-suite
benchmark table (Q12, GPQA Diamond + IFBench) and the LiveCodeBench BO status table (Q12). Each function returns HTML
(or "" when the result files are not there yet) and re-reads the result files on every page build."""
import glob
import json
import os

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
BO_EVALS = "/home/jli199/boptim_scratch/bo_lcb_router/evaluations.jsonl"
LCB_FULL_TOK_S = 6.23  # full depth on problems 0..119 (2026-11-03)


def _pct(x, d=1):
    return f"{100 * x:.{d}f}"


def _sm(text):
    return f'<span class="sm">{text}</span>'


def _chip(kind, text):
    return f'<span class="chip {kind}">{text}</span>'


def _lines(name, prefix, label):
    """Last 'PREFIX label {...}' line of TMP/bench_NAME.txt (reruns append, the last one wins)."""
    p = f"{TMP}/bench_{name}.txt"
    hit = None
    if os.path.exists(p):
        for line in open(p):
            if line.startswith(f"{prefix} {label} "):
                hit = line[len(f"{prefix} {label} "):].strip()
    return hit


def last_result(name, label):
    s = _lines(name, "RESULT", label)
    return json.loads(s) if s else None


def last_graded(name, label):
    s = _lines(name, "GRADED", label)
    return dict(kv.split("=", 1) for kv in s.split() if "=" in kv) if s else None


# ---------------------------------------------------------------- Q11: token-accuracy distribution
def tokdist_block():
    p = f"{TMP}/tokdump_summary.json"
    if not os.path.exists(p):
        return ""
    reps = {("ep4" if "ep4" in r["dump"] else "r1"): r for r in json.load(open(p))}
    rows = []
    for name, disp in (("r1", "r1 (1.67 epochs)"), ("ep4", "epoch 4 (ep4)")):
        r = reps.get(name)
        if not r:
            continue
        ar, q, qc, cs, pb, ec = r["acc_by_role"], r["row_acc_quantiles"], r["row_cmd_acc_quantiles"], r["command_spans"], r["acc_by_position"], r["command_error_by_class"]
        fe = cs["first_error_position"]
        share = " / ".join(_pct(ec[c]["share_of_errors"], 0) for c in ("word", "punct/operator", "number") if c in ec)
        rows.append(f'<tr{" class=\"best\"" if name == "ep4" else ""}><td>{disp}</td>'
                    f'<td class="n">{_pct(ar["command"]["acc"])} / {_pct(ar["prose"]["acc"])} / {_pct(ar["other"]["acc"])}</td>'
                    f'<td class="n">{_pct(q["p5"])} / {_pct(q["p50"])} / {_pct(q["p95"])}</td>'
                    f'<td class="n">{_pct(qc["p5"])} / {_pct(qc["p50"])} · {r["rows_cmd_acc_below_0.9"]}</td>'
                    f'<td class="n">{" / ".join(_pct(pb[k]["acc"]) for k in ("<2k", "2-4k", "4-8k", "8k+"))}</td>'
                    f'<td class="n">{cs["fully_correct"]}/{cs["n"]} = {cs["fully_correct_pct"]} %</td>'
                    f'<td class="n">{fe.get("first", 0)} / {fe.get("2nd-3rd", 0)} / {fe.get("4th-8th", 0)} / {fe.get("9th+", 0)}</td>'
                    f'<td class="n">{share} %</td></tr>')
    r1, ep4 = reps.get("r1"), reps.get("ep4")
    reading = ""
    if r1 and ep4:
        c1 = [d for d in r1["top_command_confusions"][:6]]
        ex = "; ".join(f'“{d["gold"]}” → “{d["pred"]}” ({d["n"]})' for d in c1)
        reading = (f'Reading: the misses are a thin, uniform layer, not a few bad rows or a long-context effect. No r1 row falls below {_pct(r1["row_acc_min_max"][0])} % '
                   f'and no ep4 row below {_pct(ep4["row_acc_min_max"][0])} %; accuracy drops under two points from the first 2k tokens to beyond 8k; no row has command accuracy '
                   f'below 90 %. Inside commands the first error usually comes late in the span (token 9 or later in {r1["command_spans"]["first_error_position"].get("9th+", 0)} of '
                   f'{r1["command_spans"]["n"] - r1["command_spans"]["fully_correct"]} imperfect r1 spans), and the frequent confusions are formatting variants and digit slips rather than '
                   f'different commands: {ex}. That is why a {r1["acc_by_role"]["command"]["acc"] * 100:.1f} % command-token accuracy still leaves {100 - r1["command_spans"]["fully_correct_pct"]:.0f} % '
                   f'of the 67-token demonstrated commands imperfect for r1, against {100 - ep4["command_spans"]["fully_correct_pct"]:.0f} % for ep4, and why the error rate, not its location, is the lever.')
    return f'''<div class="sub-head"><h3>Token-accuracy distribution on the 112 training rows</h3><p>Teacher-forced correctness of every assistant token over the full context (fit_eval_dump.py, up to 12,288 tokens), by role: command strings, analysis/plan prose, JSON scaffolding. The question: do the misses concentrate in a few rows, in long contexts, or in a token class?</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Checkpoint</th><th class="n">token acc: command / prose / scaffold</th><th class="n">per-row acc p5 / p50 / p95</th><th class="n">per-row command acc p5 / p50 · rows below 90 %</th><th class="n">acc by position &lt;2k / 2–4k / 4–8k / 8k+</th><th class="n">demonstrated command spans reproduced exactly</th><th class="n">first error at token 1 / 2–3 / 4–8 / 9+</th><th class="n">share of command errors: words / operators / numbers</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">{reading}</p>
  </div>
'''


# ---------------------------------------------------------------- Q12: Dory1.0 suite pieces (GPQA Diamond, IFBench)
def dory_block():
    rows = []
    gp_pts = [("k4", "full depth"), ("k2", "depth 2"), ("k1", "depth 1"), ("q0.5", "gate q = 0.5"), ("q0.3", "gate q = 0.3"), ("q0.1", "gate q = 0.1"),
              ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)"), ("steps0.1-0.5-0.5", "per-step (0.1, 0.5, 0.5)")]
    gd = {lab: last_result("gpqa_direct", lab) for lab, _ in gp_pts}
    for lab, disp in gp_pts:
        r = gd[lab]
        if r:
            rows.append(f'<tr><td>GPQA Diamond, direct answer (198)</td><td>{disp}</td><td class="n">{_pct(r["acc"])} %</td><td class="n">{r["mean_loops"]:.2f}</td>'
                        f'<td class="n">—</td><td class="st">{_sm("forced answer letter, 4-way, chance 25 %")}</td></tr>')
    gt_any = False
    for lab, disp in [("k4", "full depth"), ("k2", "depth 2"), ("q0.3", "gate q = 0.3"), ("steps0.1-0.5-0.5", "per-step (0.1, 0.5, 0.5)"), ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)")]:
        r = last_result("gpqa_think", lab)
        if r:
            gt_any = True
            tr = r.get("trunc_rate", 0)
            rows.append(f'<tr><td>GPQA Diamond, thinking (198)</td><td>{disp}</td><td class="n">{_pct(r["acc"])} %</td><td class="n">{r["mean_loops"]:.2f}</td>'
                        f'<td class="n">{_pct(tr, 0)} %</td><td class="st">{_chip("warn", "budget-bound") if tr > 0.3 else ""}{_sm("2,048-token budget")}</td></tr>')
    if_any = False
    for var, vdisp in (("think", "thinking, 2,048 tokens"), ("direct", "direct, 1,024 tokens")):
        for lab, disp in [("k4", "full depth"), ("k2", "depth 2"), ("q0.3", "gate q = 0.3"), ("q0.5", "gate q = 0.5"), ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)")]:
            r = last_result(f"ifbench_{var}", lab)
            if not r:
                continue
            if_any = True
            g = last_graded(f"ifbench_{var}", lab)
            acc = f'{_pct(float(g["strict_prompt"]))} / {_pct(float(g["loose_prompt"]))} %' if g else "grading"
            rows.append(f'<tr><td>IFBench ({vdisp}, 299)</td><td>{disp}</td><td class="n">{acc}</td><td class="n">{r["mean_loops"]:.2f}</td>'
                        f'<td class="n">{_pct(r.get("trunc_rate", 0), 0)} %</td><td class="st">{_sm("prompt-level accuracy, strict / loose verifiers")}</td></tr>')
    if not rows:
        return ""
    k4, k2, k1 = gd.get("k4"), gd.get("k2"), gd.get("k1")
    reading = []
    if k4 and k2 and k1:
        reading.append(f'GPQA Diamond in the direct regime: {_pct(k4["acc"])} / {_pct(k2["acc"])} / {_pct(k1["acc"])} % at depths 4 / 2 / 1 against a 25 % chance floor, so the '
                       f'single-token answer carries {"a depth signal" if k4["acc"] - k1["acc"] > 0.04 else "little depth signal"} on this set')
        gates = [(lab, gd[lab]) for lab in ("q0.5", "q0.3", "q0.1", "steps0.3-0.5-0.5", "steps0.1-0.5-0.5") if gd.get(lab)]
        if gates:
            reading.append("gates: " + ", ".join(f'{lab} {_pct(r["acc"])} % at {r["mean_loops"]:.2f} loops' for lab, r in gates))
    gk2 = last_result("gpqa_think", "k2")
    if gk2:
        reading.append(f'the thinking regime is budget-bound: at depth 2, {_pct(gk2.get("trunc_rate", 0), 0)} % of the answers are cut at 2,048 tokens and accuracy is '
                       f'{_pct(gk2["acc"])} % (below chance), so the single-token regime is the usable one; the depth-4 thinking points ran a 48 GB card out of memory at batch 6 and were not repeated')
    elif not gt_any:
        reading.append("the thinking regime (2,048 tokens) is queued")
    ik4, ik2, iq3 = (last_result("ifbench_think", l) for l in ("k4", "k2", "q0.3"))
    gk4, gk2i, gq3 = (last_graded("ifbench_think", l) for l in ("k4", "k2", "q0.3"))
    if ik4 and gk4 and ik2 and gk2i:
        s = (f'IFBench, thinking: strict prompt-level accuracy {_pct(float(gk4["strict_prompt"]))} % at depth 4 against {_pct(float(gk2i["strict_prompt"]))} % at depth 2 '
             f'(loose {_pct(float(gk4["loose_prompt"]))} / {_pct(float(gk2i["loose_prompt"]))} %), with {_pct(ik4.get("trunc_rate", 0), 0)} % of the responses cut at 2,048 tokens, '
             f'so part of the gap is budget')
        if iq3 and gq3:
            s += f'; the gate at q = 0.3 keeps {_pct(float(gq3["strict_prompt"]))} % at {iq3["mean_loops"]:.2f} loops'
        dk4, dq3 = last_result("ifbench_direct", "k4"), last_result("ifbench_direct", "q0.3")
        gd4, gd3 = last_graded("ifbench_direct", "k4"), last_graded("ifbench_direct", "q0.3")
        if dk4 and gd4:
            s += (f'. Without reasoning (an empty think block forced, 1,024 tokens, {_pct(dk4.get("trunc_rate", 0), 0)} % truncated) the model scores '
                  f'{_pct(float(gd4["strict_prompt"]))} % strict at depth 4')
            if dq3 and gd3:
                s += f' and {_pct(float(gd3["strict_prompt"]))} % at gate 0.3 ({dq3["mean_loops"]:.2f} loops)'
            s += ': the reasoning is worth about four points here even though it is cut off nearly half the time, and depth is worth two to three'
        reading.append(s)
    elif not if_any:
        reading.append("IFBench is running on GPU 1")
    return f'''<div class="sub-head"><h3>Benchmarks from the Dory1.0 paper (arXiv 2611.03998)</h3><p>Dory1.0-4B (NVIDIA, November 2026) is the closest published looped-LM reference: a looped nGPT trained with a difficulty-aware loop curriculum and reported at 1, 2 and 4 loops (Terminal-Bench v2.1 12.5 / 18.1 / 22.2 %). Its suite is mostly agentic or frontier-hard (SWE-Bench Verified, Terminal-Bench, τ3-Banking, GDPval, SciCode, HLE, AA-LCR); the two pieces a 2.6B non-agentic model can take are GPQA Diamond (198 graduate-level 4-way questions, ungated mirror) and IFBench (NeurIPS 2025, 299 prompts with 58 new verifiable constraints, scored by the official strict and loose verifiers in a sandbox). Gather mode, loops counted.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Benchmark</th><th>Point</th><th class="n">accuracy</th><th class="n">mean loops</th><th class="n">truncated</th><th class="st">Notes</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">{"; ".join(reading)}.</p>
  </div>
'''


# ---------------------------------------------------------------- Q12: BO on LiveCodeBench
def lcb_bo_block():
    evs = [json.loads(l) for l in open(BO_EVALS) if l.strip()] if os.path.exists(BO_EVALS) else []
    evs = [e for e in evs if e.get("status") == "ok"]
    best = min(evs, key=lambda e: e["value"]) if evs else None
    rows = []
    for k, e in enumerate(evs, 1):
        t = e["thresholds"][:3]
        rows.append(f'<tr{" class=\"best\"" if e is best else ""}><td class="n">{k}</td><td>({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})</td><td class="n">{e["pass1_pct"]:.1f}</td>'
                    f'<td class="n">{e["tok_s"]:.2f}</td><td class="n">{e["tok_s"] / LCB_FULL_TOK_S:.2f}×</td><td class="n">{_pct(e.get("trunc_rate") or 0, 0)} %</td>'
                    f'<td class="n">{e["value"]:.2f}</td><td class="n">{e["wall_s"] / 60:.0f} min · GPU {e["gpu"]}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="8">first two evaluations running (started 2026-11-07 21:09)</td></tr>')
    log = f"{TMP}/lcb_bo_gp.log"
    finished = os.path.exists(log) and "Completed 1 experiment" in open(log, errors="replace").read()
    n_failed = 0
    for d in glob.glob("/home/jli199/boptim_scratch/bo_lcb_router/q*/record.json"):
        try:
            n_failed += json.load(open(d)).get("status") == "generation_failed"
        except Exception:
            pass
    status = (f'finished 2026-11-08 12:30: {len(evs)} valid evaluations; {n_failed} proposals failed to generate (out of memory when another user\'s job on the shared card grew) '
              f'and were fed back as the worst value, which steers the model away from that corner' if finished else f'{len(evs)} of 24 evaluations done')
    if best:
        t = best["thresholds"][:3]
        status += f'; best ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}): {best["pass1_pct"]:.1f} % at {best["tok_s"]:.2f} tokens/s, objective {best["value"]:.2f}'
    chip = _chip("good", "finished") if finished else _chip("info", "running")
    return f'''<div class="sub-head"><h3>Bayesian optimisation of the router on LiveCodeBench {chip}</h3><p>The first BO run on the real objective. gp_hedge (boptim-agent) searches the three per-step exit thresholds (q1, q2, q3) in [0.05, 1]; every evaluation serves the 120-problem slice under real decode-time early exit (batch 1, greedy, 1,024 tokens) and grades it against the official tests in the sandbox. Objective, minimised: −tokens/s + 2 × max(0, 21.5 − pass@1 %), the fastest router that keeps the full-depth accuracy (22.5 %) within one point, with a price of two tokens/s per further point lost. Two evaluations run in parallel (GPUs 0 and 3, shared with idle jobs of other users, so tokens/s carry some noise); 4 random initial points, then 10 pairs proposed by the model, about 1 to 1.5 h per evaluation. Every evaluation is recorded (thresholds, pass@1, tokens/s, truncation, loops) as a trajectory for the agent work.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th class="n">#</th><th>thresholds (q1, q2, q3)</th><th class="n">pass@1 % (120)</th><th class="n">tokens / s</th><th class="n">speed-up vs full depth</th><th class="n">truncated</th><th class="n">objective</th><th class="n">cost</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    <p class="src">Reference points on this slice (wall-clock table above): full depth 22.5 % at 6.23 tokens/s (objective −6.2); released gate q = 0.3 21.7 % at 11.12 (−11.1); q = 0.5 20.0 % at 8.82 (−5.8); q = 0.2 12.5 % at 14.21 (+3.8). {status}. Pattern of the search: the first-loop threshold moves up to 0.4 to 0.53 (harder to exit after one loop than the released gate's 0.3), the second-loop threshold collapses towards 0.05 (a token that did not exit after loop 1 almost always exits after loop 2), the third stays near 0.31 to 0.35; a first-loop threshold at or below 0.12 falls off the cliff (10 to 13 % at 2.1 to 2.5×).</p>
  </div>
''' + lcb_bo_validation_block()


def _pooled(label):
    """RESULT + GRADED lines for label_o120 and label_o240 from the wall-clock and validation reports, pooled over the 222 held-out problems."""
    n = passed = toks = wall = 0.0
    slices = []
    for suf in ("_o120", "_o240"):
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
            slices.append(suf)
            n += grd["n"]
            passed += grd["acc"] * grd["n"]
            toks += res["mean_tokens"] * res["n"]
            wall += res["wall_s"]
    if not n:
        return None
    return {"n": int(n), "acc": passed / n, "tok_s": toks / wall, "slices": slices}


def lcb_bo_validation_block():
    cands = [("bo20", "BO best (0.53, 0.06, 0.35)"), ("bo15", "BO runner-up (0.41, 0.27, 0.31)"), ("bo18", "BO highest accuracy (0.49, 0.37, 0.31)"),
             ("gate0.3", "released gate q = 0.3"), ("gate0.5", "released gate q = 0.5"), ("gate0.7", "released gate q = 0.7"), ("full", "full depth")]
    pooled = {k: _pooled(k) for k, _ in cands}
    full = pooled.get("full")
    rows = []
    for k, disp in cands:
        p = pooled[k]
        if not p:
            rows.append(f'<tr class="dim"><td>{disp}</td><td class="n" colspan="4">{"queued on GPU 0" if k.startswith("bo") or k == "gate0.3" else "—"}</td></tr>')
            continue
        part = "" if len(p["slices"]) == 2 else _sm(" (problems 120 to 239 only so far)")
        su = f'{p["tok_s"] / full["tok_s"]:.2f}×' if full else "—"
        rows.append(f'<tr{" class=\"best\"" if k == "bo20" else ""}><td>{disp}</td><td class="n">{_pct(p["acc"])} %{part}</td><td class="n">{p["n"]}</td><td class="n">{p["tok_s"]:.2f}</td><td class="n">{su}</td></tr>')
    reading = ""
    g, b20, b15, b18 = pooled.get("gate0.3"), pooled.get("bo20"), pooled.get("bo15"), pooled.get("bo18")
    if g and b20 and b15 and b18 and len(b20["slices"]) == 2:
        reading = (f'<p class="src">Reading: the search did not transfer. On the 222 unseen problems the BO best gives up {100 * (g["acc"] - b20["acc"]):.1f} points '
                   f'against the released gate at q = 0.3 ({_pct(b20["acc"])} against {_pct(g["acc"])} %) for its {b20["tok_s"] / g["tok_s"]:.2f}× extra speed, and the two BO routers '
                   f'that hold the gate\'s accuracy ({_pct(b15["acc"])} and {_pct(b18["acc"])} %) run at {b15["tok_s"] / g["tok_s"]:.2f}× and {b18["tok_s"] / g["tok_s"]:.2f}× the gate\'s speed, '
                   f'i.e. no better. The objective was scored on 120 problems with a one-point tolerance while the standard error of pass@1 at that size is about 3.8 points, '
                   f'so the model optimised the noise of the search slice: the near-zero second-loop threshold it liked saves a loop on tokens that needed it on the unseen problems '
                   f'(easy problems drop from 23/36 to 19/36). The released gate\'s one knob, swept on a 120-problem slice, generalised; three knobs fitted to the same slice did not. '
                   f'For the paper this sets the protocol: the search needs a pooled objective over all 342 problems (or a held-out accuracy floor inside the loop) and the '
                   f'tolerance must be set from the measured standard error, not to one point.</p>')
    return f'''<div class="sub-head"><h3>Held-out check of the BO routers: LiveCodeBench problems 120 to 341</h3><p>The routers found on problems 0 to 119 are re-run under the same real-execution protocol on the 222 problems the search never saw, next to the released gate at the same cost level. Standard error about 2.5 points at n = 222.</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Router</th><th class="n">pass@1 % (held-out)</th><th class="n">n</th><th class="n">tokens / s</th><th class="n">speed-up vs full depth</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
    {reading}
  </div>
'''
