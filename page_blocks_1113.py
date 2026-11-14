"""Page block added 2026-11-13, extended 2026-11-14: the benchmark search beyond the selected four and the Dory suite, with the
literature search's venue / citation / small-model figures and our own probe results as they land. Re-reads result files on every build."""
import os

from page_blocks_1107 import TMP, _chip, _pct, _sm, last_result


def _pts(name, labels):
    out = []
    for lab, disp in labels:
        r = last_result(name, lab)
        if r:
            tr = r.get("trunc_rate", 0) or 0
            out.append(f'{disp} {_pct(r["acc"])} %' + (f' at {r["mean_loops"]:.2f} loops' if lab.startswith(("q", "steps")) else "") + (f' ({_pct(tr, 0)} % cut off)' if tr > 0.05 else ""))
    return out


def _bfcl_graded():
    """label -> dict of the last GRADED line of the BFCL report (smoke runs excluded)."""
    p = f"{TMP}/bench_bfcl.txt"
    out = {}
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("GRADED "):
                parts = line.split()
                if parts[1].startswith("smoke"):
                    continue
                kv = dict(x.split("=", 1) for x in parts[2:] if "=" in x)
                try:
                    out[parts[1]] = {k: float(v) for k, v in kv.items() if v not in ("None", "")}
                except ValueError:
                    pass
    return out


def _bfcl_text():
    g = _bfcl_graded()
    if not g:
        return "", None

    def o(lab):
        return f'{100 * g[lab]["overall"]:.1f} %'

    parts = []
    if "plain_k4" in g:
        parts.append(f'reasoning allowed: {o("plain_k4")}')
    gather = [(lab, disp) for lab, disp in (("direct_k4", "depth 4"), ("direct_k2", "depth 2"), ("direct_k1", "depth 1"), ("direct_q0.5", "gate 0.5"),
                                            ("direct_q0.3", "gate 0.3"), ("direct_q0.2", "gate 0.2"), ("direct_q0.1", "gate 0.1"),
                                            ("direct_steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)"), ("direct_steps0.1-0.5-0.5", "per-step (0.1, 0.5, 0.5)")) if lab in g]
    if gather:
        parts.append("call list forced: " + ", ".join(f'{disp} {o(lab)}' + (f' at {g[lab]["loops"]:.2f} loops' if not lab.startswith("direct_k") and "loops" in g[lab] else "")
                                                     for lab, disp in gather))
    true = [(lab, disp) for lab, disp in (("true_full", "full depth"), ("true_q0.7", "gate 0.7"), ("true_q0.5", "gate 0.5"), ("true_q0.3", "gate 0.3"), ("true_q0.2", "gate 0.2")) if lab in g]
    if true:
        full = last_result("bfcl", "true_full")
        segs = []
        for lab, disp in true:
            r = last_result("bfcl", lab)
            sp = f' at {full["wall_s"] / r["wall_s"]:.2f}× end to end' if full and r and lab != "true_full" else ""
            segs.append(f'{disp} {o(lab)}{sp}')
        parts.append("real early exit, batch 1: " + ", ".join(segs))
    return "; ".join(parts), g


def more_candidates_block():
    mm = _pts("mmluredux_direct", [("k4", "depth 4"), ("k2", "depth 2"), ("k1", "depth 1"), ("q0.5", "gate 0.5"), ("q0.3", "gate 0.3"), ("q0.1", "gate 0.1"),
                                   ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)"), ("steps0.1-0.5-0.5", "per-step (0.1, 0.5, 0.5)")])
    lx = _pts("lcbexec_think", [("k4", "depth 4"), ("k2", "depth 2"), ("q0.3", "gate 0.3")])
    lt = _pts("lcbtest_think", [("k4", "depth 4"), ("k2", "depth 2"), ("q0.3", "gate 0.3")])
    tt = _pts("totarith_think", [("k4", "depth 4"), ("q0.3", "gate 0.3")])
    tb = _pts("tbench_think", [("k4", "depth 4"), ("q0.3", "gate 0.3")])
    bf_text, bf = _bfcl_text()

    def st(pts, pending):
        return ("; ".join(pts)) if pts else _sm(pending)

    def trunc(name):
        r = last_result(name, "k4")
        return (r.get("trunc_rate", 0) or 0) if r else None

    def budget_chip(name, ok_text):
        t = trunc(name)
        if t is None:
            return _chip("info", "queued")
        return _chip("warn", "budget-bound at 1,024 tokens") if t > 0.4 else _chip("good", ok_text)

    rows = [
        ("BFCL v3, single-turn AST categories (400-item sample of the 1,150 non-live)", "ICML 2025", "721",
         "Llama-3.2-3B 83, Gemma-3-4b 61; Ouro-2.6B 47.9 base and 86.4 tool-tuned (arXiv 2608.18171)",
         (bf_text or _sm("official checker in a sandbox image being set up")),
         _chip("good", "best new candidate") if bf else _chip("info", "queued")),
        ("LiveCodeBench: code execution (first 100 of 479, with reasoning)", "ICLR 2025", "2,651 (whole paper)", "18–27 % without reasoning; GPT-4o 91 with",
         st(lx, "queued"), budget_chip("lcbexec_think", "usable")),
        ("LiveCodeBench: test-output prediction (first 100 of 442, with reasoning)", "ICLR 2025", "same paper", "8–27 %",
         st(lt, "queued"), budget_chip("lcbtest_think", "usable")),
        ("TableBench, numerical reasoning + fact checking (first 100 of 493, with reasoning)", "AAAI 2025", "198", "7–8B models 11–22 % on the full benchmark",
         st(tb, "queued"), budget_chip("tbench_think", "usable")),
        ("MMLU-Redux 2.0 (1,995 of the 5,330 verified questions, answer letter forced)", "NAACL 2025", "229", "Qwen3-1.7B 64, Phi-4-mini 68, Qwen3-4B 77",
         st(mm, "running"), _chip("info", "single-token regime") if mm else _chip("info", "running")),
        ("Test of Time, arithmetic (first 100 of 350 sampled, with reasoning)", "ICLR 2025", "129", "none reported at 8B and below",
         st(tt, "queued"), budget_chip("totarith_think", "usable")),
        ("Reasoning Gym (procedural generators)", "NeurIPS 2025", "155", "Qwen2.5-3B: arithmetic 30–90, algorithmic 13–52",
         _sm("unlimited fresh items, so a search set and a held-out set of any size; the package install was refused by the session&#39;s permission system and waits for a go-ahead"), _chip("info", "needs approval")),
        ("KOR-Bench (1,250)", "ICLR 2025", "87", "Qwen2.5-1.5B 20, Phi-3.5-mini 39", _sm("rule-following with regex + SymPy grading; not wired in yet"), ""),
        ("JSONSchemaBench (2,867 test)", "arXiv 2025", "122", "Llama-3.2-1B 13–90 by subset", _sm("schema-validity only; not wired in yet"), ""),
        ("NESTFUL (1,861)", "EMNLP 2025", "71", "tool-tuned Ouro-2.6B win rate 37 (arXiv 2608.18171)", _sm("official scorer repository unavailable"), ""),
    ]
    body = "\n".join(f'        <tr><td>{a}</td><td>{b}</td><td class="n">{c}</td><td>{_sm(d)}</td><td class="st">{f}{e}</td></tr>' for a, b, c, d, e, f in rows)
    reading = ""
    if bf and "direct_k4" in bf and "direct_k1" in bf and "direct_q0.3" in bf:
        k4, k2, k1, q3 = bf["direct_k4"], bf.get("direct_k2"), bf["direct_k1"], bf["direct_q0.3"]
        reading = (f'<p class="src">Reading. <b>BFCL is the strongest new candidate</b>: scored by the official AST checker, the released Ouro-2.6B goes from '
                   f'{100 * k1["overall"]:.1f} % at one loop' + (f' to {100 * k2["overall"]:.1f} % at two' if k2 else "") + f' and {100 * k4["overall"]:.1f} % at four, and the released gate at '
                   f'q = 0.3 keeps {100 * q3["overall"]:.1f} % at {q3["loops"]:.2f} loops, so depth matters and the gate already recovers most of it at half the loops. Its answers are one-line '
                   f'function calls of 30 to 100 tokens, which makes an evaluation cost minutes instead of hours, and the benchmark has 1,150 non-live plus 1,351 live items, enough for a '
                   f'search set and a held-out set. Forcing the call list is what the comparison should use: with free-form reasoning the model spends its budget on prose '
                   + (f'({100 * bf["plain_k4"]["overall"]:.1f} %; the published base number is 47.9 %, protocol not stated). ' if "plain_k4" in bf else ". ")
                   + 'The two LiveCodeBench scenarios and Test of Time are budget-bound at 1,024 tokens with reasoning (the gate even raises the score by shortening the answers), '
                   'so they need a 2,048-token budget before they say anything about routing. TableBench is usable as is. MMLU-Redux shows the largest gap between fixed depths, but its answer '
                   'is a single token: the exit rule only acts on the two generated tokens while the prompt is processed at full depth, so it measures depth sensitivity, not decode-time savings.</p>')
    return f'''<div class="sub-head"><h3>Benchmark search beyond the selected four (rounds of 2026-11-13 and 11-14)</h3><p>A literature search for 2025-venue, highly cited, programmatically graded, ungated benchmarks a 2.6B non-agentic model can take, ranked by fit after our own probes. Venue, citation counts (Semantic Scholar, read on 2026-11-13) and small-model reference scores are as reported by that search and the cited papers, not re-measured here; the last column is our probe on the released Ouro-2.6B (gather mode, loops counted, unless marked real early exit). Rejected after checking: ACPBench (small models near chance), PlatinumBench / GSM8K-Platinum (saturated), BBEH (at the floor), MedXpertQA, EquiBench and CodeJudge-Eval (near random for small models), UGMathBench and BigOBench (few citations), MATH-Perturb (budget-bound like AIME), MathCheck-GSM (GSM8K-derived, close to GSM-Symbolic).</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Benchmark</th><th>Venue</th><th class="n">citations</th><th>small-model reference</th><th class="st">Our probe on Ouro-2.6B</th></tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
    {reading}
    <p class="src">Two 2026 papers from one Cambridge group already study the released Ouro checkpoints and matter for positioning: arXiv 2607.20519 (learned halting gates against post-hoc confidence read-outs on MMLU, ARC, OpenBookQA, HellaSwag and CommonsenseQA, concluding that post-hoc read-outs often match or beat the learned gate) and arXiv 2608.18171 (Ouro fine-tuned for tool calling on BFCL v3, NESTFUL and API-Bank, with the released gate swept over thresholds 0.1 to 0.8). Neither searches the per-loop thresholds or measures real decode-time early exit; both use the gate as released.</p>
  </div>
'''
