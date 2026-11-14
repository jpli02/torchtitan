"""Page block added 2026-11-13: the second benchmark search (candidates beyond the selected four and the Dory suite), with the
literature search's venue / citation / small-model figures and our own probe results as they land. Re-reads result files on every build."""
import json
import os

from page_blocks_1107 import TMP, _chip, _pct, _sm, last_result


def _pts(name, labels):
    out = []
    for lab, disp in labels:
        r = last_result(name, lab)
        if r:
            tr = r.get("trunc_rate", 0) or 0
            out.append(f'{disp} {_pct(r["acc"])} %' + (f' at {r["mean_loops"]:.2f} loops' if lab.startswith(("q", "steps")) else "") + (f' ({_pct(tr, 0)} % cut)' if tr > 0.05 else ""))
    return out


def _bfcl():
    p = f"{TMP}/bench_bfcl.txt"
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        if line.startswith("GRADED "):
            parts = line.split()
            kv = dict(x.split("=", 1) for x in parts[2:] if "=" in x)
            out.append(f'{parts[1]} {100 * float(kv.get("overall", 0)):.1f} %' + (f' at {float(kv["loops"]):.2f} loops' if "loops" in kv else ""))
    return out


def more_candidates_block():
    mm = _pts("mmluredux_direct", [("k4", "depth 4"), ("k2", "depth 2"), ("k1", "depth 1"), ("q0.5", "gate 0.5"), ("q0.3", "gate 0.3"), ("q0.1", "gate 0.1"),
                                   ("steps0.3-0.5-0.5", "per-step (0.3, 0.5, 0.5)"), ("steps0.1-0.5-0.5", "per-step (0.1, 0.5, 0.5)")])
    lx = _pts("lcbexec_think", [("k4", "depth 4"), ("k2", "depth 2"), ("q0.3", "gate 0.3")])
    lt = _pts("lcbtest_think", [("k4", "depth 4"), ("k2", "depth 2"), ("q0.3", "gate 0.3")])
    tt = _pts("totarith_think", [("k4", "depth 4"), ("q0.3", "gate 0.3")])
    tb = _pts("tbench_think", [("k4", "depth 4"), ("q0.3", "gate 0.3")])
    bf = _bfcl()

    def st(pts, pending):
        return ("; ".join(pts)) if pts else _sm(pending)

    rows = [
        ("LiveCodeBench: code execution (479)", "ICLR 2025", "2,651 (whole paper)", "18–27 % without reasoning; GPT-4o 91 with",
         st(lx, "wired in (thinking, 100 items); queued on GPU 0 behind the BO"), _chip("good", "same paper as the headline") if lx else _chip("info", "queued")),
        ("LiveCodeBench: test-output prediction (442)", "ICLR 2025", "same paper", "8–27 %",
         st(lt, "wired in (thinking, 100 items); queued on GPU 0 behind the BO"), _chip("info", "queued") if not lt else ""),
        ("BFCL, single-turn AST categories (1,150 non-live)", "ICML 2025", "721", "Llama-3.2-3B 83, Gemma-3-4b 61; Ouro-2.6B 47.9 base and 86.4 tool-tuned (arXiv 2608.18171)",
         st(bf, "official checker in a sandbox image being set up; answers are one-line function calls"), _chip("good", "published Ouro reference") if not bf else ""),
        ("MMLU-Redux 2.0 (1,995 of the 5,330 verified questions)", "NAACL 2025", "229", "Qwen3-1.7B 64, Phi-4-mini 68, Qwen3-4B 77",
         st(mm, "running"), _chip("good", "strong depth signal") if mm else _chip("info", "running")),
        ("Test of Time, arithmetic (350 sampled)", "ICLR 2025", "129", "none reported at 8B and below",
         st(tt, "wired in (JSON answers, exact match); answering without reasoning scores 1 of 8; thinking probe queued"), ""),
        ("TableBench, numerical reasoning + fact checking (493)", "AAAI 2025", "198", "7–8B models 11–22 %",
         st(tb, "wired in; answering without reasoning scores 1 of 6; thinking probe queued"), _chip("warn", "likely too hard") if not tb else ""),
        ("Reasoning Gym (procedural generators)", "NeurIPS 2025", "155", "Qwen2.5-3B: arithmetic 30–90, algorithmic 13–52",
         _sm("unlimited fresh items, so a search set and a held-out set of any size (the remedy for the first BO run); the package install was refused by the session's permission system and waits for a go-ahead"), _chip("info", "needs approval")),
        ("KOR-Bench (1,250)", "ICLR 2025", "87", "Qwen2.5-1.5B 20, Phi-3.5-mini 39", _sm("rule-following with regex + SymPy grading; not wired in yet"), ""),
        ("JSONSchemaBench (2,867 test)", "arXiv 2025", "122", "Llama-3.2-1B 13–90 by subset", _sm("schema-validity only; not wired in yet"), ""),
        ("NESTFUL (1,861)", "EMNLP 2025", "71", "tool-tuned Ouro-2.6B win rate 37 (arXiv 2608.18171)", _sm("official scorer repository unavailable"), ""),
    ]
    body = "\n".join(f'        <tr><td>{a}</td><td>{b}</td><td class="n">{c}</td><td>{_sm(d)}</td><td class="st">{f}{e}</td></tr>' for a, b, c, d, e, f in rows)
    return f'''<div class="sub-head"><h3>Second benchmark search (2026-11-13): candidates beyond the selected four</h3><p>A literature search for 2025-venue, highly cited, programmatically graded, ungated benchmarks a 2.6B non-agentic model can take, ranked by fit. Venue, citation counts (Semantic Scholar, read on 2026-11-13) and small-model reference scores are as reported by that search and the cited papers, not re-measured here; the last column is our own probe (gather mode, loops counted). Rejected after checking: ACPBench (small models near chance), PlatinumBench / GSM8K-Platinum (saturated), BBEH (at the floor), MedXpertQA, EquiBench and CodeJudge-Eval (near random for small models), UGMathBench and BigOBench (few citations), MATH-Perturb (budget-bound like AIME), MathCheck-GSM (GSM8K-derived, close to GSM-Symbolic).</p></div>
<div class="tablewrap">
    <table>
      <thead><tr><th>Benchmark</th><th>Venue</th><th class="n">citations</th><th>small-model reference</th><th class="st">Our probe on Ouro-2.6B</th></tr></thead>
      <tbody>
{body}
      </tbody>
    </table>
    <p class="src">Two 2026 papers from one Cambridge group already study the released Ouro checkpoints and matter for positioning: arXiv 2607.20519 (learned halting gates against post-hoc confidence read-outs on MMLU, ARC, OpenBookQA, HellaSwag and CommonsenseQA, concluding that post-hoc read-outs often match or beat the learned gate) and arXiv 2608.18171 (Ouro fine-tuned for tool calling on BFCL v3, NESTFUL and API-Bank, with the released gate swept over thresholds 0.1 to 0.8). Neither searches the per-loop thresholds or measures real decode-time early exit; both use the gate as released.</p>
  </div>
'''
