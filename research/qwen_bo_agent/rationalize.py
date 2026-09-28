#!/usr/bin/env python3
"""Fill the `explanation` of gp_hedge trajectories with teacher-written
"Analysis: ... | Plan: ..." text that rationalises the point gp_hedge chose.

Why: the Qwen3-8B teacher's OWN point policy is worse than random search on
fourier2d (normalised regret 0.33 vs random 0.27, gp_hedge 0.25 on 18 held-out
seeds), so distilling its choices would make the student worse than random.
gp_hedge's choices are the policy we want; the teacher only supplies the
analysis/plan language so the student keeps the LLM-optimizer interface.

For iteration i the teacher sees exactly the optimizer prompt the student
will see (system + _build_prompt over observations[:i]) plus one line naming
the point that was chosen, and must answer {"analysis","plan"} only.

  python research/qwen_bo_agent/rationalize.py --in_dir bo_traj/gp10_fourier2d --out_dir bo_traj/gp10r_fourier2d \
      --base_url http://127.0.0.1:8301/v1 --seeds 0-24
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bo_replay_format import SYSTEM_MESSAGE, build_prompt, load_template  # noqa: E402

TAIL = (
    "\n\nA Bayesian optimization procedure has already selected the next point to evaluate: {point}.\n"
    "Explain that choice as the expert would: in \"analysis\" summarise what the evaluation history "
    "shows (where the good values are, which regions are unexplored), and in \"plan\" say why this "
    "point is the right next evaluation (exploitation near the best points, exploration of "
    "uncertain regions, or a balance). Do not propose a different point. "
    "Respond ONLY with JSON: {{\"analysis\": \"...\", \"plan\": \"...\"}}"
)


def ask(client, model, prompt, max_tokens):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.7, top_p=0.8,
    )
    txt = r.choices[0].message.content or ""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    d = json.loads(m.group(0)) if m else {}
    a, p = str(d.get("analysis", "")).strip(), str(d.get("plan", "")).strip()
    if not a or not p:
        raise ValueError("missing analysis/plan")
    return f"Analysis: {a} | Plan: {p}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--seeds", default=None, help="a-b inclusive filter on trajectory seed")
    ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--repo", default=os.environ.get("BOPTIM_REPO", "/home/jli199/boptim-agent"))
    a = ap.parse_args()
    import openai
    client = openai.OpenAI(api_key="local", base_url=a.base_url)
    template = load_template(a.repo, "chatgpt_prompt.txt")
    os.makedirs(a.out_dir, exist_ok=True)
    lo, hi = (map(int, a.seeds.split("-")) if a.seeds else (None, None))
    done = skipped = failed = 0
    for fp in sorted(glob.glob(os.path.join(a.in_dir, "*.json"))):
        t = json.load(open(fp))
        if lo is not None and not (lo <= t["seed"] <= hi):
            continue
        out = os.path.join(a.out_dir, os.path.basename(fp))
        if os.path.exists(out):
            skipped += 1
            continue
        bounds = t["objective"]["bounds"]
        sign = -1.0 if t.get("direction", "minimize") == "maximize" else 1.0
        history, ok = [], True
        for o in t["observations"]:
            prompt = build_prompt(template, bounds, history, True)
            pt = "[" + ", ".join(f"{v:.4f}" for v in o["point"]) + "]"
            try:
                o["explanation"] = ask(client, a.model, prompt + TAIL.format(point=pt), a.max_tokens)
            except Exception as e:
                try:
                    o["explanation"] = ask(client, a.model, prompt + TAIL.format(point=pt), a.max_tokens)
                except Exception as e2:
                    print(f"  {os.path.basename(fp)} it{o['iteration']}: FAILED {str(e2)[:100]}", flush=True)
                    ok = False
                    break
            history.append(([float(v) for v in o["point"]], sign * float(o["value"])))
        if not ok:
            failed += 1
            continue
        t["optimizer_config"]["rationalized_by"] = a.model
        json.dump(t, open(out, "w"), indent=1)
        done += 1
        print(f"  {os.path.basename(fp)}: {len(t['observations'])} explanations", flush=True)
    print(f"rationalized {done} trajectories ({skipped} already present, {failed} failed) -> {a.out_dir}")


if __name__ == "__main__":
    main()
