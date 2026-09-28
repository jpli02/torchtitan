#!/usr/bin/env python3
"""Convert raw BO trajectories into SFT rows that match the LLM optimizer's
INFERENCE contract exactly.

Why not datagen/ConversationFormatter: it emits prose ("I suggest evaluating
at: lr = 0.25, ...") and one long multi-turn chat. At inference the optimizer
(optim/llm_base.py) does neither -- every ask() sends ONE fresh user message
(_build_prompt: template + bounds + full history + current best) and parses
the reply as JSON {"analysis","plan","points":[[...]],"is_complete"}. Training
on a different shape would teach a format the optimizer cannot parse.

So this REPLAYS the ask/tell loop: for iteration i, rebuild the exact prompt
the optimizer would have produced from observations[:i], and pair it with the
JSON reply encoding what the optimizer actually chose (the recorded point) and,
when present, its recorded reasoning ("Analysis: ... | Plan: ..."). One SFT row
per iteration, single-turn.

Sign convention: TrajectoryLogger calls tell(point, sign*value) with sign=-1
for maximize, and _build_prompt picks the best as min(y). The replay uses the
same internal values; otherwise a maximize trajectory would label its WORST
point as the current best.

Usage:
  python research/qwen_bo_agent/bo_replay_format.py data/gpt_cifar_3d_seed0000.json [more.json ...] \
      --out data/bo_sft/train.jsonl [--no-explanation] [--repo /path/to/boptim-agent]
"""
import argparse
import glob
import json
import os
import re

DEFAULT_REPO = os.environ.get("BOPTIM_REPO", "/home/jli199/boptim-agent")


def load_template(repo: str, name: str) -> str:
    return open(os.path.join(repo, "optim", name), encoding="utf-8").read()


def json_format(explanation_enabled: bool, example_format: str) -> str:
    if explanation_enabled:
        return f"""{{
  "analysis": "Analyze the current optimization state and patterns",
  "plan": "Describe your plan for selecting the next points",
  "points": {example_format},
  "is_complete": true_or_false
}}"""
    return f"""{{
  "points": {example_format},
  "is_complete": true_or_false
}}"""


def format_bounds(bounds) -> str:
    return "\n".join(f"  x{i}: [{lo:.2f}, {hi:.2f}]" for i, (lo, hi) in enumerate(bounds))


def format_history(history) -> str:
    if not history:
        return "No evaluations yet."
    out = ""
    for x, y in history:
        x_str = ", ".join(f"{xi:.4f}" for xi in x)
        out += f"  [{x_str}] → {y:.6f}\n"
    return out.strip()


def build_prompt(template: str, bounds, history, explanation_enabled: bool) -> str:
    """Faithful copy of LLMOptimizerMixin._build_prompt for n_points=1."""
    points_desc = "1 point"
    example_format = "[[1.23, -0.45, 2.67]]"
    if len(history) == 0:
        history_section = "No evaluations yet - this is the first iteration."
        best_info = "No current best (first evaluation)."
    else:
        best_y = min(p[1] for p in history)
        best_x = next(p[0] for p in history if p[1] == best_y)
        history_section = f"Evaluation History ({len(history)} points):\n{format_history(history)}"
        best_info = f"Current Best:\nPoint: {best_x}\nValue: {best_y:.6f}"
    return template.format(
        points_desc=points_desc,
        bounds=format_bounds(bounds),
        history_section=history_section,
        best_info=best_info,
        diversity_note="",
        json_format=json_format(explanation_enabled, example_format),
    )


def split_explanation(expl: str):
    """'Analysis: A | Plan: P' -> (A, P); tolerate missing parts."""
    if not expl:
        return "", ""
    m = re.match(r"\s*Analysis:\s*(.*?)\s*\|\s*Plan:\s*(.*)$", expl, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return expl.strip(), ""


SYSTEM_MESSAGE = "You are an expert optimization assistant."


def rows_from_trajectory(traj: dict, template: str, explanation_enabled: bool,
                         system_message: str = SYSTEM_MESSAGE):
    bounds = traj["objective"]["bounds"]
    sign = -1.0 if traj.get("direction", "minimize") == "maximize" else 1.0
    obs = traj["observations"]
    n = len(obs)
    history = []
    rows = []
    for i, o in enumerate(obs):
        prompt = build_prompt(template, bounds, history, explanation_enabled)
        analysis, plan = split_explanation(o.get("explanation", ""))
        reply = {}
        if explanation_enabled:
            reply["analysis"] = analysis
            reply["plan"] = plan
        reply["points"] = [[float(v) for v in o["point"]]]
        reply["is_complete"] = bool(i == n - 1 and traj.get("final_result", {}).get("stopped_early", False))
        # The chatgpt/qwen optimizers send this fixed system message ahead of
        # the prompt (optim/qwen.py, optim/chatgpt.py); training without it
        # would put a system-less prompt distribution in front of the model.
        msgs = []
        if system_message:
            msgs.append({"role": "system", "content": system_message})
        msgs += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)},
        ]
        rows.append({"messages": msgs, "meta": {"trajectory_id": traj.get("trajectory_id"), "iteration": i,
                    "objective": traj["objective"].get("name"),
                    "has_reasoning": bool(o.get("explanation"))}})
        history.append(([float(v) for v in o["point"]], sign * float(o["value"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="raw trajectory JSON files or globs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-explanation", action="store_true",
                    help="emit the points-only JSON contract (for gp_hedge trajectories)")
    ap.add_argument("--template", default=os.environ.get("BOPTIM_PROMPT_TEMPLATE", "chatgpt_prompt.txt"))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--no-system", action="store_true",
                    help="omit the optimizer's fixed system message from the rows")
    args = ap.parse_args()
    system_message = "" if args.no_system else SYSTEM_MESSAGE
    files = []
    for pat in args.inputs:
        files.extend(sorted(glob.glob(pat)) or [pat])
    template = load_template(args.repo, args.template)
    total, with_reasoning = 0, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for fp in files:
            traj = json.load(open(fp))
            expl_on = (not args.no_explanation) and any(o.get("explanation") for o in traj["observations"])
            for r in rows_from_trajectory(traj, template, expl_on, system_message):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
                with_reasoning += r["meta"]["has_reasoning"]
    print(f"wrote {total} single-turn rows from {len(files)} trajectories -> {args.out}")
    print(f"rows with recorded reasoning: {with_reasoning}/{total}")


if __name__ == "__main__":
    main()
