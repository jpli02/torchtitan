"""Two training trajectories vs two eval tasks, side by side.

Picks TRAINING rows near the data's median length (8 assistant turns) so they
are representative rather than cherry-picked, and shows the task statement, the
opening move, the closing move, and the turn count.

For EVAL, shows the task.yaml instruction plus what our model actually did --
one task it solves and one it never has -- so the horizon and difficulty gap is
visible rather than only tabulated.
"""
import glob
import json
import os
import re
import sys
import textwrap

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
os.environ.setdefault("OURO_SFT_MIX", "longhorizon")
os.environ["OURO_SFT_MIN_CMDS"] = "0"
from torchtitan.hf_datasets.text_datasets import _load_terminal_agent_sft_dataset

W = 96


def wrap(s, indent="      "):
    s = " ".join(str(s).split())
    return textwrap.fill(s, W, initial_indent=indent, subsequent_indent=indent)


def obj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def cmds(o):
    c = (o or {}).get("commands")
    if not isinstance(c, list):
        return "(none)"
    return " ; ".join(str(x.get("keystrokes", "")).replace("\n", "\\n")[:44]
                      for x in c) or "(empty)"


print("=" * W)
print("TRAINING DATA  (2 trajectories near the mix's median of 8 turns)")
print("=" * W)
shown = 0
for row in _load_terminal_agent_sft_dataset("unused"):
    msgs = row.get("messages") or []
    asst = [m for m in msgs if m.get("role") == "assistant"]
    user = [m for m in msgs if m.get("role") == "user"]
    if not (6 <= len(asst) <= 10) or not user:
        continue
    objs = [o for o in (obj(str(m.get("content") or "")) for m in asst) if o]
    if not objs:
        continue
    shown += 1
    print(f"\n--- training example {shown}:  {len(asst)} assistant turns ---")
    print("  TASK GIVEN TO THE MODEL:")
    print(wrap(str(user[0].get("content") or "")[:600]))
    print(f"\n  FIRST turn  -> {cmds(objs[0])[:150]}")
    print(f"  LAST  turn  -> {cmds(objs[-1])[:150]}")
    print(f"  ends with task_complete = {objs[-1].get('task_complete')}")
    if shown >= 2:
        break

print("\n" + "=" * W)
print("EVAL SET  (2 of the 80 terminal-bench tasks)")
print("=" * W)
TB = "/home/jli199/terminal_bench_eval/tb_tasks"
for name in ["hello-world", "sanitize-git-repo"]:
    y = os.path.join(TB, name, "task.yaml")
    if not os.path.isfile(y):
        continue
    txt = open(y, errors="ignore").read()
    m = re.search(r"^instruction:\s*\|-?\s*\n((?:[ \t]+.*\n?)+)", txt, re.M)
    instr = " ".join(m.group(1).split()) if m else "(unparsed)"
    diff = (re.search(r"^difficulty:\s*(\w+)", txt, re.M) or [None, "?"])[1]
    cat = (re.search(r"^category:\s*(\S+)", txt, re.M) or [None, "?"])[1]
    tmo = (re.search(r"^max_agent_timeout_sec:\s*([\d.]+)", txt, re.M) or [None, "?"])[1]
    print(f"\n--- eval task: {name}   [{diff}, {cat}, timeout {tmo}s] ---")
    print("  INSTRUCTION:")
    print(wrap(instr[:700]))
    got = glob.glob(f"/tmp/tb_runs/e80_batching80_s*/{name}/*/results.json")
    for f in got[:1]:
        d = json.load(open(f))
        eps = len(glob.glob(f.rsplit('/', 1)[0] + "/agent-logs/episode-*"))
        print(f"\n  OUR MODEL: resolved={d.get('is_resolved')}  "
              f"mode={d.get('failure_mode')}  episodes={eps}")
        print(f"  tests: {d.get('parser_results')}")
