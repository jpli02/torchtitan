"""TRAIN-ON-TEST DIAGNOSTIC. Build SFT rows from the 12 eval tasks' own oracle
solutions. The resulting checkpoint is contaminated BY CONSTRUCTION -- the
canary in every task file exists to flag exactly this -- and must never be
reported as a benchmark score. It answers one question only:

    Given perfect demonstrations of these exact tasks, can this model execute
    them through the terminus-2 agent loop?

  YES -> capacity and the agent interface are fine; the ceiling was the data
         (the model never saw good demonstrations). Data work is worth it.
  NO  -> even memorised solutions fail under the real loop, so the bottleneck
         is execution/state-tracking/format, not what the corpus contains.
         More data will not move it.

Fidelity choices:
  - The USER prompt for each task is harvested VERBATIM from our own eval logs
    (episode-0 debug.json 'messages'), so the training prompt distribution is
    identical to what the agent sees at eval time -- template, instruction,
    initial terminal screen, all of it.
  - solution.sh  -> one heredoc write + bash run, which is what the oracle agent
                    itself does. A quoted delimiter prevents expansion.
  - solution.yaml-> one keystroke per listed command, honouring append_enter.
  - Followed by terminus-2's real confirmation prompt and an empty-commands
    task_complete=true reply, matching the double-confirm the loop requires.
"""
import glob
import json
import os
import re

import yaml

TB = "/home/jli199/terminal_bench_eval/tb_tasks"
LOGS = "/tmp/tb_runs/clean_continue10k"
OUT_DIR = "/home/jli199/boptim_scratch/oracle12"
TASKS = ["hello-world", "fix-permissions", "extract-safely", "csv-to-parquet",
         "processing-pipeline", "fix-git", "heterogeneous-dates",
         "chess-best-move", "fibonacci-server", "crack-7z-hash.easy",
         "cron-broken-network", "sanitize-git-repo"]
DELIM = "__ORACLE_EOF__"

CONFIRM = (
    "Current terminal state:\nroot@host:/app# \n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)


def harvest_prompt(task: str) -> str:
    files = sorted(glob.glob(f"{LOGS}/{task}/*/agent-logs/episode-0/debug.json"))
    if not files:
        raise SystemExit(f"{task}: no episode-0 log to harvest a prompt from")
    msgs = json.load(open(files[0]))["messages"]
    user = [m for m in msgs if m.get("role") == "user"]
    txt = user[0]["content"]
    assert txt.startswith("You are an AI assistant"), task
    return txt


def instruction(task: str) -> str:
    y = open(f"{TB}/{task}/task.yaml", errors="ignore").read()
    m = re.search(r"^instruction:\s*\|-?\s*\n((?:[ \t]+.*\n?)+)", y, re.M)
    return " ".join(m.group(1).split()) if m else task


def commands_for(task: str) -> list[dict]:
    sh = f"{TB}/{task}/solution.sh"
    yml = f"{TB}/{task}/solution.yaml"
    if os.path.isfile(sh):
        lines = [l for l in open(sh, errors="ignore").read().splitlines()
                 if "terminal-bench-canary" not in l]
        script = "\n".join(lines).strip("\n")
        assert DELIM not in script, task
        ks = (f"cat > /tmp/oracle_solve.sh << '{DELIM}'\n{script}\n{DELIM}\n"
              f"bash /tmp/oracle_solve.sh\n")
        return [{"keystrokes": ks, "duration": 30.0 if len(script) > 1000 else 5.0}]
    if os.path.isfile(yml):
        raw = [l for l in open(yml, errors="ignore").read().splitlines()
               if "terminal-bench-canary" not in l]
        steps = yaml.safe_load("\n".join(raw)) or []
        out = []
        for s in steps:
            cmd = str(s.get("command", ""))
            if s.get("append_enter", True):
                cmd += "\n"
            out.append({"keystrokes": cmd,
                        "duration": max(float(s.get("min_timeout_sec", 1.0)), 1.0)})
        return out
    raise SystemExit(f"{task}: no solution.sh or solution.yaml")


os.makedirs(OUT_DIR, exist_ok=True)
rows = []
print(f"{'task':<24} {'cmds':>4} {'prompt_chars':>12} {'sol_chars':>9}")
print("-" * 54)
for t in TASKS:
    prompt = harvest_prompt(t)
    cmds = commands_for(t)
    instr = instruction(t)
    turn1 = {
        "analysis": (f"The task is: {instr[:220]} I know the exact steps that "
                     f"solve this, so I will apply them directly."),
        "plan": "Run the solution commands, then mark the task complete.",
        "commands": cmds,
        "task_complete": True,
    }
    turn2 = {
        "analysis": "The solution commands have been executed successfully.",
        "plan": "Confirm completion so the task can be graded.",
        "commands": [],
        "task_complete": True,
    }
    rows.append({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(turn1, indent=2)},
        {"role": "user", "content": CONFIRM},
        {"role": "assistant", "content": json.dumps(turn2, indent=2)},
    ]})
    sol = sum(len(c["keystrokes"]) for c in cmds)
    print(f"{t:<24} {len(cmds):>4} {len(prompt):>12} {sol:>9}")

path = f"{OUT_DIR}/train.jsonl"
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"\nwrote {len(rows)} rows -> {path}")
print("LABEL: TRAIN-ON-TEST diagnostic. Contaminated by construction. "
      "Never report as a score.")
