"""Build SFT rows from terminal-bench oracle solutions, for a chosen task set.

Generalises build_oracle12.py. The default configuration is the HELD-OUT
experiment: train on the 68 tasks NOT in the 12-task eval set, evaluate on the
12. That is clean for the 12-task eval and contaminated for the 80-task one --
label any result accordingly -- and it is the direct test of whether
oracle-style demonstrations TRANSFER to unseen tasks of the same distribution.

oracle12 (train on the 12, eval on the 12) established the ceiling: 20/24,
effectively 12/12 once a duration bug is discounted. So the model can execute
correct demonstrations. This asks whether it can learn from them.

Row construction is identical to oracle12 so the two results are comparable:
  - USER prompt harvested VERBATIM from our own eval logs (episode-0
    debug.json 'messages'): template, instruction, initial screen, all of it.
  - solution.sh   -> one heredoc write + bash run, what the oracle agent does.
  - solution.yaml -> one keystroke per command, honouring append_enter.
  - then terminus-2's real confirmation prompt and an empty-commands
    task_complete=true reply, matching the loop's double-confirm.

Usage:
  python build_oracle.py                      # 68 held-out -> oracle68/
  python build_oracle.py --tasks a,b,c --out DIR --logs 'glob1,glob2'
"""
import argparse
import glob
import json
import os
import re

import yaml

TB = "/home/jli199/terminal_bench_eval/tb_tasks"
TWELVE = ["hello-world", "fix-permissions", "extract-safely", "csv-to-parquet",
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

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=None,
                help="comma-separated; default = all 80 minus the 12")
ap.add_argument("--out", default="/home/jli199/boptim_scratch/oracle68")
ap.add_argument("--logs", default="/tmp/tb_runs/e80_batching80_s*,"
                                  "/tmp/tb_runs/e80_c10k80_s*,"
                                  "/tmp/tb_runs/clean_continue10k",
                help="comma-separated run-dir globs to harvest prompts from")
args = ap.parse_args()

if args.tasks:
    tasks = [t for t in args.tasks.split(",") if t]
else:
    tasks = sorted(d for d in os.listdir(TB)
                   if os.path.isdir(f"{TB}/{d}") and d not in TWELVE)
log_globs = [g for g in args.logs.split(",") if g]


def harvest_prompt(task):
    for g in log_globs:
        files = sorted(glob.glob(f"{g}/{task}/*/agent-logs/episode-0/debug.json"))
        for f in files:
            try:
                msgs = json.load(open(f))["messages"]
            except Exception:
                continue
            user = [m for m in msgs if m.get("role") == "user"]
            if user and user[0]["content"].startswith("You are an AI assistant"):
                return user[0]["content"]
    return None


def instruction(task):
    y = open(f"{TB}/{task}/task.yaml", errors="ignore").read()
    m = re.search(r"^instruction:\s*\|-?\s*\n((?:[ \t]+.*\n?)+)", y, re.M)
    return " ".join(m.group(1).split()) if m else task


def commands_for(task):
    sh = f"{TB}/{task}/solution.sh"
    yml = f"{TB}/{task}/solution.yaml"
    if os.path.isfile(sh):
        lines = [l for l in open(sh, errors="ignore").read().splitlines()
                 if "terminal-bench-canary" not in l]
        script = "\n".join(lines).strip("\n")
        if DELIM in script or not script.strip():
            return None
        ks = (f"cat > /tmp/oracle_solve.sh << '{DELIM}'\n{script}\n{DELIM}\n"
              f"bash /tmp/oracle_solve.sh\n")
        # Generous: short scripts that install packages or wait on cron cost
        # oracle12 two tasks at duration=5. Wall clock is the only price.
        return [{"keystrokes": ks, "duration": 120.0}]
    if os.path.isfile(yml):
        raw = [l for l in open(yml, errors="ignore").read().splitlines()
               if "terminal-bench-canary" not in l]
        try:
            steps = yaml.safe_load("\n".join(raw)) or []
        except Exception:
            return None
        out = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            cmd = str(s.get("command", ""))
            if s.get("append_enter", True):
                cmd += "\n"
            out.append({"keystrokes": cmd,
                        "duration": max(float(s.get("min_timeout_sec", 1.0)), 5.0)})
        return out or None
    return None


os.makedirs(args.out, exist_ok=True)
rows, skipped = [], []
for t in tasks:
    prompt = harvest_prompt(t)
    cmds = commands_for(t)
    if prompt is None or cmds is None:
        skipped.append((t, "no prompt" if prompt is None else "no solution"))
        continue
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
    rows.append({"task": t, "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(turn1, indent=2)},
        {"role": "user", "content": CONFIRM},
        {"role": "assistant", "content": json.dumps(turn2, indent=2)},
    ]})

path = f"{args.out}/train.jsonl"
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps({"messages": r["messages"]}) + "\n")
with open(f"{args.out}/tasks.txt", "w") as f:
    f.write("\n".join(r["task"] for r in rows) + "\n")

overlap = sorted(set(r["task"] for r in rows) & set(TWELVE))
print(f"requested {len(tasks)}  built {len(rows)}  skipped {len(skipped)}")
for t, why in skipped:
    print(f"   skipped {t}: {why}")
print(f"overlap with the 12-task eval set: {overlap if overlap else 'NONE (clean for the 12)'}")
print(f"wrote -> {path}")
