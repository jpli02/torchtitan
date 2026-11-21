"""Interim score of a real-execution run that is still going (lcb_real_eval2.py): grade the answers of the finished chunks with
the official tests (same sandbox and grader as the driver, under a separate label) and compare them, problem by problem, with
the released model's runs on the same problems (paired.py). Reads only finished chunk files; the run itself is not touched.

  python randrouter/interim_grade.py RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired  # noqa: E402

WT = Path("/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
STAGE = Path("/srv/scratch/jli199/lcb_grade_stage_bo")
DATA = "lcb_all.jsonl"


def main():
    for run_dir in sys.argv[1:]:
        run_dir = Path(run_dir.rstrip("/"))
        rows, n_tok, n_cut, loops_w, exec_n = [], 0.0, 0.0, 0.0, 0
        for f in sorted(glob.glob(str(run_dir / "chunk_*.jsonl"))):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if "question_id" in r:
                    rows.append(line if line.endswith("\n") else line + "\n")
                elif "summary" in r:
                    s = r["summary"]
                    n_tok += s["mean_tokens"] * s["n"]
                    n_cut += s["trunc_rate"] * s["n"]
                    if s.get("exec_n"):
                        loops_w += s["exec_loops"] * s["exec_n"]
                        exec_n += s["exec_n"]
        if not rows:
            print(f"{run_dir.name}: no finished chunk")
            continue
        label = f"interim_{run_dir.name}"
        (STAGE / f"{label}.jsonl").write_text("".join(rows))
        if not (STAGE / DATA).exists():
            subprocess.run(["cp", f"/home/jli199/.claude/jobs/c0d2da0a/tmp/{DATA}", str(STAGE / DATA)], check=False)
        subprocess.run(["cp", str(WT / "lcb_grade.py"), str(STAGE / "lcb_grade.py")], check=False)
        log = subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "12g", "--cpus", "4", "-v", f"{STAGE}:/w", "python:3.12-slim",
                              "python", "/w/lcb_grade.py", f"/w/{DATA}", f"/w/{label}.jsonl", f"/w/{label}.graded.json"],
                             capture_output=True, text=True, timeout=3600).stdout
        m = re.search(r"pass@1=([0-9.]+)", log)
        if not m:
            print(f"{run_dir.name}: grading produced no pass@1: {log[-300:]}")
            continue
        n = len(rows)
        el = f", executed loops {loops_w / exec_n:.2f}" if exec_n else ""
        print(f"{run_dir.name}: {n} problems graded so far: pass@1 {100 * float(m.group(1)):.1f} %, {n_tok / n:.0f} answer tokens, "
              f"{100 * n_cut / n:.0f} % cut at 1,024{el}")
        paired.compare(paired.passed([str(STAGE / f"{label}.graded.json")]))


if __name__ == "__main__":
    main()
