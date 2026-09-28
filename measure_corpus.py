"""Score any Hub trajectory corpus against the terminal-bench-80 profile.

What the oracle experiments and traj20k established: the data that moves
pass@1 is verified, MULTI-TURN demonstrations in terminus-2 format on tasks of
the same distribution. So each corpus is measured on exactly those axes:

  fmt       assistant turns that parse as terminus-2 JSON with a commands list
            (what the agent emits; other formats need conversion)
  turns     assistant turns per trajectory -- TB-80 median episode is 22
  finish    ends in task_complete=true (a claim, not a verified solve)
  label     does the corpus carry a success/reward/resolved field
  domain    keyword-classified against TB-80's own categories, so the mix can
            be reweighted to match the benchmark:
              software-engineering 17  system-administration 13  security 12
              debugging 10  file-operations 8  data-science 8  model-training 7
              games 3  scientific-computing 2
  contam    rows mentioning any of our 80 task names, or the canary

Usage: python measure_corpus.py repo[:config] [repo ...]   (env ROWS=200)
"""
import json
import os
import re
import statistics
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from torchtitan.hf_datasets.text_datasets import _normalise_terminal_messages  # noqa: E402

from datasets import load_dataset  # noqa: E402

ROWS = int(os.environ.get("ROWS", "200"))
TB = "/home/jli199/terminal_bench_eval/tb_tasks"
TB_TASKS = sorted(d for d in os.listdir(TB) if os.path.isdir(os.path.join(TB, d)))
CANARY = "terminal-bench-canary"
LABEL_KEYS = ("reward", "resolved", "success", "exit_status", "passed", "score", "ground_truth")

DOMAINS = {
    "security": ["password", "hash", "crack", "encrypt", "decrypt", "ssl", "cert",
                 "openssl", "vulnerab", "secret", "token", "permission", "chmod",
                 "firewall", "ssh", "gpg", "exploit", "sanitize", "malware", "cve"],
    "system-administration": ["cron", "systemd", "service", "nginx", "apache",
                              "network", "dns", "port ", "daemon", "apt", "install",
                              "disk", "mount", "user account", "logrotate", "tmux",
                              "environment variable", "docker", "process", "kill"],
    "debugging": ["fix", "bug", "broken", "error", "fails", "failing", "crash",
                  "debug", "traceback", "not working", "doesn't work", "repair"],
    "file-operations": ["rename", "move", "copy", "archive", "tar", "zip",
                        "extract", "compress", "directory", "find files",
                        "convert", "parquet", "csv"],
    "data-science": ["pandas", "dataframe", "plot", "statistic", "aggregate",
                     "sql", "query", "jsonl", "analysis", "dataset", "numpy"],
    "model-training": ["train", "model", "pytorch", "neural", "epoch",
                       "checkpoint", "huggingface", "fine-tune", "gpu", "cuda",
                       "torch", "weights"],
    "software-engineering": ["implement", "function", "class ", "refactor", "api",
                             "endpoint", "library", "pytest", "test suite",
                             "compile", "build", "git", "repository", "commit",
                             "merge", "rebase", "package", "server"],
    "games": ["game", "chess", "maze", "puzzle", "sudoku", "tetris", "player"],
    "scientific-computing": ["physics", "simulat", "numerical", "ode", "matrix",
                             "fortran", "scipy", "spectrum", "fitting"],
}
TB_DIST = {"software-engineering": 17, "system-administration": 13, "security": 12,
           "debugging": 10, "file-operations": 8, "data-science": 8,
           "model-training": 7, "games": 3, "scientific-computing": 2}


def classify(text: str) -> str:
    t = text.lower()
    scores = {d: sum(t.count(k) for k in kws) for d, kws in DOMAINS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def obj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def find_conv(row):
    """Return (column, value) for the first conversation-shaped column."""
    for k in ("messages", "conversations", "trajectory", "conversation", "turns"):
        v = row.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k, v
        if isinstance(v, str) and v.strip().startswith("["):
            try:
                p = json.loads(v)
                if p and isinstance(p[0], dict):
                    return k, p
            except Exception:
                pass
    for k, v in row.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and \
                any(x in v[0] for x in ("role", "from", "content", "value")):
            return k, v
    return None, None


TEMPLATE_MARK = "You are an AI assistant tasked with solving command-line tasks"


def task_text(msgs):
    """The task statement only. The terminus-2 template shares ~1,200 chars of
    boilerplate across every row ('install', 'process', 'commands', ...), which
    swamped the keyword classifier -- OpenThoughts came out 99% sysadmin. Prefer
    the first USER turn (system prompts of other harnesses are pure boilerplate)
    and cut the template away."""
    user = [m for m in msgs if m.get("role") == "user"]
    sysm = [m for m in msgs if m.get("role") == "system"]
    src = user[0] if user else (sysm[0] if sysm else {})
    c = str(src.get("content") or "")
    if TEMPLATE_MARK in c:
        i = c.find("Task Description:")
        j = c.find("Current terminal state")
        if i >= 0:
            return c[i + len("Task Description:"):(j if j > i else i + 1600)]
        return c[1200:2800]
    return c[:1500]


def measure(spec):
    repo, _, cfg = spec.partition(":")
    try:
        ds = load_dataset(repo, cfg or None, split="train", streaming=True)
    except Exception as e:
        print(f"{spec:<58} LOAD FAILED: {str(e)[:60]}")
        return
    n = fmt_ok = fmt_tot = fin = contam = canary = 0
    turns, dom = [], {}
    label = None
    col = None
    for row in ds:
        if label is None:
            label = ",".join(k for k in row if k.lower() in LABEL_KEYS) or "-"
        col, conv = find_conv(row)
        if conv is None:
            n += 1
            if n >= 5:
                break
            continue
        msgs = _normalise_terminal_messages(conv)
        if not msgs:
            continue
        n += 1
        a = [m for m in msgs if m.get("role") == "assistant"]
        turns.append(len(a))
        for m in a:
            fmt_tot += 1
            o = obj(str(m.get("content") or ""))
            if o is not None and isinstance(o.get("commands"), list):
                fmt_ok += 1
        last = None
        for m in a:
            o = obj(str(m.get("content") or ""))
            if o:
                last = o
        fin += bool(last and last.get("task_complete") is True)
        blob = "\n".join(str(m.get("content") or "") for m in msgs)
        if CANARY in blob:
            canary += 1
        if any(t in blob for t in TB_TASKS):
            contam += 1
        d = classify(task_text(msgs))
        dom[d] = dom.get(d, 0) + 1
        if n >= ROWS:
            break
    if not turns:
        print(f"{spec:<58} no conversation column found (col={col})")
        return
    fmt = 100 * fmt_ok / max(fmt_tot, 1)
    print(f"{spec:<58} rows={n:<4} col={col:<13} label={label:<12} "
          f"fmt={fmt:5.1f}%  turns={statistics.mean(turns):5.1f}/{statistics.median(turns):<4.0f} "
          f"finish={100 * fin / n:5.1f}%  contam={contam:<3} canary={canary}")
    tot = sum(dom.values())
    order = list(TB_DIST) + ["other"]
    print("   domain%  " + "  ".join(f"{k[:6]}={100 * dom.get(k, 0) / tot:4.0f}" for k in order))


if __name__ == "__main__":
    tb_tot = sum(TB_DIST.values())
    print("TB-80 target domain%  " + "  ".join(f"{k[:6]}={100 * v / tb_tot:4.0f}" for k, v in TB_DIST.items()))
    print()
    for s in sys.argv[1:]:
        measure(s)
