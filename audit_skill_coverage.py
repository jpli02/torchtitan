#!/usr/bin/env python3
"""Skill-level coverage audit of an SFT set against the 80 terminal-bench tasks.

Category distribution (what tb80sft was built for) is not the same as skill
coverage: vim-terminal-task needs vim demonstrations, qemu-* need qemu, etc.
This maps each of the 80 tasks to the concrete tool/skill its solution
exercises, then counts how many SFT trajectories (and rows) actually run that
tool in an assistant `keystrokes` command. A task whose skill appears in 0
trajectories is an uncovered skill even if its category is well represented.

  python audit_skill_coverage.py /home/jli199/boptim_scratch/tb80sft/train.jsonl
"""
import json
import re
import sys
from collections import defaultdict

# skill -> regex matched against the concatenated keystrokes of a trajectory
SKILL_PAT = {
    "vim":          r"\bvim?\b|\bnvim\b|:wq|:%s/",
    "tmux":         r"\btmux\b",
    "git":          r"\bgit\s",
    "qemu":         r"\bqemu-system|\bqemu-img\b",
    "ssh/telnet":   r"\bssh\b|\bsshd\b|\btelnet\b|ssh-keygen",
    "cron":         r"\bcron\b|crontab|/etc/cron",
    "nginx":        r"\bnginx\b",
    "sqlite":       r"\bsqlite3?\b|\.db\b",
    "7z/crack":     r"\b7z\b|7za|p7zip|john\b|hashcat|\bzip2john",
    "openssl":      r"\bopenssl\b",
    "gpg/encrypt":  r"\bgpg\b|\benc\b -|rencrypt|cryptsetup",
    "tar/extract":  r"\btar\b|\bunzip\b|\bgunzip\b|\bxz\b|\bbunzip2\b",
    "aws/cloud":    r"\baws\b|s3api|s3://|boto3|minio|\bmc\b ",
    "conda":        r"\bconda\b|mamba\b",
    "pip/pkg":      r"\bpip\b|pip3|apt-get|\bapt\b|dpkg|\bpoetry\b",
    "gcc/clang":    r"\bgcc\b|\bg\+\+|\bclang\b|\bmake\b|cmake|gfortran|\bcc\b ",
    "pytorch":      r"\btorch\b|import torch|pytorch|\.pt\b|\.pth\b",
    "huggingface":  r"huggingface|transformers|from_pretrained|hf_hub|datasets\.",
    "fasttext":     r"fasttext",
    "jupyter":      r"\bjupyter\b|notebook|\.ipynb",
    "pandas/csv":   r"pandas|\bpd\.|read_csv|to_parquet|\.csv\b|pyarrow",
    "numpy/sci":    r"numpy|\bnp\.|scipy|matplotlib",
    "coq/proof":    r"\bcoq\b|coqc|\.v\b|Qed\.|Lemma |Theorem ",
    "curl/web":     r"\bcurl\b|\bwget\b|requests\.|urllib|beautifulsoup|bs4",
    "server/port":  r"flask|uvicorn|gunicorn|http\.server|listen\(|:3000|:8080|socket\b",
    "ffmpeg/video": r"\bffmpeg\b|yt-dlp|youtube-dl|\.mp4\b",
    "hexdump/elf":  r"hexdump|\bxxd\b|readelf|\bobjdump\b|\bfile\b |\bstrings\b",
    "kernel/boot":  r"initramfs|bzImage|vmlinuz|\bmkinitramfs\b|busybox",
    "chmod/perms":  r"\bchmod\b|\bchown\b|\bumask\b",
    "grep/sed/awk": r"\bgrep\b|\bsed\b|\bawk\b",
    "json":         r"\bjq\b|json\.load|json\.dump|import json",
}

# each of the 80 tasks -> the skills its solution genuinely needs.
# (keyed by task id; a task can list several.)
TASK_SKILLS = {
    "blind-maze-explorer-5x5": ["grep/sed/awk"],
    "blind-maze-explorer-algorithm.easy": ["gcc/clang", "numpy/sci"],
    "blind-maze-explorer-algorithm.hard": ["gcc/clang", "numpy/sci"],
    "blind-maze-explorer-algorithm": ["gcc/clang", "numpy/sci"],
    "build-initramfs-qemu": ["qemu", "kernel/boot"],
    "build-linux-kernel-qemu": ["qemu", "kernel/boot", "gcc/clang"],
    "build-tcc-qemu": ["qemu", "kernel/boot", "gcc/clang"],
    "cartpole-rl-training": ["pytorch"],
    "chess-best-move": ["numpy/sci"],
    "conda-env-conflict-resolution": ["conda", "pip/pkg"],
    "configure-git-webserver": ["git", "ssh/telnet"],
    "count-dataset-tokens": ["huggingface"],
    "crack-7z-hash.easy": ["7z/crack"],
    "crack-7z-hash.hard": ["7z/crack"],
    "crack-7z-hash": ["7z/crack"],
    "create-bucket": ["aws/cloud"],
    "cron-broken-network": ["cron", "curl/web"],
    "csv-to-parquet": ["pandas/csv"],
    "decommissioning-service-with-sensitive-data": ["gpg/encrypt", "chmod/perms"],
    "download-youtube": ["ffmpeg/video"],
    "eval-mteb.hard": ["huggingface"],
    "eval-mteb": ["huggingface"],
    "extract-moves-from-video": ["ffmpeg/video", "curl/web"],
    "extract-safely": ["tar/extract"],
    "fibonacci-server": ["server/port"],
    "fix-git": ["git"],
    "fix-pandas-version": ["pandas/csv", "pip/pkg"],
    "fix-permissions": ["chmod/perms"],
    "get-bitcoin-nodes": ["server/port", "curl/web"],
    "git-multibranch": ["git", "ssh/telnet"],
    "git-workflow-hack": ["git"],
    "gpt2-codegolf": ["gcc/clang"],
    "grid-pattern-transform": ["numpy/sci"],
    "hello-world": ["grep/sed/awk"],
    "heterogeneous-dates": ["pandas/csv"],
    "hf-model-inference": ["huggingface", "server/port"],
    "incompatible-python-fasttext.base_with_hint": ["fasttext", "pip/pkg"],
    "incompatible-python-fasttext": ["fasttext", "pip/pkg"],
    "intrusion-detection": ["grep/sed/awk"],
    "jupyter-notebook-server": ["jupyter", "server/port"],
    "modernize-fortran-build": ["gcc/clang"],
    "new-encrypt-command": ["gpg/encrypt"],
    "nginx-request-logging": ["nginx"],
    "oom": ["huggingface"],
    "openssl-selfsigned-cert": ["openssl"],
    "organization-json-generator": ["json"],
    "password-recovery": ["hexdump/elf", "grep/sed/awk"],
    "path-tracing-reverse": ["gcc/clang"],
    "path-tracing": ["gcc/clang"],
    "play-zork": ["grep/sed/awk"],
    "polyglot-c-py": ["gcc/clang"],
    "polyglot-rust-c": ["gcc/clang"],
    "processing-pipeline": ["grep/sed/awk"],
    "prove-plus-comm": ["coq/proof"],
    "pytorch-model-cli.easy": ["pytorch", "gcc/clang"],
    "pytorch-model-cli.hard": ["pytorch", "gcc/clang"],
    "pytorch-model-cli": ["pytorch", "gcc/clang"],
    "qemu-alpine-ssh": ["qemu", "ssh/telnet"],
    "qemu-startup": ["qemu", "ssh/telnet"],
    "raman-fitting.easy": ["numpy/sci"],
    "raman-fitting": ["numpy/sci"],
    "reshard-c4-data": ["pandas/csv", "huggingface"],
    "run-pdp11-code": ["hexdump/elf"],
    "sanitize-git-repo.hard": ["git"],
    "sanitize-git-repo": ["git"],
    "security-vulhub-minio": ["aws/cloud"],
    "simple-sheets-put": ["curl/web"],
    "simple-web-scraper": ["curl/web", "pandas/csv"],
    "solana-data": ["server/port", "curl/web"],
    "sqlite-db-truncate": ["sqlite"],
    "sqlite-with-gcov": ["sqlite", "gcc/clang"],
    "super-benchmark-upet": ["huggingface", "pytorch"],
    "swe-bench-astropy-1": ["numpy/sci"],
    "swe-bench-astropy-2": ["numpy/sci"],
    "swe-bench-fsspec": ["pip/pkg"],
    "swe-bench-langcodes": ["pip/pkg"],
    "tmux-advanced-workflow": ["tmux"],
    "train-fasttext": ["fasttext"],
    "vim-terminal-task": ["vim"],
    "write-compressor": ["gcc/clang"],
}


def keystrokes(row):
    out = []
    for m in row["messages"]:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        for ks in re.findall(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', c):
            out.append(ks)
    return "\n".join(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/jli199/boptim_scratch/tb80sft/train.jsonl"
    pats = {s: re.compile(p, re.I) for s, p in SKILL_PAT.items()}
    traj_hits = defaultdict(int)   # skill -> number of trajectories that use it
    row_hits = defaultdict(int)    # skill -> number of assistant turns that use it
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            ks = keystrokes(json.loads(line))
            seen = set()
            for s, pat in pats.items():
                m = pat.findall(ks)
                if m:
                    traj_hits[s] += 1
                    row_hits[s] += len(m)
                    seen.add(s)
    print(f"trajectories scanned: {n}\n")
    print(f"{'skill':14s} {'trajs':>6s} {'cmds':>7s}  covers tasks")
    task_by_skill = defaultdict(list)
    for t, sk in TASK_SKILLS.items():
        for s in sk:
            task_by_skill[s].append(t)
    for s in sorted(SKILL_PAT, key=lambda x: -traj_hits[x]):
        ntask = len(task_by_skill.get(s, []))
        flag = "  <-- UNCOVERED" if traj_hits[s] == 0 and ntask else ""
        print(f"{s:14s} {traj_hits[s]:6d} {row_hits[s]:7d}  {ntask} tasks{flag}")

    print("\n=== per-task coverage (skills with 0 trajectories) ===")
    uncovered_tasks = []
    weak_tasks = []
    for t, sk in sorted(TASK_SKILLS.items()):
        miss = [s for s in sk if traj_hits[s] == 0]
        weak = [s for s in sk if 0 < traj_hits[s] < 5]
        if miss:
            uncovered_tasks.append(t)
            print(f"  MISSING {t:44s} needs {sk} -> no trajectory for {miss}")
        elif weak:
            weak_tasks.append((t, weak))
    print(f"\n{len(uncovered_tasks)}/80 tasks have a skill with ZERO trajectories")
    print(f"{len(weak_tasks)}/80 tasks have a skill with <5 trajectories (thin):")
    for t, weak in weak_tasks:
        print(f"  thin    {t:44s} {[(s, traj_hits[s]) for s in weak]}")


if __name__ == "__main__":
    main()
