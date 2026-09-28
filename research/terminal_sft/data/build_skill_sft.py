#!/usr/bin/env python3
"""Build a per-skill SFT set: one target skill concentrated to a target
fraction, the rest filled with general terminal rows so the agent format and
broad competence survive. Used to test, skill by skill, whether matching the
train tool-distribution to a test task's tools lifts pass@1 on that task.

A row "uses" the skill if the skill's regex matches the ASSISTANT turns only
(analysis/plan/keystrokes) -- never the terminus-2 boilerplate, which names
tmux in every prompt and would match everything.

  python research/terminal_sft/data/build_skill_sft.py --skill tmux --total 2000 --frac 0.40 \
      --src /home/jli199/boptim_scratch/tb80sft/train_v3.jsonl \
      --out /home/jli199/boptim_scratch/tb80sft/skill_tmux.jsonl
"""
import argparse
import json
import random
import re

# assistant-side signatures (usage, not mention). Keep in sync with tb_dist.
SKILL_PAT = {
    "tmux": r"\btmux\b",
    "git": r"\bgit\s+(clone|commit|checkout|branch|log|merge|rebase|reset|reflog|push|init|stash|filter|remote|config|add)\b",
    "qemu": r"\bqemu",
    "sqlite": r"\bsqlite3?\b|\bpsql\b|\bmysql\b",
    "python-env": r"\bpip3? install\b|\bconda (install|create|env)\b|\bvenv\b|\buv (pip|sync|add)\b",
    "pytorch": r"\btorch\b|\bpytorch\b|\.backward\(\)|model\.train\(\)",
    "huggingface": r"\btransformers\b|\bhuggingface\b|\bfrom datasets\b|AutoModel|\btokenizer",
    "fasttext": r"\bfasttext\b",
    "jupyter": r"\bjupyter\b|\.ipynb\b|\bnbconvert\b",
    "cbuild": r"\bgcc\b|\bg\+\+|\bclang\b|\bcmake\b|\bmake\b",
    "textproc": r"\bsed\b|\bawk\b|re\.(compile|search|findall|sub)\(",
    "web-net": r"\bnginx\b|\bcurl\b|\bflask\b|\bfastapi\b|\bgrpc\b|\bssh\b|http\.server|\buvicorn\b",
    "crypto": r"\bopenssl\b|\bjohn\b|\bhashcat\b|\b7z\b|\bgpg\b",
    "data-files": r"\bpandas\b|\.csv\b|\bparquet\b|\bpyarrow\b",
    "vim": r"\bvim?\s|\bnano\s|\bemacs\b",
    "cron-svc": r"\bcron(tab)?\b|\bsystemctl\b|\bsystemd\b",
    "perms-arch": r"\bchmod\b|\bchown\b|\btar\b|\bunzip\b",
    "games": r"\bmaze\b|\bchess\b|\bzork\b|\bcorewars?\b|\bpmars\b",
    "sci": r"\bnumpy\b|\bscipy\b|\bRscript\b|\bstan\b|\bmcmc\b",
}
KS = re.compile(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"')


def assistant_text(row):
    return "\n".join(m["content"] for m in row["messages"] if m["role"] == "assistant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, choices=list(SKILL_PAT))
    ap.add_argument("--src", default="/home/jli199/boptim_scratch/tb80sft/train_v3.jsonl")
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--frac", type=float, default=0.40, help="target fraction of rows using the skill")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pat = re.compile(SKILL_PAT[a.skill], re.I)
    skill_rows, general = [], []
    for line in open(a.src):
        r = json.loads(line)
        if pat.search(assistant_text(r)):
            skill_rows.append(r)
        else:
            general.append(r)
    n_skill_unique = len(skill_rows)
    want_skill = int(a.total * a.frac)
    want_gen = a.total - want_skill
    # upsample skill rows with replacement to want_skill; sample general without
    out = []
    if skill_rows:
        for _ in range(want_skill):
            out.append(rng.choice(skill_rows))
    rng.shuffle(general)
    out += general[:want_gen]
    rng.shuffle(out)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    upsample = want_skill / max(n_skill_unique, 1)
    print(f"skill={a.skill} unique_skill_rows={n_skill_unique} "
          f"-> {want_skill} skill ({upsample:.1f}x upsample) + {want_gen} general = {len(out)} rows -> {a.out}")


if __name__ == "__main__":
    main()
