"""Item 3: composition arms from the take-turns shards (rounds 1 and 2).

Shard k used pattern PATTERNS[k] = odd, every:3, every:2, prob:0.5, after:6, odd.
  student-heavy = shards 0, 1, 5 (student opens; teacher takes <= half the turns)
  teacher-heavy = shards 2, 3, 4 (teacher opens, random half, or teacher finishes)
  random        = same number of rows drawn from the union (size control)
All arms are cut to the same row count N = min(|student|, |teacher|).
Writes comp_student.jsonl, comp_teacher.jsonl, comp_random.jsonl to TMP.
"""
import hashlib
import json
import random

TMP = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
PATTERNS = ["odd", "every:3", "every:2", "prob:0.5", "after:6", "odd"]
STUDENT, TEACHER = {0, 1, 5}, {2, 3, 4}


def load(path):
    rows = []
    try:
        for l in open(path):
            if l.strip():
                rows.append(json.loads(l))
    except FileNotFoundError:
        pass
    return rows


def h(row):
    return hashlib.md5(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()


arms = {"student": [], "teacher": []}
seen = set()
per_pattern = {}
for rnd in (1, 2):
    for k in range(6):
        for row in load(f"{TMP}/tvary_r{rnd}_sh{k}.jsonl"):
            key = h(row)
            if key in seen:
                continue
            seen.add(key)
            row["_pattern"] = PATTERNS[k]
            arm = "student" if k in STUDENT else "teacher"
            arms[arm].append(row)
            per_pattern[(arm, PATTERNS[k])] = per_pattern.get((arm, PATTERNS[k]), 0) + 1

n = min(len(arms["student"]), len(arms["teacher"]))
random.seed(0)
union = arms["student"] + arms["teacher"]
out = {
    "student": arms["student"][:n] if len(arms["student"]) == n else random.sample(arms["student"], n),
    "teacher": arms["teacher"][:n] if len(arms["teacher"]) == n else random.sample(arms["teacher"], n),
    "random": random.sample(union, n),
}
print(f"unique rows: student={len(arms['student'])} teacher={len(arms['teacher'])} -> N={n} per arm")
for (arm, p), c in sorted(per_pattern.items()):
    print(f"  {arm:8s} {p:9s} {c}")
for name, rows in out.items():
    path = f"{TMP}/comp_{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    pats = {}
    for r in rows:
        pats[r["_pattern"]] = pats.get(r["_pattern"], 0) + 1
    print(f"wrote {path}: {len(rows)} rows, patterns {pats}")
