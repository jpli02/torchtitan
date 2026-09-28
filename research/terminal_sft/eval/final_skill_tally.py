import glob, os, re
base = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
solved = {}
n_done = 0
for f in sorted(glob.glob(os.path.join(base, "skill_*.txt"))):
    t = open(f).read()
    if "pass@k on skill" not in t:
        continue
    n_done += 1
    name = os.path.basename(f)[len("skill_"):-len(".txt")]
    hits = re.findall(r"^\s+(\S+): (\d+)/(\d+)\s+<-- SOLVED", t, re.M)
    if hits:
        solved[name] = [(task, s, c) for task, s, c in hits]
print(f"skills with results: {n_done}")
print("skills that moved a task off 0:")
for name, hits in solved.items():
    for task, s, c in hits:
        print(f"  {name:12s} {task:34s} {s}/{c}")
print(f"total (skill,task) flips: {sum(len(v) for v in solved.values())}")
