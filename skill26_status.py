import glob, os
base = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
done = []; aborted = []; other = []
for f in sorted(glob.glob(os.path.join(base, "skill26_*.txt"))):
    if f.endswith("_launch.log"):
        continue
    t = open(f).read()
    name = os.path.basename(f)[len("skill26_"):-len(".txt")]
    if "pass@k on skill" in t:
        n = t.count("<-- SOLVED")
        done.append((name, n))
    elif "ABORT" in t:
        reason = [l for l in t.splitlines() if "ABORT" in l]
        aborted.append((name, reason[0][:70] if reason else "?"))
    else:
        other.append(name)
print("DONE:", done)
print("ABORTED:", len(aborted))
for n, r in aborted:
    print(f"   {n}: {r}")
print("OTHER:", other)
