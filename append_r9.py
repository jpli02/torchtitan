import json, os, hashlib
WT = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft"
GEN = "/home/jli199/boptim_scratch/gen_out"
r8 = f"{GEN}/more_flash300_r9.jsonl"
# harvest round-8 resolved trajectories -> r8 file
os.system(f"/home/jli199/torchtitan/.venv/bin/python {WT}/harvest_traj.py genflash300_r9 --out {r8} >/dev/null 2>&1")
new = [l for l in open(r8)] if os.path.exists(r8) else []

def h(o): return hashlib.md5(json.dumps(o.get("messages"), sort_keys=True).encode()).hexdigest()

def dedup_append(target, rows):
    seen = set()
    if os.path.exists(target):
        for l in open(target):
            if l.strip():
                try: seen.add(h(json.loads(l)))
                except Exception: pass
    added = 0
    with open(target, "a") as w:
        for l in rows:
            if not l.strip(): continue
            try: o = json.loads(l)
            except Exception: continue
            hh = h(o)
            if hh in seen: continue
            seen.add(hh); w.write(l if l.endswith("\n") else l + "\n"); added += 1
    return added

a1 = dedup_append(f"{GEN}/deepseek_flash_generated.jsonl", new)
a2 = dedup_append(f"{GEN}/deepseek_generated.jsonl", new)
fl = sum(1 for _ in open(f"{GEN}/deepseek_flash_generated.jsonl"))
dg = sum(1 for _ in open(f"{GEN}/deepseek_generated.jsonl"))
print(f"round8 harvested={len(new)} appended flash+{a1} deepseek+{a2} | flash_total={fl} deepseek_total={dg}")
