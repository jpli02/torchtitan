"""Install the LiveCodeBench router objective of this worktree in boptim-agent with its 2026-11-21 options: copy the file, pass
ckpt / data / cost / fill_maps / q1_lo from the factory, add the matching --ouro_lcb_* flags to main.py. Idempotent; files are
replaced atomically, and the defaults keep the old behaviour, so a search that is already running is not affected.

Also writes the search set of the improved protocol: LiveCodeBench releases v1-v4, easy + medium (517 problems, none of the
342 test problems; randrouter/prep_lcb_v1v4.py) -> /home/jli199/boptim_scratch/randrouter/lcb_v1v4_em.jsonl.

  /home/jli199/boptim-agent/.venv/bin/python boptim_lcb/setup_lcb_options.py
"""
import json
import os
import py_compile
import shutil
import tempfile

BO = "/home/jli199/boptim-agent"
SRC = "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/boptim_lcb/ouro_lcb_router.py"
V1V4 = "/home/jli199/boptim_scratch/randrouter/lcb_v1v4.jsonl"
EM = "/home/jli199/boptim_scratch/randrouter/lcb_v1v4_em.jsonl"


def replace(path: str, text: str):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    py_compile.compile(tmp, doraise=True)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


replace(f"{BO}/objective/ouro_lcb_router.py", open(SRC).read())

p = f"{BO}/objective/__init__.py"
s = open(p).read()
if "ouro_lcb_cost" not in s:
    a = '            work_dir=getattr(args, "ouro_lcb_work_dir", None),\n            seed=args.objective_seed,\n        )\n    elif name in ("ouro", "ouro_humaneval"'
    assert s.count(a) == 1, "factory anchor of ouro_lcb_router not found"
    b = ('            work_dir=getattr(args, "ouro_lcb_work_dir", None),\n'
         '            seed=args.objective_seed,\n'
         '            **({"ckpt": args.ouro_lcb_ckpt} if getattr(args, "ouro_lcb_ckpt", None) else {}),\n'
         '            **({"data": args.ouro_lcb_data} if getattr(args, "ouro_lcb_data", None) else {}),\n'
         '            cost=str(getattr(args, "ouro_lcb_cost", "speed") or "speed"),\n'
         '            fill_maps=getattr(args, "ouro_lcb_fill_maps", None),\n'
         '            q1_lo=float(getattr(args, "ouro_lcb_q1_lo", 0.05)),\n'
         '        )\n    elif name in ("ouro", "ouro_humaneval"')
    s = s.replace(a, b, 1)
    replace(p, s)
    print("factory of ouro_lcb_router passes ckpt / data / cost / fill_maps / q1_lo")
else:
    print("factory already passes the new options")

p = f"{BO}/main.py"
s = open(p).read()
if "--ouro_lcb_cost" not in s:
    a = '    parser.add_argument("--ouro_lcb_gpus", type=str, default=None,'
    assert s.count(a) == 1, "main.py anchor not found"
    block = ('    parser.add_argument("--ouro_lcb_ckpt", type=str, default=None, help="[ouro_lcb_router] model directory (default: the released model\'s serving dir)")\n'
             '    parser.add_argument("--ouro_lcb_data", type=str, default=None, help="[ouro_lcb_router] problem file (default: the 342 test problems)")\n'
             '    parser.add_argument("--ouro_lcb_cost", type=str, default="speed", help="[ouro_lcb_router] speed term: speed (measured tok/s) or loops (executed loops per token)")\n'
             '    parser.add_argument("--ouro_lcb_fill_maps", type=str, default=None, help="[ouro_lcb_router] calibrated cache fill (needs the exit-lab model as ckpt)")\n'
             '    parser.add_argument("--ouro_lcb_q1_lo", type=float, default=0.05, help="[ouro_lcb_router] lower bound of q1 in steps mode")\n')
    s = s.replace(a, block + a, 1)
    replace(p, s)
    print("added --ouro_lcb_ckpt/data/cost/fill_maps/q1_lo to main.py")
else:
    print("main.py already has the new flags")

if not os.path.exists(EM):
    rows = [json.loads(line) for line in open(V1V4) if line.strip()]
    em = [r for r in rows if r.get("difficulty") in ("easy", "medium")]
    with open(EM, "w") as f:
        for r in em:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {EM}: {len(em)} problems (easy {sum(r['difficulty'] == 'easy' for r in em)}, medium {sum(r['difficulty'] == 'medium' for r in em)})")
else:
    print(f"{EM} exists ({sum(1 for _ in open(EM))} problems)")
