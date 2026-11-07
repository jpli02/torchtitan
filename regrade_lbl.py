"""Re-grade LiveBench language generations offline with depth_eval's current lbl functions.
Usage: python regrade_lbl.py GEN.jsonl"""
import json
import re
import sys

src = open("/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft/depth_eval.py").read()
i, j = src.index("def lbl_extract"), src.index("def lbr_norm")
ns = {"json": json, "re": re}
exec(src[i:j], ns)  # noqa: S102 - our own file
recs = [json.loads(l) for l in open(sys.argv[1])][1:]
ok_by = {}
for r in recs:
    pred = ns["lbl_extract"](r["gen"], r["subtask"])
    ok = ns["lbl_correct"](pred, r["gold"], r["subtask"])
    d = ok_by.setdefault(r["subtask"], [0, 0])
    d[0] += ok
    d[1] += 1
    print(f"{r['subtask']:18s} ok={ok!s:5s} ntok={r['ntok']:5d} pred={repr((pred or '')[:60])} gold={repr(str(r['gold'])[:40])}")
print({k: f"{v[0]}/{v[1]}" for k, v in ok_by.items()})
