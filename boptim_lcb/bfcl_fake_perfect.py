"""Checker self-test for BFCL: write a generations file whose answers are built from the answer key (first allowed value of every
required parameter), to be graded by bfcl_tool.py grade inside the sandbox. A correct pipeline scores close to 100 %."""
import json
import sys

prep, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(prep) if l.strip()]
lines = [json.dumps({"summary": {"mean_loops": 4.0}})]
for i, r in enumerate(rows):
    calls = []
    for call in r["possible_answer"]:
        for fn, params in call.items():
            args = ", ".join(f"{k}={v[0]!r}" for k, v in params.items() if "" not in v)
            calls.append(f"{fn}({args})")
    lines.append(json.dumps({"i": i, "gen": "[" + ", ".join(calls) + "]", "pred": None}))
open(out, "w").write("\n".join(lines) + "\n")
print("fake generations:", len(lines) - 1, "| example:", json.loads(lines[1])["gen"][:140])
