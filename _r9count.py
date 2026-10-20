import json, glob
g = glob.glob('/tmp/tb_runs/genflash300_r9/*/*/results.json')
r = 0
for f in g:
    try:
        if json.load(open(f)).get('is_resolved'): r += 1
    except Exception:
        pass
print(f"graded={len(g)} resolved={r}")
