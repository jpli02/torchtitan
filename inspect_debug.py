import json, glob
f = sorted(glob.glob("/tmp/tb_runs/ds_smoke/*/*/agent-logs/episode-0/debug.json"))[0]
d = json.load(open(f))
print("keys:", list(d.keys()))
for k, v in d.items():
    vs = json.dumps(v) if not isinstance(v, str) else v
    print(f"--- {k} ({len(vs)} chars) ---")
    print(vs[:1200])
