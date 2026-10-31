"""Why did the guard refuse to send commands 176 times? Find episodes whose prompt carries the
'[guard] ... STILL RUNNING' note and show the screen's last lines just before, i.e. what the
busy detector saw (from the previous user message in debug.json)."""
import glob
import json
import re

from guard_agent import PROMPT_RE, at_prompt, last_line

hits = 0
shown = 0
tails = {}
for dbg in sorted(glob.glob("/tmp/tb_runs/easy12_guard/*/*/agent-logs/episode-*/debug.json")):
    try:
        msgs = json.load(open(dbg))["messages"]
    except Exception:
        continue
    users = [m for m in msgs if m.get("role") == "user"]
    if len(users) < 2:
        continue
    cur = users[-1]["content"]
    if "STILL RUNNING" not in cur:
        continue
    hits += 1
    # the screen the guard inspected is the tail of the output shown AFTER the note in this same message
    body = cur.split("\n\n", 1)[1] if "\n\n" in cur else cur
    ll = last_line(body)
    tails[ll[-60:]] = tails.get(ll[-60:], 0) + 1
    if shown < 4:
        shown += 1
        print(f"--- {dbg.split('/')[4]}/{dbg.split('/')[-2]}: last line seen = {ll[-100:]!r}  at_prompt={at_prompt(body)}")
        print("    tail:", repr(body[-260:]))
print(f"\nbusy-refusal episodes found: {hits}")
print("most common last lines the detector rejected:")
for k, v in sorted(tails.items(), key=lambda x: -x[1])[:8]:
    print(f"  {v:4d}  {k!r}")
