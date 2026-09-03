"""Show a real turn from our model vs a real turn from the batching-dense data."""
import json, glob, re, sys

def extract(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def show(tag, d, src):
    if not d:
        print(f"--- {tag}: unparseable ---"); return
    cmds = d.get("commands") or []
    print(f"=== {tag} ({src}) -- {len(cmds)} command(s) ===")
    for k in ("analysis", "plan"):
        v = str(d.get(k, ""))
        print(f'  "{k}": {v[:150]}{"..." if len(v) > 150 else ""}')
    print('  "commands": [')
    for c in cmds:
        ks = str(c.get("keystrokes", "")).replace("\n", "\\n")
        print(f'      {{"keystrokes": "{ks[:70]}", "duration": {c.get("duration")}}},')
    print("  ],")
    print(f'  "task_complete": {d.get("task_complete")}')
    print()

# --- our model, from the cleanest 80-task run ---
best = None
for f in glob.glob("/tmp/tb_runs/e80_c10k80_s*/*/*/agent-logs/episode-*/debug.json"):
    try:
        j = json.loads(json.load(open(f))["original_response"])
        msg = j["choices"][0]["message"].get("content")
        txt = msg if isinstance(msg, str) else "".join(b.get("text", "") for b in (msg or []))
        d = extract(txt or "")
        if d and isinstance(d.get("commands"), list):
            best = (d, f.split("/tmp/tb_runs/")[1].split("/")[1])
            break
    except Exception:
        pass
if best:
    show("OUR MODEL (continue10k)", best[0], f"task={best[1]}")

# --- training data, a batching-dense row ---
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from torchtitan.hf_datasets.text_datasets import _load_terminal_agent_sft_dataset

ds = _load_terminal_agent_sft_dataset("unused")
shown = 0
for row in ds:
    for m in (row.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        d = extract(m.get("content") or "")
        if d and isinstance(d.get("commands"), list) and len(d["commands"]) >= 3:
            show("TRAINING DATA (batching-dense)", d, "TerminalTraj/Nemotron")
            shown += 1
            break
    if shown >= 2:
        break
