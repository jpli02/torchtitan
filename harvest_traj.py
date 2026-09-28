"""Harvest verified trajectories from tb generation runs into SFT rows.

Keeps only is_resolved trials. For each, reconstructs the full terminus-2
conversation from the agent logs: terminus-2 accumulates the chat, so the
highest-numbered episode's debug.json 'messages' holds every prior turn
(system+user, assistant, user, ...) and its 'original_response' is the final
assistant turn. Result is native terminus-2 JSON -- the exact format the eval
agent emits -- so no rendering is needed, unlike the scraped-leaderboard set.

Emits {"messages":[...]} rows: the accumulated messages + the final response,
with the leading system/user turns merged to a single user turn to match the
tb80sft prompt shape. Verifies each assistant turn parses as terminus-2 JSON.

Usage: python harvest_traj.py RUNID [RUNID ...] --out FILE.jsonl
"""
import argparse
import glob
import json
import os
import re


def load_response_text(dbg):
    try:
        j = json.loads(dbg["original_response"])
        msg = j["choices"][0]["message"].get("content")
        return msg if isinstance(msg, str) else "".join(
            b.get("text", "") for b in (msg or []))
    except Exception:
        return None


def valid_terminus(s):
    m = re.search(r"\{.*\}", s or "", re.S)
    if not m:
        return False
    try:
        o = json.loads(m.group(0))
        return isinstance(o.get("commands"), list) or o.get("task_complete") is True
    except Exception:
        return False


def harvest_trial(trial_dir):
    eps = sorted(glob.glob(f"{trial_dir}/agent-logs/episode-*"),
                 key=lambda p: int(re.search(r"episode-(\d+)", p).group(1)))
    if not eps:
        return None
    last = f"{eps[-1]}/debug.json"
    if not os.path.isfile(last):
        return None
    try:
        dbg = json.load(open(last))
    except Exception:
        return None
    msgs = dbg.get("messages") or dbg.get("input")
    if not msgs:
        return None
    conv = [{"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in msgs if m.get("content")]
    final = load_response_text(dbg)
    if final:
        conv.append({"role": "assistant", "content": final})
    # need at least user + a few assistant turns, all assistant turns valid JSON
    a = [m for m in conv if m["role"] == "assistant"]
    if len(a) < 2 or not all(valid_terminus(m["content"]) for m in a):
        return None
    # merge any leading system into the first user turn (tb80sft shape)
    if conv and conv[0]["role"] == "system":
        if len(conv) > 1 and conv[1]["role"] == "user":
            conv[1]["content"] = conv[0]["content"].rstrip() + "\n\n" + conv[1]["content"]
            conv = conv[1:]
        else:
            conv[0]["role"] = "user"
    return conv, len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runids", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    turns = []
    per_task = {}
    for rid in args.runids:
        for rf in glob.glob(f"/tmp/tb_runs/{rid}/*/*/results.json"):
            try:
                d = json.load(open(rf))
            except Exception:
                continue
            if not d.get("is_resolved"):
                continue
            trial = rf.rsplit("/", 1)[0]
            got = harvest_trial(trial)
            if got:
                conv, n = got
                rows.append({"messages": conv})
                turns.append(n)
                t = d.get("task_id", "?")
                per_task[t] = per_task.get(t, 0) + 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    med = sorted(turns)[len(turns) // 2] if turns else 0
    print(f"harvested {len(rows)} verified trajectories over {len(per_task)} tasks "
          f"-> {args.out}")
    print(f"assistant turns: median {med}  max {max(turns) if turns else 0}")
    print(f"per-task: {dict(sorted(per_task.items(), key=lambda kv: -kv[1])[:8])}")


if __name__ == "__main__":
    main()
