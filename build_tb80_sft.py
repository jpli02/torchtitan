"""Build the TB-80-targeted SFT set.

What the experiments established, and what this encodes:
  oracle12   the model executes correct demonstrations perfectly (12/12)
  oracle68   one-shot answers transfer SHAPE, not skill (3/24, 2 turns/trial)
  traj20k    verified multi-turn trajectories at 20k steps = best real result
So the target is verified, MULTI-TURN, terminus-2-format demonstrations on
tasks distributed like terminal-bench, and nothing single-shot.

Two sources:

  ON-DIST   yoonholee/terminalbench-trajectories -- real agent runs on
            Terminal-Bench 2.0 scraped from the tbench.ai leaderboard, with
            reward=1 meaning the task's tests passed. Filtered to:
              * tasks NOT among our 80 (27 of TB-2's 89 overlap by name)
              * reward == 1
              * terminus-2 (native format) and mini-swe-agent (bash-only,
                converted); other scaffolds use file-edit tools
              * pointer-free: the scrape replaced some strings with '$<hex>'
                page-serializer references that no card or file resolves, so
                any row with one in an agent msg/cmd/obs is dropped (v1)
            Steps are stored parsed (msg='Analysis: .. Plan: ..', tools=[bash]),
            so each is RE-RENDERED into the exact terminus-2 JSON the eval agent
            parses. User-slot pointers are the task text; TB-2's own task.yaml
            (tb2_meta.json) supplies it. Capped per task, upsampled x UPSAMPLE.

  BULK      m-a-p/TerminalTraj, finished trajectories only (the traj20k recipe),
            REWEIGHTED by keyword-classified domain toward TB-80's category mix
            (it is 40% sysadmin where TB wants 16%, ~2% data-science and
            model-training where TB wants ~10% each).

Output: OUT/train.jsonl (shuffled) + OUT/manifest.json.
Env: OUT UPSAMPLE=5 PER_TASK_CAP=40 TT_TARGET=12000 SEED=42 DRY=0
"""
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
from measure_corpus import TB_DIST, classify, task_text  # noqa: E402
from torchtitan.hf_datasets.text_datasets import (  # noqa: E402
    _ends_complete, _normalise_terminal_messages,
)

from datasets import load_dataset  # noqa: E402

OUT = os.environ.get("OUT", "/home/jli199/boptim_scratch/tb80sft")
UPSAMPLE = int(os.environ.get("UPSAMPLE", "5"))
PER_TASK_CAP = int(os.environ.get("PER_TASK_CAP", "40"))
TT_TARGET = int(os.environ.get("TT_TARGET", "12000"))
SEED = int(os.environ.get("SEED", "42"))
DRY = os.environ.get("DRY", "0") == "1"
TB = "/home/jli199/terminal_bench_eval/tb_tasks"
OURS = set(d for d in os.listdir(TB) if os.path.isdir(os.path.join(TB, d)))
META = json.load(open("/home/jli199/boptim_scratch/tb2_meta.json"))
PTR = re.compile(r"^\$[0-9a-f]+$")
rng = random.Random(SEED)

TB2_TO_TB80 = {"mathematics": "scientific-computing", "optimization": "scientific-computing",
               "data-processing": "data-science", "data-querying": "data-science",
               "machine-learning": "model-training", "personal-assistant": "other",
               "video-processing": "other"}
CONFIRM = ("Are you sure you want to mark the task as complete? This will trigger your "
           "solution to be graded and you won't be able to make any further corrections. "
           'If so, include "task_complete": true in your JSON response again.')
LONG = re.compile(r"apt|pip |install|make|cmake|gcc|g\+\+|compile|train|pytest|npm|cargo|docker|wget|curl")


def is_ptr(v):
    return isinstance(v, str) and PTR.match(v) is not None


def jturn(analysis, plan, cmds, complete):
    return json.dumps({"analysis": analysis, "plan": plan,
                       "commands": cmds, "task_complete": complete}, indent=2)


def keystroke(cmd):
    return cmd if cmd.endswith("\n") else cmd + "\n"


def duration(cmd):
    return 30.0 if LONG.search(cmd) else 1.0


def clean_obs(obs):
    """Drop the harness's format-warning preamble; keep the terminal part."""
    if obs is None:
        return None
    if obs.startswith("Previous response had"):
        for mark in ("New Terminal Output", "Current terminal state", "Current Terminal Screen"):
            i = obs.find(mark)
            if i >= 0:
                return obs[i:]
        return None
    return obs


def first_prompt(template, task):
    template = template.rstrip()
    if "Task Description" not in template:
        template += "\n\nTask Description:"
    return (f"{template}\n{task}\n\nCurrent terminal state:\n"
            f"Current Terminal Screen:\nroot@host:/app# ")


def render_terminus2(steps, task, template_cache):
    sysm = next((s.get("msg") for s in steps if s.get("src") == "system"), None)
    if sysm and not is_ptr(sysm) and template_cache.get("t") is None:
        template_cache["t"] = sysm
    template = template_cache.get("t")
    if not template:
        return None
    msgs = [{"role": "user", "content": first_prompt(template, task)}]
    agent = [s for s in steps if s.get("src") == "agent"]
    if len(agent) < 3:
        return None
    for i, s in enumerate(agent):
        m = s.get("msg") or ""
        tools = s.get("tools") or []
        if is_ptr(m) or any(is_ptr(t.get("cmd")) for t in tools) or is_ptr(s.get("obs")):
            return None
        mm = re.match(r"\s*Analysis:\s*(.*?)\s*Plan:\s*(.*)$", m, re.S)
        analysis, plan = (mm.group(1).strip(), mm.group(2).strip()) if mm else (m.strip(), "")
        cmds = [{"keystrokes": keystroke(t["cmd"]), "duration": duration(t["cmd"])}
                for t in tools if t.get("fn") == "bash_command" and t.get("cmd")]
        complete = any(t.get("fn") == "mark_task_complete" for t in tools)
        msgs.append({"role": "assistant", "content": jturn(analysis, plan, cmds, complete)})
        obs = clean_obs(s.get("obs"))
        last = i == len(agent) - 1
        if not last:
            if obs is None:
                return None
            msgs.append({"role": "user", "content": obs})
    # must end in a confirmed completion
    finals = [s for s in agent[-2:] if any(t.get("fn") == "mark_task_complete" for t in (s.get("tools") or []))]
    if not finals:
        return None
    if len(finals) == 1:
        msgs.append({"role": "user", "content": f"Current terminal state:\nroot@host:/app# \n\n{CONFIRM}"})
        msgs.append({"role": "assistant", "content": jturn(
            "The task has been completed and verified.", "Confirm completion.", [], True)})
    return msgs


def render_miniswe(steps, task, template):
    msgs = [{"role": "user", "content": first_prompt(template, task)}]
    agent = [s for s in steps if s.get("src") == "agent"]
    if len(agent) < 3:
        return None
    for i, s in enumerate(agent):
        m = s.get("msg") or ""
        tools = s.get("tools") or []
        obs = s.get("obs")
        if is_ptr(m) or any(is_ptr(t.get("cmd")) for t in tools) or is_ptr(obs):
            return None
        thought = re.sub(r"```bash.*?```", "", m, flags=re.S).replace("THOUGHT:", "").strip()
        cmd = next((t.get("cmd") for t in tools if t.get("fn") == "bash_command"), None)
        if cmd is None:
            return None
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            msgs.append({"role": "assistant", "content": jturn(thought or "All requirements are met.", "Mark the task complete.", [], True)})
            msgs.append({"role": "user", "content": f"Current terminal state:\nroot@host:/app# \n\n{CONFIRM}"})
            msgs.append({"role": "assistant", "content": jturn("Confirmed.", "Confirm completion.", [], True)})
            return msgs
        msgs.append({"role": "assistant", "content": jturn(
            thought, "", [{"keystrokes": keystroke(cmd), "duration": duration(cmd)}], False)})
        if i == len(agent) - 1:
            return None  # never reached the submit marker
        if obs is None:
            return None
        out = re.sub(r"</?returncode>.*?</?returncode>|<returncode>\d+</returncode>", "", str(obs), flags=re.S)
        out = re.sub(r"</?output>", "", out).strip()
        msgs.append({"role": "user", "content": f"New Terminal Output:\n{out[:5000]}\nroot@host:/app# "})
    return None


# ------------------------------------------------------------------ on-dist
def build_ondist():
    ds = load_dataset("yoonholee/terminalbench-trajectories", split="train", streaming=True)
    tcache = {}
    rows = collections.defaultdict(list)   # task -> [(source, msgs)]
    seen = counted = 0
    for r in ds:
        seen += 1
        t = r["task_name"]
        if t in OURS or int(r.get("reward") or 0) != 1 or r["agent"] not in ("terminus-2", "mini-swe-agent"):
            continue
        s = r.get("steps")
        if s in (None, "null", ""):
            continue
        try:
            steps = json.loads(s)
        except Exception:
            continue
        if not isinstance(steps, list):
            continue
        meta = META.get(t, {})
        task = meta.get("instruction") or ""
        umsg = next((x.get("msg") for x in steps if x.get("src") == "user"), None)
        if umsg and not is_ptr(umsg) and len(umsg) > 40:
            task = umsg
        if not task:
            continue
        if r["agent"] == "terminus-2":
            msgs = render_terminus2(steps, task, tcache)
        else:
            if not tcache.get("t"):
                continue
            msgs = render_miniswe(steps, task, tcache["t"])
        if msgs:
            rows[t].append((r["agent"], msgs))
            counted += 1
        if DRY and counted >= 6:
            break
    return rows, seen, tcache.get("t")


# --------------------------------------------------------------------- bulk
def build_bulk():
    if DRY or TT_TARGET <= 0:
        return [], {}
    ds = load_dataset("m-a-p/TerminalTraj", split="train", streaming=True)
    # pass 1: domain histogram on a sample
    hist = collections.Counter()
    sample = []
    for i, r in enumerate(ds):
        msgs = _normalise_terminal_messages(r.get("messages"))
        if not msgs or not _ends_complete(msgs):
            continue
        d = classify(task_text(msgs))
        hist[d] += 1
        if i >= 2500:
            break
    tot = sum(hist.values())
    tb_tot = sum(TB_DIST.values())
    w = {}
    for d in list(TB_DIST) + ["other"]:
        tt = hist.get(d, 0) / max(tot, 1)
        tb = (TB_DIST.get(d, 0) / tb_tot) if d != "other" else 0.02
        w[d] = max(0.3, min(4.0, (tb / tt) if tt > 0 else 4.0))
    wmax = max(w.values())
    out = []
    kept = collections.Counter()
    for r in load_dataset("m-a-p/TerminalTraj", split="train", streaming=True):
        msgs = _normalise_terminal_messages(r.get("messages"))
        if not msgs or not _ends_complete(msgs):
            continue
        d = classify(task_text(msgs))
        if rng.random() <= w[d] / wmax:
            out.append({"source": "terminaltraj", "domain": d, "messages": msgs})
            kept[d] += 1
            if len(out) >= TT_TARGET:
                break
    return out, {"sample_hist": dict(hist), "weights": w, "kept": dict(kept)}


# --------------------------------------------------------------------- main
os.makedirs(OUT, exist_ok=True)
rows, seen, template = build_ondist()
ondist = []
per_task = {}
for t, lst in rows.items():
    rng.shuffle(lst)
    lst = lst[:PER_TASK_CAP]
    per_task[t] = len(lst)
    cat = META.get(t, {}).get("category", "other")
    dom = TB2_TO_TB80.get(cat, cat)
    for src, msgs in lst:
        ondist.append({"source": f"tb2:{src}", "task": t, "domain": dom, "messages": msgs})

if DRY:
    print(f"scanned {seen} rows; rendered {len(ondist)} on-dist rows from {len(rows)} tasks")
    if ondist:
        ex = ondist[0]["messages"]
        print(f"\n=== rendered example: task={ondist[0]['task']} source={ondist[0]['source']} turns={len(ex)} ===")
        print("--- user[0] tail ---\n" + ex[0]["content"][-420:])
        print("--- assistant[0] ---\n" + ex[1]["content"][:600])
        print("--- user[1] head ---\n" + ex[2]["content"][:220])
        print("--- assistant[-1] ---\n" + ex[-1]["content"][:300])
    h = sorted(glob.glob("/tmp/tb_runs/clean_continue10k/hello-world/*/agent-logs/episode-0/debug.json"))
    if h:
        real = [m for m in json.load(open(h[0]))["messages"] if m["role"] == "user"][0]["content"]
        print("\n=== REAL eval prompt tail (for alignment) ===\n" + real[-420:])
    sys.exit(0)

bulk, bulk_info = build_bulk()
final = []
for row in ondist:
    for _ in range(UPSAMPLE):
        final.append(row)
final.extend(bulk)
rng.shuffle(final)

path = os.path.join(OUT, "train.jsonl")
chars = 0
with open(path, "w") as f:
    for row in final:
        f.write(json.dumps({"messages": row["messages"]}) + "\n")
        chars += sum(len(m["content"]) for m in row["messages"])

dom_counts = collections.Counter(r["domain"] for r in final)
src_counts = collections.Counter(r["source"] for r in final)
manifest = {
    "rows_written": len(final), "est_tokens": chars // 4,
    "ondist_unique": len(ondist), "ondist_tasks": len(rows), "upsample": UPSAMPLE,
    "per_task_cap": PER_TASK_CAP, "per_task": per_task,
    "bulk_rows": len(bulk), "bulk": bulk_info,
    "sources": dict(src_counts), "domains": dict(dom_counts),
    "excluded_overlap_tasks": sorted(t for t, m in META.items() if m.get("overlap")),
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"wrote {len(final)} rows (~{chars // 4 / 1e6:.1f}M tokens) -> {path}")
print(f"on-dist unique {len(ondist)} over {len(rows)} tasks (x{UPSAMPLE}); bulk {len(bulk)}")
print("sources:", dict(src_counts))
print("domains:", dict(dom_counts.most_common()))
