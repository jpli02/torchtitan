"""Trajectory statistics for Terminal-Bench run dirs: python tb_traj_stats.py ROOT run1 [run2 ...] [--tasks]
Per trial (agent-logs/episode-N/{prompt,response}.txt): episodes, response chars (mean), prompt chars at the last
episode, estimated max context tokens (chars / 3.5), repeated responses (identical to the previous episode), wall time
(first to last response mtime). Aggregates per run and splits outcomes by whether the context passed ~4,096 tokens.
--tasks prints one line per trial of the first run."""
import collections
import glob
import json
import os
import statistics
import sys

root = sys.argv[1]
runs = [a for a in sys.argv[2:] if not a.startswith("--")]
show_tasks = "--tasks" in sys.argv
CHARS_PER_TOK = 3.5


def trial_stats(tdir):
    eps = sorted(glob.glob(f"{tdir}/agent-logs/episode-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resp, prompt_chars, times, repeats, prev = [], [], [], 0, None
    for e in eps:
        try:
            r = open(f"{e}/response.txt", errors="replace").read()
            times.append(os.path.getmtime(f"{e}/response.txt"))
            try:  # the full conversation sent to the model (prompt.txt holds only the newest user turn)
                dbg = json.load(open(f"{e}/debug.json"))
                p = sum(len(str(m.get("content", ""))) for m in dbg.get("input", []))
            except Exception:
                p = os.path.getsize(f"{e}/prompt.txt")
        except OSError:
            continue
        resp.append(len(r))
        prompt_chars.append(p)
        if prev is not None and r.strip() == prev.strip():
            repeats += 1
        prev = r
    try:
        res = json.load(open(f"{tdir}/results.json"))
    except Exception:
        res = {}
    out = {"task": res.get("task_id") or os.path.basename(tdir.rstrip("/")).split(".")[0], "resolved": bool(res.get("is_resolved")),
           "mode": res.get("failure_mode") or ("RESOLVED" if res.get("is_resolved") else "failed_tests"), "episodes": len(resp),
           "resp_mean": statistics.mean(resp) if resp else 0, "last_prompt_chars": prompt_chars[-1] if prompt_chars else 0,
           "max_ctx_tok": (max(prompt_chars) / CHARS_PER_TOK) if prompt_chars else 0, "repeats": repeats,
           "wall_s": (times[-1] - times[0]) if len(times) > 1 else 0}
    return out


for run in runs:
    trials = [trial_stats(t) for t in sorted(glob.glob(f"{root}/{run}/*/*/")) if os.path.isdir(f"{t}/agent-logs")]
    if not trials:
        print(f"{run}: no trials")
        continue
    n = len(trials)
    ok = sum(t["resolved"] for t in trials)
    modes = collections.Counter(t["mode"] for t in trials)
    long_ctx = [t for t in trials if t["max_ctx_tok"] > 4096]
    short_ctx = [t for t in trials if t["max_ctx_tok"] <= 4096]
    print(f"\n== {run}: {n} trials, resolved {ok} ({100*ok/n:.0f}%), modes {dict(modes)}")
    print(f"   episodes/trial mean {statistics.mean(t['episodes'] for t in trials):.1f} median {statistics.median(t['episodes'] for t in trials):.0f}; "
          f"response chars mean {statistics.mean(t['resp_mean'] for t in trials):.0f}; "
          f"repeated responses/trial {statistics.mean(t['repeats'] for t in trials):.1f}; "
          f"wall s/episode {statistics.mean(t['wall_s']/max(t['episodes'],1) for t in trials):.0f}")
    print(f"   max context > 4,096 tok (est.): {len(long_ctx)}/{n} trials; resolved {sum(t['resolved'] for t in long_ctx)}/{len(long_ctx)} vs "
          f"{sum(t['resolved'] for t in short_ctx)}/{len(short_ctx)} when the context stayed under 4,096")
    if show_tasks and run == runs[0]:
        for t in sorted(trials, key=lambda t: (not t["resolved"], t["task"])):
            print(f"   {t['task'][:44]:44s} {t['mode']:14s} ep={t['episodes']:3d} resp={t['resp_mean']:5.0f} ctx_max={t['max_ctx_tok']:6.0f} "
                  f"rep={t['repeats']:2d} wall={t['wall_s']:5.0f}s")
