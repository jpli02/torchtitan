"""Validation curves of the learning-rate runs (files written by torchtitan/components/val_probe.py).

  python lr_debug_valcurve.py <tag> [<tag> ...]      reads <job tmp>/valprobe_<tag>.jsonl
Prints, per run, the final-step token loss of every validation group at every validation step, and per-step losses of the
four recurrence steps for the test-task group.
"""
import collections
import json
import sys

T = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
GROUPS = ["oracle_test", "r2_unseen", "replay_unseen", "train_new", "train_replay"]


def load(tag):
    by = collections.defaultdict(dict)
    for line in open(f"{T}/valprobe_{tag}.jsonl"):
        if line.strip():
            r = json.loads(line)
            by[r["step"]][r["group"]] = r
    return dict(sorted(by.items()))


def main():
    for tag in sys.argv[1:]:
        try:
            by = load(tag)
        except FileNotFoundError:
            print(f"{tag}: no validation file")
            continue
        print(f"\n== {tag}: final-step token loss (accuracy %) per group")
        print(f"{'step':>5s} {'lr':>9s}" + "".join(f"{g:>22s}" for g in GROUPS))
        for step, d in by.items():
            lr = next(iter(d.values()))["lr"]
            print(f"{step:5d} {lr:9.2e}" + "".join((f"{d[g]['ce_final']:13.3f} ({100 * d[g]['acc_final']:5.1f})" if g in d else f"{'':22s}") for g in GROUPS))
        base = by.get(0)
        if base:
            best = {g: min(by, key=lambda s: by[s][g]["ce_final"] if g in by[s] else 9e9) for g in GROUPS if g in base}
            print("   lowest loss at step: " + ", ".join(f"{g} {s} ({by[s][g]['ce_final']:.3f}, start {base[g]['ce_final']:.3f})" for g, s in best.items()))
        last = by[max(by)]
        g = "oracle_test"
        if g in last and "ce_step" in last[g]:
            print(f"   {g} at step {max(by)}: loss per recurrence step {[round(v, 3) for v in last[g]['ce_step']]}, "
                  f"exit shares {[round(v, 3) for v in last[g]['exit_pdf']]}, mixture loss {last[g]['ce_mix']:.3f}")


if __name__ == "__main__":
    main()
