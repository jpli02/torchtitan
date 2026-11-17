"""Table of teacher-forced token loss / accuracy per checkpoint and validation group (results of lr_debug_fit.sh).

  python lr_debug_fit_report.py [name ...]        default: every <job tmp>/lrval_*.json, in a fixed training order
"""
import glob
import json
import os
import sys

T = "/home/jli199/.claude/jobs/c0d2da0a/tmp"
GROUPS = ["oracle_test", "r2_unseen", "replay_unseen", "train_new", "train_replay"]
ORDER = ["base", "v3s5000", "v3s10000", "v3s15000", "v3", "r1", "ep1", "ep2", "ep3", "ep4", "ep5", "ep6", "ep7", "ep8"]


def load(name):
    d = json.load(open(f"{T}/lrval_{name}.json"))
    out = {}
    for g in GROUPS:
        rs = [r for r in d["rows"] if r["group"] == g and r["sup"] > 0]
        sup = sum(r["sup"] for r in rs)
        if sup:
            out[g] = (sum(r["ce"] * r["sup"] for r in rs) / sup, sum(r["acc"] * r["sup"] for r in rs) / sup, len(rs))
    return out


def main():
    names = sys.argv[1:] or [os.path.basename(p)[6:-5] for p in glob.glob(f"{T}/lrval_*.json")]
    names = sorted(set(names), key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    print("teacher-forced token loss (accuracy %) per checkpoint; oracle_test = oracle runs of the TEST tasks, r2_unseen = new runs of the "
          "training tasks, replay_unseen = v3 training rows not replayed, train_* = rows trained on")
    print(f"{'checkpoint':16s}" + "".join(f"{g:>22s}" for g in GROUPS))
    for n in names:
        try:
            s = load(n)
        except Exception as e:  # noqa: BLE001
            print(f"{n:16s} unreadable: {repr(e)[:60]}")
            continue
        print(f"{n:16s}" + "".join((f"{s[g][0]:13.3f} ({100 * s[g][1]:5.1f})" if g in s else f"{'':22s}") for g in GROUPS))


if __name__ == "__main__":
    main()
