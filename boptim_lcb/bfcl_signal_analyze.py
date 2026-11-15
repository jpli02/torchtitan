"""Which signal ranks the tokens that may leave early? Reads bfcl_signal_dump.py's output (teacher-forced, full-depth answers).

A token is SAFE after loop s when the head's prediction after loop s already equals the full-depth token. For the exit gate (P_s),
the head's top-1 probability (conf_s) and its margin to the runner-up (margin_s) this prints:
  1. how many tokens are safe after each loop, and the oracle (leave at the first safe loop);
  2. the error among the tokens a signal would release first, at fixed release shares, after loop 1 and after loop 2;
  3. offline routers: executed loops against (a) the share of released tokens that are wrong and (b) the share of ITEMS with at
     least one wrong released token, for gate thresholds, confidence thresholds, margin thresholds and a stability rule
     (leave when two consecutive loops agree). Real accuracy needs real decoding; (b) bounds how many answers can change.
  python bfcl_signal_analyze.py [signal_probs.jsonl]        -> prints the report, writes <work>/signal_analysis.json"""
import json
import sys

import numpy as np

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{WORK}/signal_probs.jsonl"
P, conf, margin, pred, target, item = [], [], [], [], [], []
for k, line in enumerate(open(SRC)):
    r = json.loads(line)
    n = r["n"]
    P.extend(r["P"])
    conf.extend(zip(*r["conf"]))
    margin.extend(zip(*r["margin"]))
    pred.extend(zip(*r["pred"]))
    target.extend(r["target"])
    item.extend([k] * n)
P, conf, margin, pred, target, item = (np.array(a) for a in (P, conf, margin, pred, target, item))
N, n_items = len(target), int(item.max()) + 1
safe = pred == target[:, None]                      # [tokens, 4]
out = {"tokens": N, "items": n_items}
print(f"{N} answer tokens of {n_items} items")
print("1. safe after loop 1 / 2 / 3 / 4: " + " / ".join(f"{100 * safe[:, s].mean():.1f} %" for s in range(4)))
first = np.where(safe.any(1), safe.argmax(1), 3) + 1
print(f"   oracle (leave at the first loop whose prediction is already right): {first.mean():.3f} loops; "
      + " / ".join(f"{100 * (first == s).mean():.0f} %" for s in (1, 2, 3, 4)) + " of the tokens leave after loops 1 / 2 / 3 / 4")
out["safe_share"] = [float(safe[:, s].mean()) for s in range(4)]
out["oracle_loops"] = float(first.mean())

print("\n2. error among the tokens released first, by signal (share of released tokens whose prediction is wrong, %)")
cov = [0.057, 0.086, 0.126, 0.176, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
out["precision"] = {}
for s in (0, 1):
    sig = {"gate": P[:, s], "confidence": conf[:, s], "margin": margin[:, s]}
    print(f"   after loop {s + 1} (released share -> " + ", ".join(sig) + ")")
    out["precision"][f"loop{s + 1}"] = {}
    for c in cov:
        k = max(1, int(round(c * N)))
        row = []
        for name, v in sig.items():
            idx = np.argsort(-v, kind="stable")[:k]
            row.append(100 * (1 - safe[idx, s].mean()))
        out["precision"][f"loop{s + 1}"][str(c)] = [round(float(x), 3) for x in row]
        print(f"     {100 * c:5.1f} %  ->  " + "   ".join(f"{x:6.2f}" for x in row))


def route(exit_mask):
    """exit_mask [tokens, 3]: may the token leave after loop 1, 2, 3 -> loops, wrong-release share, items with a wrong release."""
    m = np.concatenate([exit_mask, np.ones((N, 1), bool)], 1)
    s = m.argmax(1)
    wrong = ~safe[np.arange(N), s]
    bad_items = np.zeros(n_items, bool)
    np.logical_or.at(bad_items, item, wrong)
    return float((s + 1).mean()), float(100 * wrong.mean()), float(100 * bad_items.mean())


print("\n3. offline routers: loops | wrong released tokens % | items with at least one wrong release %")
out["routers"] = []


def show(label, mask):
    L, w, b = route(mask)
    out["routers"].append({"router": label, "loops": round(L, 4), "wrong_tokens_pct": round(w, 3), "items_changed_pct": round(b, 2)})
    print(f"   {label:46s} {L:.3f} | {w:5.2f} | {b:5.1f}")


for q in (0.7, 0.5, 0.3, 0.25, 0.2):
    show(f"gate {q}", P >= q)
for q in ((0.252, 0.232, 0.067), (0.467, 0.113, 0.082), (0.333, 0.2, 0.2)):
    show(f"gate thresholds {q}", P >= np.array(q))
for t in (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5):
    show(f"confidence >= {t}", conf[:, :3] >= t)
for t in (0.99, 0.95, 0.9, 0.8, 0.6, 0.4):
    show(f"margin >= {t}", margin[:, :3] >= t)
stable = np.zeros((N, 3), bool)
stable[:, 1] = pred[:, 1] == pred[:, 0]
stable[:, 2] = pred[:, 2] == pred[:, 1]
show("two consecutive loops agree (from loop 2)", stable)
for t in (0.9, 0.7, 0.5):
    m = stable & (conf[:, :3] >= t)
    show(f"two loops agree and confidence >= {t}", m)
    m2 = m.copy()
    m2[:, 0] = conf[:, 0] >= 0.99
    show(f"  + leave after loop 1 when confidence >= 0.99", m2)
json.dump(out, open(f"{WORK}/signal_analysis.json", "w"), indent=1)
print("\nwrote", f"{WORK}/signal_analysis.json")
