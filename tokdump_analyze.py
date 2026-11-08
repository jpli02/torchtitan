"""Token-accuracy distribution of an SFT checkpoint on its training rows, from fit_eval_dump.py output
({"i", "tokens": [[piece, role, correct, pos, pred_piece], ...]} per row; pred_piece is set when the token is wrong).

  python tokdump_analyze.py TMP/tokdump_r1.jsonl [TMP/tokdump_ep4.jsonl ...] --out TMP/tokdump_summary.json

Reports, per dump: token accuracy by role (command / prose / other), the per-row accuracy distribution (quantiles),
accuracy by sequence position, command spans: fully-correct share, where in the span the first error falls,
error rate and error share by token class inside commands, and the most frequent gold -> predicted confusions.
"""
import argparse
import collections
import json
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+")
ap.add_argument("--out", default=None)
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()


def tok_class(piece):
    p = piece.strip()
    if not p:
        return "whitespace"
    if re.fullmatch(r"[|&;<>()\[\]{}$`'\"\\,:=.*+/-]+", p):
        return "punct/operator"
    if p.startswith("-"):
        return "flag"
    if re.fullmatch(r"\d+", p):
        return "number"
    if "/" in p or re.fullmatch(r"\.[a-zA-Z0-9]+", p):
        return "path"
    if re.fullmatch(r"[A-Za-z_]+", p):
        return "word"
    return "mixed"


def quantiles(xs, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {f"p{int(q * 100)}": round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 4) for q in qs}


def analyze(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_role = collections.defaultdict(lambda: [0, 0])
    row_acc, row_cmd_acc = [], []
    pos_bins = collections.defaultdict(lambda: [0, 0])
    span_total = span_full = 0
    first_err_idx = collections.Counter()
    span_len = []
    cls_stats = collections.defaultdict(lambda: [0, 0])
    conf_cmd, conf_prose = collections.Counter(), collections.Counter()
    for row in rows:
        toks = row["tokens"]
        ok_all = [t[2] for t in toks]
        row_acc.append(sum(ok_all) / max(1, len(ok_all)))
        cmd_ok = [t[2] for t in toks if t[1] == "command"]
        if cmd_ok:
            row_cmd_acc.append(sum(cmd_ok) / len(cmd_ok))
        for piece, role, ok, pos, pred in toks:
            by_role[role][0] += ok
            by_role[role][1] += 1
            b = "<2k" if pos < 2048 else "2-4k" if pos < 4096 else "4-8k" if pos < 8192 else "8k+"
            pos_bins[b][0] += ok
            pos_bins[b][1] += 1
            if role == "command":
                c = tok_class(piece)
                cls_stats[c][0] += (not ok)
                cls_stats[c][1] += 1
                if not ok:
                    conf_cmd[(piece, pred)] += 1
            elif role == "prose" and not ok:
                conf_prose[(piece, pred)] += 1
        # command spans = maximal runs of consecutive command tokens
        span = []
        for t in toks + [[None, "end", 1, 0, ""]]:
            if t[1] == "command":
                span.append(t)
                continue
            if span:
                span_total += 1
                span_len.append(len(span))
                errs = [k for k, s in enumerate(span) if not s[2]]
                if not errs:
                    span_full += 1
                else:
                    k = errs[0]
                    first_err_idx["first" if k == 0 else "2nd-3rd" if k < 3 else "4th-8th" if k < 8 else "9th+"] += 1
                span = []
    n_cmd_err = sum(v[0] for v in cls_stats.values())
    rep = {
        "dump": path,
        "rows": len(rows),
        "tokens": sum(v[1] for v in by_role.values()),
        "acc_by_role": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(by_role.items())},
        "row_acc_quantiles": quantiles(row_acc),
        "row_acc_min_max": [round(min(row_acc), 4), round(max(row_acc), 4)],
        "row_cmd_acc_quantiles": quantiles(row_cmd_acc),
        "rows_cmd_acc_below_0.9": sum(a < 0.9 for a in row_cmd_acc),
        "acc_by_position": {k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(pos_bins.items(), key=lambda kv: ["<2k", "2-4k", "4-8k", "8k+"].index(kv[0]))},
        "command_spans": {"n": span_total, "fully_correct": span_full, "fully_correct_pct": round(100 * span_full / max(1, span_total), 1),
                          "mean_len": round(statistics.mean(span_len), 1) if span_len else None,
                          "first_error_position": dict(first_err_idx)},
        "command_error_by_class": {c: {"n": v[1], "err_rate": round(v[0] / v[1], 4), "share_of_errors": round(v[0] / max(1, n_cmd_err), 3)}
                                   for c, v in sorted(cls_stats.items(), key=lambda kv: -kv[1][0])},
        "top_command_confusions": [{"gold": g, "pred": p, "n": n} for (g, p), n in conf_cmd.most_common(args.top)],
        "top_prose_confusions": [{"gold": g, "pred": p, "n": n} for (g, p), n in conf_prose.most_common(args.top)],
    }
    return rep


reports = [analyze(p) for p in args.dumps]
for rep in reports:
    print(f"\n=== {rep['dump']} : {rep['rows']} rows, {rep['tokens']} assistant tokens")
    print("  acc by role:", {k: f"{v['acc']*100:.1f}% (n={v['n']})" for k, v in rep["acc_by_role"].items()})
    print("  per-row acc quantiles:", rep["row_acc_quantiles"], "min/max", rep["row_acc_min_max"])
    print("  per-row COMMAND acc quantiles:", rep["row_cmd_acc_quantiles"], f"rows <90%: {rep['rows_cmd_acc_below_0.9']}")
    print("  acc by position:", {k: f"{v['acc']*100:.1f}% (n={v['n']})" for k, v in rep["acc_by_position"].items()})
    cs = rep["command_spans"]
    print(f"  command spans: {cs['n']} (mean {cs['mean_len']} tokens), fully correct {cs['fully_correct']} = {cs['fully_correct_pct']}%, first error at {cs['first_error_position']}")
    print("  command errors by token class:")
    for c, v in rep["command_error_by_class"].items():
        print(f"    {c:15s} n={v['n']:6d} err_rate={v['err_rate']*100:5.2f}%  share_of_errors={v['share_of_errors']*100:4.1f}%")
    print("  top command confusions (gold -> pred):", [(d["gold"], d["pred"], d["n"]) for d in rep["top_command_confusions"][:15]])
    print("  top prose confusions (gold -> pred):", [(d["gold"], d["pred"], d["n"]) for d in rep["top_prose_confusions"][:15]])
if args.out:
    json.dump(reports, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
