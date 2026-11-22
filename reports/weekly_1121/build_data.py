"""Data files of the weekly progress page (Nov 16-21, 2026): Terminal-Bench learning-rate study, LiveCodeBench and BFCL router
experiments. Results are read from the run files where they exist; the rest are the recorded results of runs whose files were
summarised earlier (sources named in each dataset's description on the page).

  /home/jli199/boptim-agent/.venv/bin/python reports/weekly_1121/build_data.py      -> reports/weekly_1121/data/*.csv
"""
import csv
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
S = "/home/jli199/boptim_scratch"


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print(f"{name}: {len(rows)} rows")


def pct(a, b):
    return round(100.0 * a / b, 1)


# 1. Terminal-Bench test set (11 tasks x 2 trials), every checkpoint of the learning-rate study
tb = [  # key, checkpoint, trained from, recipe, solved, trials
    ("v3", "v3 (starting point)", "base", "2e-5 cosine, 20,000 steps", 8, 22),
    ("r1", "r1", "v3", "2e-5 cosine, 896 steps", 9, 22),
    ("ep1", "epoch 1 of 4", "v3", "2e-5 cosine, 4 epochs", 6, 22),
    ("ep2", "epoch 2 of 4", "v3", "2e-5 cosine, 4 epochs", 4, 20),
    ("ep4", "epoch 4", "v3", "2e-5 cosine, 4 epochs", 5, 21),
    ("ep8", "epoch 8", "v3 (via epoch 4)", "2e-5 cosine restarted, 4 more epochs", 4, 22),
    ("half", "halfway v3 to r1", "v3", "v3 + 0.5 x (r1 - v3), two evaluations", 21, 44),
    ("lr5e6", "5e-6 recipe", "v3", "5e-6 cosine, 804 steps", 9, 22),
    ("frombase", "from base, one pass", "base", "2e-5 cosine, 22,184 steps, v3 data + 112 new rows", 9, 22),
]
write("tb_test.csv", ["key", "checkpoint", "trained_from", "recipe", "solved", "trials", "pass_pct"],
      [(k, c, f, r, s, t, pct(s, t)) for k, c, f, r, s, t in tb])

# 2. held-out token loss on the 86-row validation file (lr_debug_valset.py), lower is better
held = [  # key, checkpoint, test-task oracle runs, unseen take-turns runs, v3 rows not replayed, the 112 new rows (fit)
    ("base", "released base", 0.939, 0.927, 0.806, 0.998),
    ("v3", "v3", 0.879, 0.618, 0.218, 0.656),
    ("r1", "r1", 0.958, 0.545, 0.265, 0.197),
    ("ep1", "epoch 1", 0.959, 0.575, 0.299, 0.318),
    ("ep2", "epoch 2", 1.014, 0.579, 0.317, 0.152),
    ("ep4", "epoch 4", 1.119, 0.649, 0.355, 0.033),
    ("ep8", "epoch 8", 1.283, 0.783, 0.441, 0.017),
    ("frombase", "from base, one pass", 0.909, 0.563, 0.211, 0.387),
]
write("tb_heldout.csv", ["key", "checkpoint", "test_task_runs", "unseen_runs", "v3_rows_not_replayed", "new_rows"], held)

# 3. the from-base run's in-training validation (every 1,000 steps)
rows = [json.loads(l) for l in open(f"{S}/lr_debug/valprobe_frombase_v3r1_p1.jsonl") if l.strip()]
names = {"v3_heldout": "held-out v3 rows", "r2_unseen": "unseen take-turns runs", "oracle_test": "test-task runs"}
curve = [(r["step"], names[r["group"]], round(r["ce_final"], 4)) for r in rows if r["group"] in names]
write("tb_frombase_curve.csv", ["step", "series", "loss"], sorted(curve, key=lambda x: (x[1], x[0])))

# 4. LiveCodeBench real execution, 342 test problems; paired = problems only this run solves / only the released gate at 0.3 solves
lcb = [  # key, group, system, router, loops, loops_kind, pass_pct, only_this, only_gate, sign_p
    ("full", "released model", "full depth", "1, 1, 1", 4.0, "executed", 20.5, None, None, None),
    ("depth2", "released model", "fixed depth 2", "", 2.0, "executed", 15.8, None, None, None),
    ("depth1", "released model", "fixed depth 1", "", 1.0, "executed", 5.6, None, None, None),
    ("gate03", "released model", "released gate", "0.3, 0.3, 0.3", 2.21, "executed", 17.8, None, None, None),
    ("luna_best", "threshold search", "best router of the LLM search", "0.38, 0.30, 0.34", 2.21, "executed", 18.4, 1, 0, 1.0),
    ("copy025", "threshold search", "first proposal of both LLM runs", "0.25, 0.3, 0.3", 2.02, "executed", 15.2, 4, 14, 0.031),
    ("fill025", "cache repair", "calibrated cache fill", "0.25, 0.3, 0.3", 1.99, "executed", 17.3, 6, 9, 0.61),
    ("fill0245", "cache repair", "calibrated cache fill", "0.245, 0.3, 0.3", 1.76, "executed", 15.2, 5, 15, 0.041),
    ("lora", "cache repair", "random-router LoRA, 125 steps", "0.245, 0.3, 0.3", 1.66, "executed", 15.2, 3, 13, 0.021),
    ("exact0245", "exact cache", "exact cache (no compute saved)", "0.245, 0.3, 0.3", 1.87, "would be", 19.0, 10, 7, 0.63),
    ("exact0132", "exact cache", "exact cache (no compute saved)", "0.132, 1, 1", 2.33, "would be", 18.1, 10, 10, 1.0),
    ("exact005", "exact cache", "exact cache (no compute saved)", "0.05, 1, 1", 2.22, "would be", 13.2, 3, 20, 0.0002),
]
write("lcb_runs.csv", ["key", "group", "system", "router", "loops", "loops_kind", "pass_pct", "only_this", "only_gate", "sign_p"], lcb)

# 5. the two threshold searches on all 342 problems (objective = -6.25 x measured speed-up + 2 x points below 18.3 %)
write("lcb_searches.csv", ["key", "optimizer", "new_evaluations", "best_router", "pass_pct", "speedup", "objective"],
      [("start", "released gate (shared start)", 0, "0.3, 0.3, 0.3", 18.1, 1.71, -10.31),
       ("gp", "Gaussian process (gp_hedge)", 22, "1.00, 0.38, 0.30", 18.4, 1.78, -11.10),
       ("luna", "LLM agent (gpt-6-luna)", 10, "0.38, 0.30, 0.34", 18.4, 1.80, -11.26)])

# 6. the LLM search's evaluations, with how many answers each shares with its best run
luna = [json.loads(l) for l in open(f"{S}/bo_lcb_router_pooled_luna6/evaluations.jsonl") if l.strip()]
best = min(luna, key=lambda r: r["value"])


def gens(r):
    p = os.path.join(r.get("run_dir", ""), "gen.jsonl")
    if not os.path.exists(p):
        return None
    return {g["question_id"]: g["gen"] for g in (json.loads(l) for l in open(p)) if "question_id" in g}


gb = gens(best)
lr = []
for k, r in enumerate(luna, 1):
    g = gens(r)
    same = sum(g[q] == gb[q] for q in gb if q in g) if g is not None else None
    t = r["thresholds"][:3]
    lr.append((k, f"{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}", round(r["pass1_pct"], 1), round(r.get("exec_loops") or 0, 2), round(r["speedup"], 2),
               "yes" if r.get("cal_clean") else "no", round(r["value"], 2), same))
write("lcb_llm_search.csv", ["evaluation", "router", "pass_pct", "executed_loops", "measured_speedup", "quiet_card", "objective", "answers_same_as_best"], lr)

# 7. gate NAS with BO (snapshot of the running searches)
gn = []
for bench, path in (("BFCL", f"{S}/bo_bfcl_gate_nas_gp_s1/evaluations.jsonl"), ("LiveCodeBench", f"{S}/bo_lcb_gate_nas_gp_s1/evaluations.jsonl")):
    for k, r in enumerate((json.loads(l) for l in open(path) if l.strip()), 1):
        if r.get("status") not in ("ok", "shortcut"):
            continue
        if bench == "BFCL":
            name = (r.get("extra_cfg") or {}).get("decode_router", "").rsplit("/", 1)[-1].replace(".pt", "")
            arch, g = name.rsplit("_g", 1)
            gamma = float(g.split("_s")[0])
            acc, loops = r["acc_pct"], r["exec_loops"]
        else:
            arch, gamma, acc, loops = r["router_arch"], r["gamma"], r["pass1_pct"], r["loops_eff"]
        gn.append((f"{bench[:4].lower()}{k}", bench, k, arch, round(gamma, 4), round(r["thresholds"][0], 3), round(acc, 2), round(loops, 3), round(r["value"], 2)))
gn += [("bfclref", "BFCL", "released", "released gate (threshold 0.25)", None, 0.25, 80.87, 1.911, 48.8),
       ("lcbref", "LiveCodeBench", "released", "released gate (threshold 0.3)", None, 0.3, 17.8, 2.208, -10.32)]
write("gate_nas.csv", ["key", "benchmark", "evaluation", "gate", "gamma", "threshold", "accuracy_pct", "executed_loops", "objective"], gn)

# 8. BFCL v3 held-out non-live items (575), real early exit; paired differences in points with 95 % bootstrap intervals
bh = [  # key, group, router, loops, acc, vs full depth (diff, lo, hi), vs plain gate 0.2 (diff, lo, hi, only router, only gate, p)
    ("full", "released gate", "full depth", 4.0, 80.52, None, None, None, None, None, None, None, None, None),
    ("g07", "released gate", "0.7", 3.245, 78.96, None, None, None, None, None, None, None, None, None),
    ("g05", "released gate", "0.5", 2.866, 78.61, None, None, None, None, None, None, None, None, None),
    ("g03", "released gate", "0.3", 2.129, 80.0, None, None, None, None, None, None, None, None, None),
    ("g025", "released gate", "0.25", 1.923, 79.13, None, None, None, None, None, None, None, None, None),
    ("g0244", "released gate", "0.2443", 1.714, 77.39, None, None, None, None, None, None, None, None, None),
    ("g02", "released gate", "0.2", 1.433, 74.96, None, None, None, None, None, None, None, None, None),
    ("f03", "released gate + fill", "0.3", 2.127, 80.7, 0.17, -1.57, 1.91, 5.74, 3.3, 8.35, 44, 11, 0.0001),
    ("f025", "released gate + fill", "0.25", 1.907, 79.83, -0.7, -2.78, 1.39, 4.87, 2.43, 7.3, 40, 12, 0.0001),
    ("f02", "released gate + fill", "0.2", 1.374, 77.39, -3.13, -5.57, -0.7, 2.43, 0.17, 4.7, 30, 16, 0.054),
    ("s1", "searched router + fill", "0.2372, 0.2935, 0.4755", 1.572, 80.0, -0.52, -2.78, 1.74, 5.04, 2.61, 7.48, 40, 11, 0.0001),
    ("s2", "searched router + fill", "0.2386, 0.3059, 0.4748", 1.604, 80.52, 0.0, -2.09, 2.09, 5.57, 3.3, 8.0, 41, 9, 0.0001),
    ("s3", "searched router + fill", "0.2386, 0.3084, 0.4644", 1.606, 80.52, 0.0, -2.09, 2.09, 5.57, 3.3, 8.0, 41, 9, 0.0001),
    ("s4", "searched router + fill", "0.2401, 0.2999, 0.4876", 1.626, 79.83, -0.7, -2.78, 1.39, 4.87, 2.61, 7.3, 39, 11, 0.0001),
    ("s5", "searched router + fill", "0.2358, 0.2019, 0.5019", 1.507, 78.61, -1.91, -4.35, 0.52, 3.65, 1.22, 6.26, 39, 18, 0.0075),
    ("n1", "searched router, no fill", "0.2372, 0.2935, 0.4755", 1.653, 75.83, None, None, None, None, None, None, None, None, None),
    ("n3", "searched router, no fill", "0.2386, 0.3084, 0.4644", 1.678, 76.17, None, None, None, None, None, None, None, None, None),
    ("n4", "searched router, no fill", "0.2401, 0.2999, 0.4876", 1.701, 76.0, None, None, None, None, None, None, None, None, None),
    ("n5", "searched router, no fill", "0.2358, 0.2019, 0.5019", 1.595, 75.65, None, None, None, None, None, None, None, None, None),
]
write("bfcl_heldout.csv", ["key", "group", "router", "executed_loops", "accuracy_pct", "vs_full_pts", "vs_full_lo", "vs_full_hi",
                           "vs_gate02_pts", "vs_gate02_lo", "vs_gate02_hi", "only_router", "only_gate02", "sign_p"], bh)

# 9. BFCL v3 live items (1,350), real early exit
write("bfcl_live.csv", ["key", "group", "router", "executed_loops", "accuracy_pct"], [
    ("full", "released gate", "full depth", 4.0, 64.15),
    ("g03", "released gate", "0.3", 2.258, 63.11),
    ("g025", "released gate", "0.25", 2.092, 62.0),
    ("g02", "released gate", "0.2", 1.668, 58.15),
    ("f03", "released gate + fill", "0.3", 2.253, 61.7),
    ("f025", "released gate + fill", "0.25", 2.085, 61.85),
    ("f02", "released gate + fill", "0.2", 1.605, 57.85),
    ("s1", "searched router + fill", "0.2372, 0.2935, 0.4755", 1.892, 60.37),
    ("s2", "searched router + fill", "0.2386, 0.3059, 0.4748", 1.921, 60.89),
    ("s3", "searched router + fill", "0.2386, 0.3084, 0.4644", 1.916, 60.0),
    ("s4", "searched router + fill", "0.2401, 0.2999, 0.4876", 1.95, 60.67),
    ("s5", "searched router + fill", "0.2358, 0.2019, 0.5019", 1.81, 59.7),
])
