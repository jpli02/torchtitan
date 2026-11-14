"""BFCL (Berkeley Function-Calling Leaderboard, ICML 2025) single-turn AST categories for depth_eval. RUNS INSIDE THE SANDBOX IMAGE
bfcl-grader (boptim_lcb/bfcl_grader.Dockerfile: the official bfcl-eval package, used offline):

  prep   python bfcl_tool.py prep RAW_DIR OUT.jsonl [PER_CAT]
         RAW_DIR holds the v3 files of gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL_v3_{simple,multiple,parallel,
         parallel_multiple,java,javascript}.json + possible_answer/). Builds the official prompt-mode prompt of every entry (the
         package's system prompt with the function docs, merged into the user turn the way the package does for models without a
         system role) -> rows {"id", "category", "language", "prompt", "function", "possible_answer"}. PER_CAT (optional) keeps a
         seeded sample per leaderboard category (simple = python + java + javascript in their 400:100:50 proportion).
  grade  python bfcl_tool.py grade PREP.jsonl GEN.jsonl OUT.json
         GEN = depth_eval --task bfcl output. Decodes each response with the package's prompt-mode decoder and checks it with the
         official AST checker. Prints: GRADED n=.. overall=.. micro=.. simple=.. multiple=.. parallel=.. parallel_multiple=.. decode_fail=..
         (overall = the leaderboard's unweighted mean over the four categories, simple being the mean of its three languages).
"""
import json
import random
import re
import sys

import bfcl_eval.model_handler.utils as mu
import bfcl_eval.utils as bu
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

CATS = [("simple", "BFCL_v3_simple.json", "python", "simple_python"), ("java", "BFCL_v3_java.json", "java", "simple_java"),
        ("javascript", "BFCL_v3_javascript.json", "javascript", "simple_javascript"), ("multiple", "BFCL_v3_multiple.json", "python", "multiple"),
        ("parallel", "BFCL_v3_parallel.json", "python", "parallel"), ("parallel_multiple", "BFCL_v3_parallel_multiple.json", "python", "parallel_multiple")]
LANG = {"python": Language.PYTHON, "java": Language.JAVA, "javascript": Language.JAVASCRIPT}
RET = {"python": ReturnFormat.PYTHON, "java": ReturnFormat.JAVA, "javascript": ReturnFormat.JAVASCRIPT}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def prep(raw, out, per_cat=None):
    rng = random.Random(0)
    rows = []
    share = {"simple": 8 / 11, "java": 2 / 11, "javascript": 1 / 11}  # 400 : 100 : 50
    for cat, fname, lang, v4cat in CATS:
        tests = load(f"{raw}/{fname}")
        answers = {a["id"]: a["ground_truth"] for a in load(f"{raw}/possible_answer/{fname}")}
        if per_cat:
            k = round(per_cat * share[cat]) if cat in share else per_cat
            rng.shuffle(tests)
            tests = tests[:k]
        for e in tests:
            functions = bu._func_doc_language_specific_pre_processing(json.loads(json.dumps(e["function"])), v4cat)
            msgs = mu.system_prompt_pre_processing_chat_model([dict(m) for m in e["question"][0]], functions, e["id"])
            merged = mu.convert_system_prompt_into_user_prompt(msgs)
            users = [m["content"] for m in merged if m["role"] == "user"]
            rows.append({"id": e["id"], "category": cat, "v4_category": v4cat, "language": lang, "prompt": "\n\n".join(users),
                         "function": functions, "possible_answer": answers[e["id"]]})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"prep: {len(rows)} rows ->", out, {c: sum(1 for r in rows if r["category"] == c) for c, *_ in CATS})


def pick_model_name():
    """A prompt-mode entry of the package's model table (the checker only reads its underscore_to_dot flag)."""
    for name, cfg in MODEL_CONFIG_MAPPING.items():
        if not getattr(cfg, "underscore_to_dot", False) and not getattr(cfg, "is_fc_model", False):
            return name
    return next(iter(MODEL_CONFIG_MAPPING))


def clean(text):
    t = (text or "").split("</think>")[-1].strip()
    m = re.findall(r"```[a-zA-Z]*\s*\n?(.*?)```", t, re.S)
    if m:
        t = m[-1].strip()
    if not t.startswith("["):  # the model reasons in plain text first (its chat template has no think tags): keep the final call list
        lines = [l.strip() for l in t.splitlines() if l.strip().startswith("[") and l.strip().endswith("]")]
        if lines:
            t = lines[-1]
    return t


def grade(prep_path, gen_path, out_path):
    rows = load(prep_path)
    gens = [g for g in load(gen_path) if "i" in g]
    summary = next((g["summary"] for g in load(gen_path) if "summary" in g), None)
    model_name = pick_model_name()
    per, fails, details, lenient = {}, 0, [], 0
    for g in gens:
        r = rows[g["i"]]
        text = clean(g.get("gen") or "")
        ok, err = False, ""
        try:
            decoded = mu.default_decode_ast_prompting(text, RET[r["language"]])
            res = ast_checker(r["function"], decoded, r["possible_answer"], LANG[r["language"]], r["v4_category"], model_name)
            ok = bool(res.get("valid"))
            err = "" if ok else str(res.get("error_type") or res.get("error") or "")[:80]
        except Exception as e:
            fails += 1
            err = f"decode: {type(e).__name__}"
        ok_len = ok
        if not ok and "\\_" in text:  # Markdown-escaped underscores in names: counted wrong officially, right in the lenient score
            try:
                dec2 = mu.default_decode_ast_prompting(text.replace("\\_", "_"), RET[r["language"]])
                ok_len = bool(ast_checker(r["function"], dec2, r["possible_answer"], LANG[r["language"]], r["v4_category"], model_name).get("valid"))
            except Exception:
                ok_len = False
        lenient += ok_len
        d = per.setdefault(r["category"], [0, 0])
        d[0] += ok
        d[1] += 1
        details.append({"id": r["id"], "category": r["category"], "ok": ok, "err": err, "response": text[:300]})
    acc = {c: (v[0] / v[1]) for c, v in per.items()}
    simple_parts = [acc[c] for c in ("simple", "java", "javascript") if c in acc]
    simple = sum(simple_parts) / len(simple_parts) if simple_parts else None
    four = [x for x in (simple, acc.get("multiple"), acc.get("parallel"), acc.get("parallel_multiple")) if x is not None]
    overall = sum(four) / len(four) if four else 0.0
    micro = sum(v[0] for v in per.values()) / max(1, sum(v[1] for v in per.values()))
    out = {"summary": summary, "n": len(gens), "overall": overall, "micro": micro, "simple": simple, "by_category": {c: f"{v[0]}/{v[1]}" for c, v in per.items()},
           "decode_fail": fails, "model_name_for_checker": model_name, "details": details}
    json.dump(out, open(out_path, "w"))
    loops = f" loops={summary['mean_loops']:.3f}" if summary and "mean_loops" in summary else ""
    print(f"GRADED n={len(gens)} overall={overall:.4f} micro={micro:.4f} simple={simple if simple is None else round(simple, 4)} "
          + " ".join(f"{c}={acc[c]:.4f}" for c in ("multiple", "parallel", "parallel_multiple") if c in acc) + f" decode_fail={fails} lenient_micro={lenient / max(1, len(gens)):.4f}{loops}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        grade(sys.argv[2], sys.argv[3], sys.argv[4])
