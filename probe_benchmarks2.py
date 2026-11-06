"""Probe Hub availability and schema of non-coding 2025 candidates (streaming, first row)."""
import collections

from datasets import get_dataset_config_names, load_dataset

CANDS = [
    ("livebench/math", "test"),
    ("livebench/data_analysis", "test"),
    ("livebench/language", "test"),
    ("livebench/instruction_following", "test"),
    ("KbsdJames/Omni-MATH", "test"),
    ("math-ai/aime25", "test"),
    ("HuggingFaceH4/aime_2024", "train"),
    ("MathArena/aime_2025", "train"),
    ("MathArena/hmmt_feb_2025", "train"),
    ("CohereLabs/Global-MMLU-Lite", "test"),
    ("madrylab/gsm8k-platinum", "test"),
    ("MMLU-CF/MMLU-CF", "test"),
    ("allenai/IFBench_test", "train"),
    ("google-deepmind/bbeh", "train"),
    ("TIGER-Lab/MMLU-Pro", "test"),
]
for name, split in CANDS:
    try:
        cfgs = get_dataset_config_names(name)
    except Exception as e:  # noqa
        print(f"{name}: NOT REACHABLE ({str(e)[:80]})", flush=True)
        continue
    print(f"{name}: configs={cfgs[:6]}", flush=True)
    try:
        c = cfgs[0] if cfgs else None
        ds = load_dataset(name, c, split=split, streaming=True)
        rows = []
        for i, r in enumerate(ds):
            rows.append(r)
            if i >= 400:
                break
        row = rows[0]
        print("   keys:", list(row.keys())[:14], "| rows seen:", len(rows))
        for k in ("task", "subtask", "difficulty", "domain", "category", "level"):
            if k in row:
                vals = collections.Counter(str(r.get(k))[:30] for r in rows)
                print(f"   {k} values: {dict(list(vals.items())[:8])}")
        for k, v in list(row.items())[:7]:
            print(f"   {k}: {str(v)[:140]!r}")
    except Exception as e:  # noqa
        print(f"   load failed: {str(e)[:120]}", flush=True)
