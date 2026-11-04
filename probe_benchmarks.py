"""Probe Hub availability and schema of candidate 2025 benchmarks (streaming, first row only)."""
import sys

from datasets import get_dataset_config_names, load_dataset

CANDS = [
    ("WildEval/ZebraLogic", None, "test"),
    ("allenai/ZebraLogicBench", None, "test"),
    ("CodeEval-Pro/humaneval-pro", None, "train"),
    ("CodeEval-Pro/mbpp-pro", None, "train"),
    ("livebench/coding", None, "test"),
    ("livebench/reasoning", None, "test"),
    ("livebench/math", None, "test"),
    ("livebench/data_analysis", None, "test"),
    ("CRUXEval-X/cruxeval-x", None, "test"),
    ("Fsoft-AIC/CodeMMLU", None, "test"),
    ("evalplus/mbppplus", None, "test"),
]
for name, cfg, split in CANDS:
    try:
        cfgs = get_dataset_config_names(name)
    except Exception as e:  # noqa
        print(f"{name}: NOT REACHABLE ({str(e)[:90]})", flush=True)
        continue
    print(f"{name}: configs={cfgs[:8]}", flush=True)
    try:
        c = cfg or (cfgs[0] if cfgs else None)
        ds = load_dataset(name, c, split=split, streaming=True)
        row = next(iter(ds))
        print("   keys:", list(row.keys())[:14])
        for k, v in list(row.items())[:6]:
            print(f"   {k}: {str(v)[:120]!r}")
    except Exception as e:  # noqa
        print(f"   load failed: {str(e)[:120]}", flush=True)
