"""Token budget of an SFT data file as the trainer sees it: tokens per row with the training renderer, in file order.

  python -u count_sft_tokens.py DATA.jsonl OUT.json [NPROC] [STEPS]
The trainer reads rows in file order (no shuffle unless OURO_SFT_DATA_SEED is set) and packs them into 4096-token windows,
so `steps * 4096` tokens reach exactly the first rows whose cumulative token count fits. Writes per-row counts
{"tokens": [...], "supervised": [...]} and prints the total, the steps of one pass, and (with STEPS) the last row reached.
Runs in the torchtitan venv; rows are rendered in NPROC processes.
"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OURO_SFT_MIX", "tb80sft")
sys.path.insert(0, "/home/jli199/torchtitan/.claude/worktrees/ouro-terminal-sft")
SEQ = 4096
_tok = None


def _count(lines):
    global _tok
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import _terminal_agent_sft_tokens
    if _tok is None:
        _tok = HuggingFaceTokenizer(tokenizer_path="/home/jli199/torchtitan/assets/hf/Ouro-2.6B-Thinking")
    out = []
    for line in lines:
        ids, labels = _terminal_agent_sft_tokens(json.loads(line), _tok)
        out.append((len(ids), sum(1 for x in labels if x != -100)))
    return out


def main():
    data, out = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    lines = [line for line in open(data) if line.strip()]
    chunk = 20
    chunks = [lines[i:i + chunk] for i in range(0, len(lines), chunk)]
    with Pool(nproc) as pool:
        res = [r for part in pool.imap(_count, chunks) for r in part]
    toks, sup = [r[0] for r in res], [r[1] for r in res]
    json.dump({"data": data, "tokens": toks, "supervised": sup}, open(out, "w"))
    total = sum(toks)
    print(f"{os.path.basename(data)}: rows {len(toks):,}, tokens {total:,}, supervised {sum(sup):,} ({sum(sup) / total:.0%}); "
          f"one pass = {total / SEQ:,.0f} steps of {SEQ} tokens")
    if steps:
        seen, cum = steps * SEQ, 0
        last = len(toks)
        for i, n in enumerate(toks):
            cum += n
            if cum > seen:
                last = i
                break
        print(f"{steps:,} steps read {seen:,} tokens = {seen / total:.3f} of one pass: rows 0..{last - 1} completely, "
              f"{len(toks) - last:,} of {len(toks):,} rows at the end of the file not reached")


if __name__ == "__main__":
    main()
