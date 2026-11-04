"""Per-turn generation statistics from ouro_openai_server logs: python server_turn_stats.py LOG [LOG ...]
Parses 'secs=<s> tok/s=<r>' lines -> tokens per turn (= secs * tok/s, reasoning included); prints count, mean,
median, p90, max seconds and tokens, and the share of turns above 500 / 1,000 / 2,000 generated tokens."""
import re
import statistics
import sys

for path in sys.argv[1:]:
    secs, toks = [], []
    for line in open(path, errors="replace"):
        m = re.search(r"secs=([0-9.]+) tok/s=([0-9.]+)", line)
        if m:
            s, r = float(m.group(1)), float(m.group(2))
            secs.append(s)
            toks.append(s * r)
    if not toks:
        print(f"{path}: no turns")
        continue
    toks_sorted = sorted(toks)
    p90 = toks_sorted[int(0.9 * (len(toks) - 1))]
    print(f"{path.split('/')[-1]}: {len(toks)} turns; tokens/turn mean {statistics.mean(toks):.0f} median {statistics.median(toks):.0f} "
          f"p90 {p90:.0f} max {max(toks):.0f}; secs/turn mean {statistics.mean(secs):.1f} max {max(secs):.0f}; "
          f"share >500 tok {sum(t > 500 for t in toks)/len(toks):.2f}, >1000 {sum(t > 1000 for t in toks)/len(toks):.2f}, >2000 {sum(t > 2000 for t in toks)/len(toks):.2f}")
