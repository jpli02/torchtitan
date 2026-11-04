"""Timeline of an ouro_openai_server log: per turn, the KV-cache reuse (cached / prefilled tokens) and the generation
(secs, tokens), in 50-turn blocks, to see whether reuse or runaway generations grow with uptime.
Usage: python server_timeline.py LOG [BLOCK=50]"""
import re
import statistics
import sys

path = sys.argv[1]
block = int(sys.argv[2]) if len(sys.argv) > 2 else 50
turns = []  # (cached, prefilled, secs, tokens)
cached = prefilled = None
for line in open(path, errors="replace"):
    m = re.search(r"KV-cache HIT: reusing (\d+) cached tokens, prefilling (\d+) new", line)
    if m:
        cached, prefilled = int(m.group(1)), int(m.group(2))
        continue
    if "KV-cache MISS" in line or "cache miss" in line.lower():
        cached, prefilled = 0, None
    m = re.search(r"secs=([0-9.]+) tok/s=([0-9.]+)", line)
    if m:
        s, r = float(m.group(1)), float(m.group(2))
        turns.append((cached, prefilled, s, s * r))
        cached = prefilled = None
print(f"{len(turns)} turns in {path.split('/')[-1]}")
print(f"{'turns':>9} {'cached mean/max':>16} {'prefill mean':>12} {'secs mean/max':>14} {'tok mean':>8} {'runaway(>2k)':>12}")
for b in range(0, len(turns), block):
    t = turns[b:b + block]
    c = [x[0] for x in t if x[0] is not None]
    p = [x[1] for x in t if x[1] is not None]
    s = [x[2] for x in t]
    k = [x[3] for x in t]
    print(f"{b:4d}-{b+len(t)-1:4d} {statistics.mean(c) if c else 0:8.0f}/{max(c) if c else 0:7d} {statistics.mean(p) if p else 0:12.0f} "
          f"{statistics.mean(s):7.1f}/{max(s):6.0f} {statistics.mean(k):8.0f} {sum(x > 2000 for x in k):12d}")
