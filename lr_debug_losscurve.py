"""Learning-rate debug, part 2: loss and gradient norm of the terminal-SFT runs against the learning rate that was applied.

Reads the trainer's console logs (step / loss / grad_norm every 10 steps), bins them, and prints the mean loss, mean and max
gradient norm, and the scheduled LR per bin. The LR is recomputed from the recipe (verified in lr_debug_schedule.py).

  python lr_debug_losscurve.py <train.log> [total_steps] [bin]
"""
import math
import re
import sys

LR0, WARM, DECAY_RATIO, FLOOR = 2e-5, 50, 0.9, 0.1
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW = re.compile(r"step:\s*(\d+)\s+loss:\s*([0-9.eE+-]+|nan|inf)\s+grad_norm:\s*([0-9.eE+-]+|nan|inf)")


def lam(step, total):
    decay = round(total * DECAY_RATIO)
    if WARM + decay > total:
        decay = total - WARM
    stable = total + 1 - WARM - decay
    if step < WARM:
        return (step + 1) / WARM
    if step < WARM + stable:
        return 1.0
    prog = (step + 1 - WARM - stable) / decay
    return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * prog))


def main():
    path = sys.argv[1]
    rows = []
    total = None
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = re.search(r"total steps (\d+) \(warmup (\d+)\)", line)
        if m:
            total = int(m.group(1))
        m = ROW.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    if len(sys.argv) > 2:
        total = int(sys.argv[2])
    total = total or (rows[-1][0] if rows else 0)
    width = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, total // 16)
    print(f"{path.split('/')[-1]}: {len(rows)} logged steps, total steps {total}, bin {width}")
    print(" steps          lr (mid)    mean loss   median   mean |g|   max |g|   share |g|>1")
    b = 0
    while b * width < total:
        lo, hi = b * width, (b + 1) * width
        sel = [r for r in rows if lo < r[0] <= hi]
        if sel:
            ls = sorted(r[1] for r in sel)
            gs = [r[2] for r in sel]
            mid = (lo + hi) // 2
            print(f"{lo + 1:5d}-{hi:5d}   {LR0 * lam(mid - 1, total):.2e}   {sum(ls) / len(ls):8.4f}   {ls[len(ls) // 2]:6.4f}   "
                  f"{sum(gs) / len(gs):7.3f}   {max(gs):7.3f}   {sum(1 for g in gs if g > 1.0) / len(gs):6.0%}")
        b += 1
    bad = [r for r in rows if not math.isfinite(r[1]) or not math.isfinite(r[2])]
    print("non-finite rows:", len(bad), "| largest loss:", max(r[1] for r in rows), "at step", max(rows, key=lambda r: r[1])[0])


if __name__ == "__main__":
    main()
