"""Learning-rate debug, part 1: what LR does the terminal-SFT recipe really apply?

Builds the optimizer and the scheduler exactly as the trainer does (torchtitan OptimizersContainer + LRSchedulersContainer
config of ouro_2_6b_thinking_terminal_sft: torchao AdamW8bit, lr 2e-5, warmup 50, decay_ratio 0.9, cosine, floor 0.1) on a small
dummy parameter, steps it for the run's 2144 steps, and compares three things per step:
  expected   lr0 * lambda(step) from the schedule formula,
  scheduler  what LambdaLR reports (get_last_lr),
  group      what the optimizer's param group holds (the value the update really uses).
It also measures the applied update size |delta w| / step for a constant unit gradient (Adam's normalised step is ~lr),
for AdamW8bit and for plain torch AdamW, so a mis-scaled effective LR in the 8-bit optimizer would show.

  /home/jli199/torchtitan/.venv/bin/python lr_debug_schedule.py [steps]
"""
import math
import sys

import torch

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2144
LR0, WARM, DECAY_RATIO, FLOOR = 2e-5, 50, 0.9, 0.1


def lam(step, total=STEPS):
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


def build(name):
    w = torch.nn.Parameter(torch.ones(4096, dtype=torch.float32))
    if name == "AdamW8bit":
        from torchao.optim import AdamW8bit
        opt = AdamW8bit([w], lr=LR0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW([w], lr=LR0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    return w, opt, sch


def main():
    print("torch", torch.__version__)
    try:
        import torchao
        print("torchao", torchao.__version__)
    except Exception as e:  # noqa: BLE001
        print("torchao import failed:", e)
    marks = {0, 1, 2, 5, 10, 25, 49, 50, 51, 100, 200, 264, 265, 300, 536, 1072, 1608, 2000, 2143} | {STEPS - 1}
    for name in ("AdamW8bit", "AdamW"):
        try:
            w, opt, sch = build(name)
        except Exception as e:  # noqa: BLE001
            print(name, "build failed:", repr(e)[:200])
            continue
        g0 = opt.param_groups[0]
        print(f"\n== {name}: lr type {type(g0['lr']).__name__}, initial_lr type {type(g0.get('initial_lr')).__name__}, "
              f"same object {g0['lr'] is g0.get('initial_lr')}")
        print(" step   expected    scheduler   group       |dw| this step   |dw| / expected")
        worst = 0.0
        for step in range(STEPS):
            lr_used = float(g0["lr"])                    # the value this optimizer step uses
            before = w.detach().clone()
            opt.zero_grad()
            w.grad = torch.ones_like(w)                  # constant unit gradient: Adam's normalised step is ~lr
            opt.step()
            dw = float((w.detach() - before).abs().mean())
            exp = LR0 * lam(step)
            worst = max(worst, abs(lr_used - exp) / exp)
            if step in marks:
                print(f"{step:5d}   {exp:.3e}   {sch.get_last_lr()[0]:.3e}   {lr_used:.3e}   {dw:.3e}        {dw / exp:.3f}")
            sch.step()
        print(f"largest relative gap between the LR used and the formula over {STEPS} steps: {worst:.2e}")
        print(f"sum of LR over the run: {sum(LR0 * lam(s) for s in range(STEPS)):.4e}  (mean {sum(LR0 * lam(s) for s in range(STEPS)) / STEPS:.3e})")


if __name__ == "__main__":
    main()
