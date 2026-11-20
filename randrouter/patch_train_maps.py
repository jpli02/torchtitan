"""One-off edit (2026-11-20): train the cache FILL instead of (or with) the model.

The LoRA runs repair the cache but move the model a little even with the distillation target (full-depth agreement with the
base answers 99.76 -> 99.2 % after 125 steps), and that drift is paid at every router. The fill maps (one affine map per layer,
head and pair of loops, applied only to the keys/values written into the slots of skipped loops) are parameters the full-depth
path never touches: training THEM under the random router with the same distillation target cannot move the full-depth model
at all. So far they came from a ridge regression onto the true keys/values (fit_fill_maps.py), which treats every direction of
a key alike; trained end to end they are fitted to what the later tokens' readouts need.

train.py --train_maps: the maps of --fill_maps become trainable (learning rate --maps_lr); --r 0: no LoRA (maps only).
Checkpoints: fill_maps_step<N>.pt in the file format of fit_fill_maps.py (the exit-lab serving model reads it).
Kept for the record; the asserts refuse a second run."""
import pathlib

here = pathlib.Path(__file__).parent
p = here / "core.py"
s = p.read_text()
old = '''def _rot_half(x):
'''
new = '''def trainable_fill_maps(fill_maps):
    """The same maps as leaf parameters (float32); returns ((key maps, value maps), list of parameters)."""
    km = {k: nn.Parameter(v.detach().clone().float()) for k, v in fill_maps[0].items()}
    vm = {k: nn.Parameter(v.detach().clone().float()) for k, v in fill_maps[1].items()}
    return (km, vm), list(km.values()) + list(vm.values())


def save_fill_maps(path: str, fill_maps):
    """Write maps in the file format of fit_fill_maps.py / kv_fill_fit.py (read by the exit-lab serving model)."""
    km, vm = fill_maps
    blob = {"pairs": sorted(km.keys()), "layers": int(next(iter(km.values())).shape[0]),
            "maps": {**{f"0:{a}:{b}": m.detach().float().cpu() for (a, b), m in km.items()},
                     **{f"1:{a}:{b}": m.detach().float().cpu() for (a, b), m in vm.items()}}}
    torch.save(blob, path)


def _rot_half(x):
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)

p = here / "train.py"
s = p.read_text()
old = '''params = core.add_lora(model, args.r, args.alpha, targets=tuple(args.lora_targets.split(",")))
n_train = sum(p.numel() for p in params)
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
'''
new = '''if args.r > 0:
    params = core.add_lora(model, args.r, args.alpha, targets=tuple(args.lora_targets.split(",")))
else:                                                   # maps only: the model itself stays the released one
    for p_ in model.parameters():
        p_.requires_grad_(False)
    params = []
FILL = core.load_fill_maps(args.fill_maps) if args.fill_maps else None
map_params = []
if args.train_maps:
    assert FILL is not None, "--train_maps needs --fill_maps (the starting point)"
    FILL, map_params = core.trainable_fill_maps(FILL)
assert params or map_params, "nothing to train"
n_train = sum(p.numel() for p in params)
groups = ([{"params": params, "lr": args.lr}] if params else []) + ([{"params": map_params, "lr": args.maps_lr}] if map_params else [])
opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
BASE_LR = [g["lr"] for g in groups]
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''FILL = core.load_fill_maps(args.fill_maps) if args.fill_maps else None


def validate(step):
'''
new = '''def validate(step):
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    torch.save(core.lora_state(model), os.path.join(run_dir, "lora_last.pt"))
    torch.save(core.lora_state(model), os.path.join(run_dir, f"lora_step{step}.pt"))
'''
new = '''    if params:
        torch.save(core.lora_state(model), os.path.join(run_dir, "lora_last.pt"))
        torch.save(core.lora_state(model), os.path.join(run_dir, f"lora_step{step}.pt"))
    if map_params:
        core.save_fill_maps(os.path.join(run_dir, f"fill_maps_step{step}.pt"), FILL)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        for g in opt.param_groups:
            g["lr"] = lr_at(step)
'''
new = '''        for g, base in zip(opt.param_groups, BASE_LR):
            g["lr"] = lr_at(step) * base / args.lr
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
'''
new = '''        gnorm = float(torch.nn.utils.clip_grad_norm_(params + map_params, args.clip))
'''
assert s.count(old) == 1
s = s.replace(old, new)
old = 'ap.add_argument("--fill_maps", default=None, help="train (and validate) with the calibrated cache fill of fit_fill_maps.py instead of the plain copy")\n'
assert s.count(old) == 1
s = s.replace(old, old + 'ap.add_argument("--train_maps", action="store_true", help="the fill maps are trained too (their own learning rate --maps_lr); with --r 0 only they are")\n'
                         'ap.add_argument("--maps_lr", type=float, default=3e-4)\n')
old = '''print(f"trainable LoRA parameters {n_train / 1e6:.1f} M; train'''
assert s.count(old) == 1
s = s.replace(old, '''print(f"trainable LoRA parameters {n_train / 1e6:.1f} M, fill-map parameters {sum(p.numel() for p in map_params) / 1e6:.1f} M; train''')
p.write_text(s)
print("patched core.py and train.py")
