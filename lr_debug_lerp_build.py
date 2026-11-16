"""Build a servable checkpoint on the line between two checkpoints: theta = A + alpha * (B - A).

  python lr_debug_lerp_build.py A_DIR B_DIR ALPHA OUT_DIR
A_DIR / B_DIR are clean HF directories (model-00001-of-00001.safetensors + index + config + tokenizer); OUT_DIR gets the
same file layout with the blended weights (arithmetic in fp32, stored in the dtype of A) and A's companion files.
CPU only; about a minute.
"""
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

a_dir, b_dir, alpha, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
name = "model-00001-of-00001.safetensors"
a, b = load_file(os.path.join(a_dir, name)), load_file(os.path.join(b_dir, name))
if set(a) != set(b):
    sys.exit(f"key mismatch: {len(set(a) ^ set(b))} keys differ")
res, moved, base = {}, 0.0, 0.0
for k, va in a.items():
    vb = b[k]
    if va.is_floating_point() and va.shape == vb.shape:
        fa = va.float()
        d = vb.float() - fa
        res[k] = (fa + alpha * d).to(va.dtype)
        moved += float(d.pow(2).sum())
        base += float(fa.pow(2).sum())
    else:
        res[k] = va
os.makedirs(out, exist_ok=True)
save_file(res, os.path.join(out, name), metadata={"format": "pt"})
for f in os.listdir(a_dir):
    if f != name and os.path.isfile(os.path.join(a_dir, f)):
        shutil.copy2(os.path.join(a_dir, f), os.path.join(out, f))
print(f"wrote {out}: {len(res)} tensors, alpha {alpha}, |B-A|/|A| = {(moved / base) ** 0.5:.4f}, so this checkpoint is {alpha * (moved / base) ** 0.5:.4f} from A")
