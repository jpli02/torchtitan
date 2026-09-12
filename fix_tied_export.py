#!/usr/bin/env python3
"""Make a torchtitan HF export of a weight-tied Qwen3 loadable by transformers.

Qwen3-1.7B has tie_word_embeddings=true and torchtitan's Qwen3 1.7B config
enable_weight_tying=True, so the exporter has no separate output.weight and
writes lm_head.weight as an EMPTY tensor (dtype '', shape []) in its own
78-byte shard, which safetensors refuses to parse. Drop that entry from the
index and delete the shard; transformers re-ties lm_head to embed_tokens.

  python fix_tied_export.py <serving_dir>
"""
import json
import os
import struct
import sys

d = sys.argv[1]
idx_path = os.path.join(d, "model.safetensors.index.json")
idx = json.load(open(idx_path))
wm = idx["weight_map"]
removed = []
for key in list(wm):
    shard = os.path.join(d, wm[key])
    if not os.path.exists(shard):
        continue
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    ent = hdr.get(key)
    if ent is None or not ent.get("dtype") or not ent.get("shape"):
        removed.append((key, wm[key]))
        del wm[key]
for key, shard in removed:
    # delete the shard only if nothing else still maps to it
    if shard not in wm.values():
        os.remove(os.path.join(d, shard))
json.dump(idx, open(idx_path, "w"), indent=2)
print(f"removed empty tensors: {removed or 'none'}; {len(wm)} tensors remain")
