#!/usr/bin/env python3
"""List Qwen models available on the configured QWEN_BASE_URL (OpenRouter),
with per-token prices, so a teacher can be picked. Never prints the key."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.environ.get("BOPTIM_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "boptim-agent")))
from gen_bo_traj import _load_keys  # noqa: E402

_load_keys()
base = os.environ["QWEN_BASE_URL"].rstrip("/")
req = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + os.environ["QWEN_API_KEY"]})
data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
rows = []
for m in data:
    mid = m.get("id", "")
    if not mid.startswith("qwen/"):
        continue
    p = m.get("pricing", {}) or {}
    rows.append((mid, float(p.get("prompt", 0) or 0) * 1e6, float(p.get("completion", 0) or 0) * 1e6, m.get("context_length")))
for mid, pi, po, ctx in sorted(rows):
    print(f"{mid:45s} in ${pi:6.2f}/M out ${po:6.2f}/M ctx {ctx}")
print(f"{len(rows)} qwen models")
