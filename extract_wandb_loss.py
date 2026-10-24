"""Extract the training-loss series from offline wandb run files.

Usage: python extract_wandb_loss.py OUT.json NAME=PATH/run-*.wandb [NAME=PATH ...]
Writes {NAME: {"steps": N, "pts": [[step, loss], ...], "key": loss_key}} with the
series subsampled to <= MAXPTS points (always keeping first and last).
"""
import json
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

MAXPTS = 70
LOSS_KEYS = ("loss_metrics/global_avg_loss", "loss", "train/loss", "loss_metrics/global_max_loss")


def read_history(path):
    ds = datastore.DataStore()
    ds.open_for_scan(path)
    rows = []
    keys = set()
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if not rec.HasField("history"):
            continue
        d = {}
        for it in rec.history.item:
            k = it.key or "/".join(it.nested_key)
            try:
                d[k] = json.loads(it.value_json)
            except Exception:
                d[k] = it.value_json
        keys.update(d)
        step = d.get("_step")
        if step is None and rec.history.HasField("step"):
            step = rec.history.step.num
        d["_step"] = step
        rows.append(d)
    return rows, keys


def main():
    out_path = sys.argv[1]
    out = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        rows, keys = read_history(path)
        key = next((k for k in LOSS_KEYS if k in keys), None)
        if key is None:
            print(f"{name}: no loss key among {sorted(keys)[:20]}", file=sys.stderr)
            continue
        pts = [[int(r["_step"]), round(float(r[key]), 4)] for r in rows
               if r.get("_step") is not None and isinstance(r.get(key), (int, float))]
        pts.sort()
        n = len(pts)
        if n > MAXPTS:
            stride = max(1, n // MAXPTS)
            sub = pts[::stride]
            if sub[-1] != pts[-1]:
                sub.append(pts[-1])
            pts = sub
        losses = [p[1] for p in pts]
        out[name] = {"steps": rows[-1]["_step"] if rows else 0, "key": key, "pts": pts}
        print(f"{name}: {n} logged points (every {pts[1][0]-pts[0][0] if len(pts)>1 else '?'} steps after subsample), "
              f"last step {out[name]['steps']}, loss start {losses[0]:.3f} min {min(losses):.3f} last {losses[-1]:.3f}, key={key}")
    json.dump(out, open(out_path, "w"), indent=0)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
