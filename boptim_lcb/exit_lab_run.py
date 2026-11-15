"""One process for the exit-lab measurements on the shared BFCL card (one model load, memory held from start to end):
  1. fit the calibrated-fill maps (kv_fill_fit.fit) and report copy / affine errors by layer;
  2. teacher-forced exit tests (bfcl_tf_exit.run) for gate routers with plain copies, the same with the calibrated fill, and
     head-confidence routers with and without it.
The card also runs the held-out validation, which sizes each evaluation by the free memory it sees when it starts. This process
therefore waits for the boundary between two evaluations (no validation shard alive), reserves its memory at once and only then
loads the model, so the next evaluation starts with one shard fewer instead of colliding with it.

  setsid nohup python -u boptim_lcb/exit_lab_run.py > $TMP/exit_lab_run.log 2>&1 < /dev/null &        (ROUTERS env overrides the list)
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_tf_exit import load_model, parse_router, run  # noqa: E402
from kv_fill_fit import fit  # noqa: E402

WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
MAPS = f"{WORK}/fill_maps.pt"
LAB = "/home/jli199/boptim_scratch/ouro_base_exitlab"
RESERVE_GB = float(os.environ.get("RESERVE_GB", "9.5"))


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def shards_alive():
    out = subprocess.run(["pgrep", "-u", os.environ.get("USER", ""), "-f", "bfcl_stage/bfcl_"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    return len(out.split())


torch.zeros(1, device="cuda")                       # CUDA context up front, so the reservation below takes milliseconds
if os.environ.get("WAIT", "1") == "1":
    log(f"waiting for an evaluation boundary ({shards_alive()} validation shards alive)")
    while True:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if shards_alive() == 0 or free >= RESERVE_GB + 9.5:
            try:
                hold = torch.empty(int(RESERVE_GB * 2 ** 30), dtype=torch.uint8, device="cuda")
                break
            except torch.OutOfMemoryError:
                pass
        time.sleep(1.0)
    log(f"memory reserved ({RESERVE_GB} GB); loading the model")
    del hold                                         # the allocator keeps the block; the model is loaded into it
model, tok = load_model(LAB)
log("model loaded")
if not os.path.exists(MAPS) or os.environ.get("REFIT"):
    fit(model, tok, int(os.environ.get("N_FIT", "120")), int(os.environ.get("N_TEST", "40")), MAPS, log=log)
default = ";".join([
    "0.2,0.2,0.2", f"0.2,0.2,0.2|maps={MAPS}",                                                   # gate 0.2: plain copies, calibrated fill
    "0.999,0.999,0.999|conf=0.99,0.99,0.99", f"0.999,0.999,0.999|conf=0.99,0.99,0.99|maps={MAPS}",  # confidence alone
    "0.25,0.25,0.25", f"0.25,0.25,0.25|maps={MAPS}",
    "0.3,0.3,0.3",
    "0.999,0.999,0.999|conf=1.0,0.95,0.95",                                                        # confidence from loop 2 on, no loop-1 exits
])
routers = [parse_router(r) for r in os.environ.get("ROUTERS", default).split(";")]
log(f"teacher-forced exit tests: {[r['spec'].replace(MAPS, 'MAPS') for r in routers]}")
run(model, tok, routers, step=int(os.environ.get("STEP", "4")), out_path=f"{WORK}/tf_exit_lab.jsonl", log=log)
log("exit lab run finished")
