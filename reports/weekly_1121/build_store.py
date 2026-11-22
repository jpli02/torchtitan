"""Store documents of the weekly progress page (Data-type artifact https://claude.ai/artifact/MazbdVeqUf5qAnBXHs43A8):
dash/meta, one datasets/<id> per uploaded CSV, files/index.html. Writes them as JSON files under reports/weekly_1121/store/ for
one ArtifactData batch.

  /home/jli199/boptim-agent/.venv/bin/python reports/weekly_1121/build_store.py
"""
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "store")
AT = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
BLOBS = {
    "tb_test": "c184b62de600cf671ba295e510ccb975", "tb_heldout": "3c5478ed6f32871457dd7d0b39e58c23",
    "tb_frombase_curve": "5662e68aef9362fdc4100a55e800c9f5", "lcb_runs": "e3803239b9a7bb80e41ca6592eb6e153",
    "lcb_searches": "a56de5b4fed6ef0904f1277506721f97", "lcb_llm_search": "de5a0bc607b38ae6850e57cc8ea415dc",
    "gate_nas": "550615dc2d800b415e29e782561b650b", "bfcl_heldout": "6b7d9c166afa9ae3b219f5fa360d3cc5",
    "bfcl_live": "760c667cb5dbafc4ad236a37fe98ab4c",
}
DS = {
    "tb_test": ("Terminal-Bench test score by checkpoint",
                "Trials solved on the 11-task Terminal-Bench test set, 2 trials per task, for every checkpoint of the learning-rate study; transcribed from the run summaries of Nov 15-19."),
    "tb_heldout": ("Held-out token loss by checkpoint",
                   "Token cross-entropy of each checkpoint on the 86-row validation file; transcribed from the per-checkpoint validation reports."),
    "tb_frombase_curve": ("From-base run, held-out loss during training",
                          "In-training validation of the from-base Terminal-Bench run every 1,000 steps, read from its validation log."),
    "lcb_runs": ("LiveCodeBench real runs of the week",
                 "pass@1 and executed loops on the 342 test problems, with problem-by-problem counts against the released gate at 0.3; transcribed from the run records and paired comparisons."),
    "lcb_searches": ("LiveCodeBench threshold searches, best points",
                     "Best router of the Gaussian-process and gpt-6-luna searches on all 342 problems, scored with measured speed; transcribed from the two searches' records."),
    "lcb_llm_search": ("gpt-6-luna search, every evaluation",
                       "The LLM agent's 10 evaluations, read from its evaluation log, with how many answers each shares with its best run."),
    "gate_nas": ("Gate NAS with BO, Nov 21 snapshot",
                 "Evaluations of the running Gaussian-process gate searches on BFCL and LiveCodeBench as of Nov 21, read from their logs, with the released gate's reference values."),
    "bfcl_heldout": ("BFCL held-out non-live items, every router",
                     "Accuracy and executed loops on 575 held-out non-live BFCL v3 items, with paired differences and 95 % bootstrap intervals; read from the held-out records."),
    "bfcl_live": ("BFCL live items",
                  "Accuracy and executed loops on the 1,350 live BFCL v3 items; read from the live-item records."),
}


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".json")
    with open(p, "w") as f:
        json.dump(obj, f)
    return p


paths = {"meta": dump("dash_meta", {"title": "Ouro Weekly Progress, Nov 16–21", "lang": "en"})}
for k, (title, desc) in DS.items():
    paths[k] = dump("ds_" + k, {"title": title, "description": desc, "fixed": True, "updated": {"at": AT, "by": "Claude"},
                                "source": {"kind": "file", "url": "/_blob/" + BLOBS[k], "name": k + ".csv", "format": "csv"}})
paths["page"] = dump("file_index", {"text": open(os.path.join(HERE, "index.html")).read()})
for k, p in paths.items():
    print(k, p, os.path.getsize(p))
