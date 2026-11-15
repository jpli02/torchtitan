"""Where does real early exit lose accuracy: in WHICH tokens leave, or in the CACHE left behind by the tokens that left?

Teacher-forced incremental decoding with the real decode-time early exit: the model is fed its own full-depth answer token by token
(so one wrong prediction cannot derail the rest), exits under a router exactly as in generation (an exited token's keys and values
of the skipped loops are filled from those of its exit loop), and at every step its prediction is compared with the full-depth
token. With the exact-cache predictions of bfcl_signal_dump.py for the same tokens, every wrong prediction is then one of:
  * selection: the head was already wrong after that loop even with an exact cache (the token should not have left there);
  * cache: the head is right after that loop with an exact cache, and wrong here, where earlier tokens left early.

A router is "q1,q2,q3" (gate thresholds), optionally followed by "|conf=t1,t2,t3" (head-confidence exits) and "|maps=path"
(calibrated cache fill); the last two need the exit-lab model directory (make_exitlab_modeling.py).

  CUDA_VISIBLE_DEVICES=0 [CKPT=dir] python boptim_lcb/bfcl_tf_exit.py [step] [routers separated by ';'] [out.jsonl]
"""
import glob
import json
import os
import sys
import time

import torch

DATA = "/srv/scratch/jli199/bfcl_stage/bfcl_search.jsonl"
WORK = "/home/jli199/boptim_scratch/bo_bfcl_router"
DEFAULT_ROUTERS = "0.2,0.2,0.2;0.25,0.25,0.25;0.3,0.3,0.3"


def parse_router(spec):
    parts = spec.split("|")
    r = {"gate": [float(v) for v in parts[0].split(",")], "conf": None, "maps": None, "spec": spec}
    for p in parts[1:]:
        k, v = p.split("=", 1)
        r[k] = [float(x) for x in v.split(",")] if k == "conf" else v
    return r


def load_items():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    exact = [json.loads(l) for l in open(f"{WORK}/signal_probs.jsonl")]      # exact-cache predictions per loop, same items in order
    assert len(rows) == len(exact) and all(r["id"] == e["id"] for r, e in zip(rows, exact))
    return rows, exact


def run(model, tok, routers, step=4, out_path=f"{WORK}/tf_exit.jsonl", log=print):
    rows, exact = load_items()
    pick = list(range(0, len(rows), step))
    inner = model.model
    results = []
    with open(out_path, "a") as out, torch.inference_mode():
        for router in routers:
            q = router["spec"]
            inner.decode_early_exit_threshold = router["gate"] + [1.0]
            inner.decode_exit_conf = router["conf"]           # ignored by the deployed model file
            inner.decode_fill_maps = router["maps"]
            t0 = time.time()
            n_tok = n_wrong = n_sel = n_cache = n_fixed = bad_items = loops_sum = 0
            by_loop = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0]}  # exit loop -> [tokens, wrong, wrong by cache]
            by_pos = {}                                                           # answer position bucket -> [tokens, wrong]
            for k, i in enumerate(pick):
                r, e = rows[i], exact[i]
                text = tok.apply_chat_template([{"role": "user", "content": r["prompt"].rstrip()}], tokenize=False, add_generation_prompt=True) + "["
                p_ids = tok(text)["input_ids"]
                target = e["target"]
                o = model(input_ids=torch.tensor([p_ids], device="cuda"), use_cache=True, logits_to_keep=1)   # prefill at full depth
                past = o.past_key_values
                preds, loops = [int(o.logits[0, -1].argmax())], [4]
                for t in range(len(target) - 1):
                    s0 = getattr(inner, "_dee_exec_sum", 0)
                    o = model(input_ids=torch.tensor([[target[t]]], device="cuda"), past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    preds.append(int(o.logits[0, -1].argmax()))
                    loops.append(int(getattr(inner, "_dee_exec_sum", 0) - s0) or 4)
                del past, o
                item_bad = False
                for t, (p, L) in enumerate(zip(preds, loops)):
                    wrong = p != target[t]
                    sel_wrong = e["pred"][L - 1][t] != target[t]                           # exact cache, same loop
                    n_tok += 1
                    loops_sum += L
                    n_wrong += wrong
                    n_sel += wrong and sel_wrong
                    n_cache += wrong and not sel_wrong
                    n_fixed += (not wrong) and sel_wrong
                    by_loop[L][0] += 1
                    by_loop[L][1] += wrong
                    by_loop[L][2] += wrong and not sel_wrong
                    b = min(t // 10, 6)
                    by_pos.setdefault(b, [0, 0])
                    by_pos[b][0] += 1
                    by_pos[b][1] += wrong
                    item_bad = item_bad or wrong
                bad_items += item_bad
                if (k + 1) % 25 == 0:
                    log(f"  router {q}: {k + 1}/{len(pick)} items, {n_tok} tokens, wrong {100 * n_wrong / n_tok:.2f} % (cache {100 * n_cache / n_tok:.2f} %), "
                        f"loops {loops_sum / n_tok:.3f}, {time.time() - t0:.0f}s")
            rec = {"router": q, "items": len(pick), "tokens": n_tok, "loops": loops_sum / n_tok, "wrong_pct": 100 * n_wrong / n_tok,
                   "wrong_selection_pct": 100 * n_sel / n_tok, "wrong_cache_pct": 100 * n_cache / n_tok, "right_despite_selection_pct": 100 * n_fixed / n_tok,
                   "items_with_wrong_pct": 100 * bad_items / len(pick),
                   "by_exit_loop": {str(L): {"tokens": v[0], "wrong_pct": 100 * v[1] / max(1, v[0]), "wrong_cache_pct": 100 * v[2] / max(1, v[0])} for L, v in by_loop.items()},
                   "by_position": {f"{10 * b}-{10 * b + 9}" if b < 6 else "60+": {"tokens": v[0], "wrong_pct": 100 * v[1] / max(1, v[0])} for b, v in sorted(by_pos.items())}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log("RESULT " + json.dumps(rec))
            results.append(rec)
    inner.decode_exit_conf = None
    inner.decode_fill_maps = None
    return results


def load_model(ckpt):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    config.total_ut_steps = 4
    model = AutoModelForCausalLM.from_pretrained(ckpt, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    return model, tok


if __name__ == "__main__":
    ckpt = os.environ.get("CKPT", "/home/jli199/boptim_scratch/ouro_base_kvshare")
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    routers = [parse_router(r) for r in (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROUTERS).split(";")]
    out_path = sys.argv[3] if len(sys.argv) > 3 else f"{WORK}/tf_exit.jsonl"
    model, tok = load_model(ckpt)
    print(f"routers {[r['spec'] for r in routers]}, model {ckpt}", flush=True)
    run(model, tok, routers, step, out_path, log=lambda s: print(s, flush=True))
    print("done")
