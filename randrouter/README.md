# randrouter: why router searches cannot beat the released gate, and what was tried against it

Started 2026-11-20 from the user's question: *"can we change the router into random and retrain stage-2 to see overcome the
bo cannot beat baseline issue"*. Model: Ouro-2.6B-Thinking (released base; every training here starts from it). Benchmark:
LiveCodeBench code generation, 342 test problems (releases v5 + v6). Nothing here is reinforcement learning: router choices are
random draws or fixed rules, never optimised against a reward.

Python: `/home/jli199/boptim-agent/.venv/bin/python`. Outputs: `/home/jli199/boptim_scratch/randrouter/`.

## Findings (2026-11-20)

1. **The baseline is protected by the cache, not by the gate.** A token that leaves after loop 1 is read correctly (the half
   of the tokens the gate is surest about: 100 % right), but the decoder fills its cache slots of loops 2-4 with copies of its
   loop-1 keys and values, and those are no better than a constant as stand-ins (relative error 1.6 for keys, 1.0 for values;
   loop 2 for loop 3: 0.18). Teacher-forced on the model's own full-depth answers, a router that releases the surest half after
   loop 1 keeps 99.3 % token agreement with an exact cache and 94.6 % with the decoder's cache. Real execution agrees: router
   (0.245, 0.3, 0.3) scores 19.0 % pass@1 with an exact cache, while with the copy fill the milder neighbour (0.25, 0.3, 0.3)
   already drops to 15.2 % (gate 0.3: 18.1 %, full depth 20.5 %).
2. **Random-router training repairs most of the token-level damage but not enough for pass@1.** LoRA (rank 32) trained under
   random routers with the decoder's fill, distilled from the base model's exact-cache readouts (`train.py --target kl`):
   96.2 -> 98.1 % agreement at 1.44 loops after 125 steps, full depth 99.76 -> 99.2 %. Real: 15.2 % at 1.66 executed loops
   (the released model needs 2.02 loops for the same 15.2 %). Hard labels (`--target ce`) move the model (full depth 98.0 %).
   A training-free calibrated fill (`fit_fill_maps.py`) gets as far (98.1 %); training the fill maps end to end
   (`--train_maps --r 0`) adds nothing; LoRA on top of the calibrated fill adds 0.1-0.2.
3. **Refitting the gate (Stage II) alone changes nothing** (`stage2.py`): a gate fitted from zero ranks tokens like the released
   one. A gate target the released one never had, the cost of an exit to the LATER tokens (`harm.py`, `harm_fit.py`), is
   predictable (AUC 0.75) and concentrated (the worst 10 % of tokens carry 60 % of it); used as a loop-1 veto it buys about
   0.3 point of agreement at equal loops.
4. **pass@1 needs about 98.5 % agreement to match gate 0.3 and 99.3 % for full depth**, and every repair stops near 98 %.
5. **With an exact cache a three-threshold search halves the loops on teacher-forced answers**: 1.3 loops at the gate-0.3
   agreement, 1.5-1.7 loops at 99.0-99.3 % (the released model with the copy fill: 2.1 and 3 loops). The router it finds,
   "the surest tokens leave after loop 1, all others run to full depth", is one the single knob cannot express. **It does not
   carry over to free-running decoding as is**: the two most aggressive of these routers, (0.05, 1, 1) and (0.132, 1, 1),
   which release 75-84 % of the tokens after loop 1, give long repetitive answers in real execution (45 % and 27 % cut at
   1,024 tokens after half of the problems, against 11 % at (0.245, 0.3, 0.3)) and execute 2.2-2.3 loops, not 1.5-1.7.
   Teacher forcing hides the drift; restrict the proxy to routers that release about the half the gate is surest about.
6. **Next step (not built):** a decode step with 4 rows costs 1.27x a step with 1 row (`bench_batch_step.py`), so the deeper
   loops of an already emitted token can run in the same batched pass as the next token's first loop: exact cache, latency of
   the router's loops, no compute saving.

## Files

| file | what it does |
|---|---|
| `core.py` | parallel teacher-forced emulation of decode-time early exit (`routed_forward`: plan / thresholds / shares routers, exact-cache mode, calibrated fill, loop-1 veto), losses, LoRA helpers |
| `check_equiv.py` | checks the emulation against token-by-token decoding (exit loops 97.8-100 %, predictions 99.5-100 % identical) |
| `prep_lcb_v1v4.py`, `gen_teacher.py`, `gen_teacher_extra.sh` | training pool = LiveCodeBench v1-v4 (713 problems, none of the 342 test problems); the base model's full-depth answers with token ids |
| `tf_eval.py`, `collect_tables.py`, `gate_signal.py` | teacher-forced fidelity of routers (decoder's cache and exact cache); what the gate knows per loop |
| `train.py` | random-router training (LoRA and/or fill maps; routers iid / rank / gate / full; KL or CE target) |
| `fit_fill_maps.py` | calibrated cache fill by ridge regression (file format of `boptim_lcb/kv_fill_fit.py`) |
| `stage2.py` | gate refit with the model frozen (paper's rule and two other targets) |
| `harm.py`, `harm_fit.py`, `harm_select.py`, `make_exitlab_veto.py` | cost of an exit to the later tokens (integrated gradients under random release sets), probe, loop-1 veto, serving directory with the veto |
| `ouro_tf_router.py`, `setup_tf_objective.py`, `tf_bo_prep.py`, `tf_bo.sh` | boptim objective `ouro_tf_router` (teacher-forced proxy of the LiveCodeBench router objective, 15 s per evaluation) and its launcher |
| `export.py` | merges a LoRA update (and a gate) into a servable model directory |
| `lcb_real_eval2.py`, `paired.py` | real execution on the 342 problems across cards (`--gather` = exact cache), paired comparison with the released model's runs |
| `bench_batch_step.py` | cost of a decode step with 1 / 2 / 4 rows |
| `patch_*.py` | one-off edits kept for the record |

## Pitfalls

- Cards 8 and 9 have about 9 GB free (another user's idle inference server): enough for teacher-forced work and single-stream
  decoding of short prompts with `--mem_gb 8.3-8.6`, not for training. Card 1 is left to the running router search.
- The optimizer's resume-state key ignores the objective's own settings: give every proxy search its own `resume_dir`
  (`tf_bo_prep.py` does).
- Token agreement is a weak predictor of pass@1 below 98.5 %: do not rank systems by the linear objective on the proxy alone.
