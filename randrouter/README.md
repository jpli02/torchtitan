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
   carry over to free-running decoding**: the two most aggressive of these routers, which release 75-84 % of the tokens
   after loop 1, give long repetitive answers in real execution with an exact cache (all 342 problems):
   (0.132, 1, 1) 18.1 % pass@1 at 2.33 would-be loops, 31 % of the answers cut at 1,024 tokens (against the gate at 0.3:
   10 problems each way; against full depth: +2 / -10, p = 0.04); (0.05, 1, 1) 13.2 % at 2.22 loops, 49 % cut (against the
   gate at 0.3: +3 / -20, p < 0.001). Reference: (0.245, 0.3, 0.3) 19.0 % at 1.87 loops, 11 % cut. So no better than the
   released gate at 0.3 (18.1 % at about 2.1 loops), where teacher forcing promised 1.5-1.7 loops. Teacher forcing hides
   the drift; restrict the proxy to routers that release about the half the gate is surest about. The cache is exact in
   these runs, so the drift is not a cache effect: it is the early readouts themselves, compounding on self-generated text.
6. **Next step (not built):** a decode step with 4 rows costs 1.27x a step with 1 row (`bench_batch_step.py`), so the deeper
   loops of an already emitted token can run in the same batched pass as the next token's first loop: exact cache, latency of
   the router's loops, no compute saving.
7. **Token agreement is the wrong currency: the two kinds of token error do not cost the same** (`calib_pass1.py`, 33 routers
   that were run for real on the 342 problems, replayed on the proxy). own = % of answer tokens whose exit readout is wrong
   even with an exact cache (the token left too early); cache = % right with an exact cache and wrong with the decoder's
   cache. Fit: **pass@1 = 22.45 - 3.27 own - 1.17 cache** (weighted RMSE 0.67, leave-one-out 1.19 pass@1 points; one real
   measurement has a standard error of 2.1; rank correlation with real pass@1 0.915, leave-one-out 0.913). An early-exit
   error costs about 2.8 times a cache error: gate 0.2 has 94.4 % agreement and 15.0 % pass@1, fixed depth 1 has 95.3 % and
   5.6 %. Proxy loops are 0.07 below the executed loops. Consequences:
   - finding 4 ("98.5 % agreement needed") holds for early-exit errors, not for cache errors, so the repairs of finding 2
     were judged too harshly. Re-scored in the real objective's units (`proxy_value.py`: -6.25 * 4 / loops + 2.0 * max(0,
     18.3 - predicted pass@1)), the best router of the original space is the released gate at 0.3 (-10.93; 17.9 % at 2.14
     loops) and the calibrated fill reaches -13.77 at (0.2, 0.4, 0.7) (18.5 % at 1.82 loops) and -11.87 at (0.245, 0.45,
     0.7) (19.5 % at 2.11 loops). A PREDICTION: the price of a cache error rests on three routers. Bootstrap over the 33
     runs with the binomial noise of each pass@1 (`calib_boot.py`): alpha 3.24 (95 % interval 0.96 to 4.10), beta 1.23
     (0.00 to 3.19), alpha > beta in 93 % of the resamples; the calibrated fill comes out ahead of the released gate in
     83 % of the resamples at (0.2, 0.4, 0.7), 91 % at (0.245, 0.45, 0.7) and 97 % at (0.245, 0.3, 0.3) (17.2 % at 1.68
     loops, the most robust candidate). Real run started 2026-11-20 23:20 (`real/base_fill24_q0.2-0.4-0.7`).
   - limit of the fit: it over-predicts routers that release more than three quarters of the tokens after loop 1
     ((0.05, 1, 1) with an exact cache: predicted about 17.7 %, real 13.2 %; finding 5).
   - the proxy value with a penalty on raw agreement (`ouro_tf_router` without `ouro_tf_calib`) has its optimum at "every
     token leaves after loop 1" (-13.6), which real execution scores near +1. Searches on it that did not find that corner
     were lucky; use `ouro_tf_calib: 1`.
8. **Fast replica of the router NAS objective** (`ouro_tf_nas.py`, `nas_feats.py`, `nas_fit.py`): the 6-dimensional space of
   boptim-agent's `objective/ouro_router_nas.py`, every candidate router trained from a random initialisation with the
   Stage-II loss on cached states of the frozen model (66,474 positions), scored by the proxy in the real objective's units,
   about 30 s per evaluation. Reference check: no router trained from scratch beats the released gate (best of 14 random
   points -8.81, linear router at gamma 0.005 / threshold 0.5 -10.67, released gate -10.93). Built for multi-seed optimizer
   comparisons (`nas_compare.sh`, `opt_compare*.sh`, `opt_compare_report.py`); not run: the user chose to stay with the
   original setting (real LiveCodeBench searches) on 2026-11-20.

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
| `calib_pass1.py`, `proxy_value.py` | proxy against real execution: the two kinds of token error, the fitted pass@1 model, the proxy value in the real objective's units |
| `fill_search.sh` | threshold search on the proxy in real units, with the calibrated fill and with the copy fill |
| `nas_feats.py`, `nas_fit.py`, `ouro_tf_nas.py`, `setup_tfnas_objective.py`, `nas_probe.py` | fast replica of the router NAS objective (boptim objective `ouro_tf_nas`) and its reference check |
| `opt_compare.sh`, `opt_compare_llm.sh`, `nas_compare.sh`, `night_chain.sh`, `opt_compare_report.py`, `prompts/` | optimizer against optimizer on the proxy objectives over several seeds (random search, Gaussian process, LLM agent with and without the mechanism text); written, not run |
| `llm_check.sh`, `llm_ping.py`, `llm_credit.py` | key check before LLM-optimizer runs (remaining credit as numbers, one small call; never prints the key) |
| `patch_*.py` | one-off edits kept for the record |

## Pitfalls

- Cards 8 and 9 have about 9 GB free (another user's idle inference server): enough for teacher-forced work and single-stream
  decoding of short prompts with `--mem_gb 8.3-8.6`, not for training. Card 1 is left to the running router search.
- The optimizer's resume-state key ignores the objective's own settings: give every proxy search its own `resume_dir`
  (`tf_bo_prep.py` does).
- Token agreement is a weak predictor of pass@1 below 98.5 %: do not rank systems by the linear objective on the proxy alone.
