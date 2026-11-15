#!/bin/bash
# Removes the Terminal-Bench trial containers that killed harness runs left behind (state of 2026-11-14 19:20; none of them
# belongs to a running evaluation). Run by hand:  bash rm_leftover_containers.sh
# Why it matters: the harness starts a trial with `docker compose up` under the same project name, so a left-over container of
# the same run and output root is reused by the next trial of that task (see gpu1_lane3.sh). The first four are the ones the
# running epoch 8 evaluation would reuse; the two *_ep6_eval ones would be reused by the epoch 6 test evaluation.
docker rm -f \
  swe-bench-fsspec__r2v1-1-of-1-train75_ep8_1800 \
  swe-bench-astropy-1__r2v3-1-of-1-train75_ep8_1800 \
  fix-pandas-version__r2v1-1-of-1-train75_ep8_1800 \
  grid-pattern-transform__r2v1-1-of-1-train75_ep8_1800 \
  cron-broken-network-1-of-2-tvary_r1_8ep_ep6_eval \
  sanitize-git-repo-2-of-2-tvary_r1_8ep_ep6_eval \
  simple-web-scraper__r2v1-1-of-1-train75_ep6_1800 \
  blind-maze-explorer-5x5__v1-1-of-1-train75_ep6_1800 \
  cartpole-rl-training__r2v4-1-of-1-train25_ep7_1800 \
  fix-git__r2v3-1-of-1-train75_r1_1800_a \
  processing-pipeline__r2v5-1-of-1-train75_r1_1800_b \
  extract-moves-from-video__r2v8-1-of-1-train75_r1 \
  blind-maze-explorer-algorithm-easy__r2v2-1-of-1-train75_ep4 \
  grid-pattern-transform__v1-1-of-1-train75_v3 \
  blind-maze-explorer-5x5__v2-2-of-2-tvary_r2_sh1 \
  heterogeneous-dates__r2v6-2-of-2-tvary_r1_sh1 \
  sqlite-with-gcov__r2v5-2-of-2-tvary_r1_sh0
