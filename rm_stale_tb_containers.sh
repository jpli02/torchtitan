#!/bin/bash
# Remove leftover Terminal-Bench trial containers from my generation runs that were killed
# (tvary_r1_sh*/tvary_r2_sh* trials, up 10-12 days). Only containers named after my run ids.
for c in $(docker ps --format '{{.Names}}' | grep -E '(tvary_r[12]_sh|tcont_r1_sh|easy12_|train75_|oracle_real_)'); do
  docker rm -f "$c" >/dev/null 2>&1 && echo "removed $c"
done
echo "remaining containers: $(docker ps --format '{{.Names}}' | wc -l)"
