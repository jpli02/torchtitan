#!/bin/bash
# Validate a synthesized TB task variant: the pytest suite must FAIL on the
# fresh container and PASS after the reference solution.sh runs. Prints
# VALID / INVALID(reason). Leaves no containers behind.
set -u
V="$1"
NAME=$(basename "$V" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9' '-' | sed 's/-\{1,\}/-/g; s/^-//; s/-$//')
IMG="tbvar-${NAME}"
TD="/tmp/tdir_${NAME}_$$"
run_tests() {  # $1=cid -> echoes "PASS" or "FAIL"
  docker exec "$1" bash -lc "rm -rf $TD && mkdir -p $TD && cd /app && TEST_DIR=$TD python -m pytest tests/test_outputs.py -q >/tmp/pt.out 2>&1; tail -1 /tmp/pt.out" 2>/dev/null \
    | grep -qE "passed" && echo PASS || echo FAIL
}
docker build -t "$IMG" "$V" >/tmp/build_${NAME}.log 2>&1 || { echo "INVALID(build_failed)"; tail -3 /tmp/build_${NAME}.log; exit 0; }
cid=$(docker run -d "$IMG" sleep infinity 2>/dev/null)
[ -n "$cid" ] || { echo "INVALID(run_failed)"; exit 0; }
docker exec "$cid" mkdir -p /app/tests >/dev/null 2>&1
docker cp "$V/tests/test_outputs.py" "$cid":/app/tests/test_outputs.py >/dev/null 2>&1
fresh=$(run_tests "$cid")
docker cp "$V/solution.sh" "$cid":/app/solution.sh >/dev/null 2>&1
docker exec "$cid" bash -lc "cd /app && bash solution.sh" >/tmp/sol_${NAME}.log 2>&1
solved=$(run_tests "$cid")
docker rm -f "$cid" >/dev/null 2>&1
docker rmi -f "$IMG" >/dev/null 2>&1
if [ "$fresh" = FAIL ] && [ "$solved" = PASS ]; then echo "VALID (fresh=FAIL solved=PASS)"
else echo "INVALID(fresh=$fresh solved=$solved)"; fi
