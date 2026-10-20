#!/bin/bash
# Make each valid variant task runnable by `tb run`: add the generic TB 0.2.18
# scaffold (docker-compose.yaml, run-tests.sh, tests/{setup.sh,run.sh,
# test_requirements.txt}) and fix task.yaml test_scripts. Idempotent. Copies
# runnable tasks into POOL (so gen_traj can point at one dataset dir).
set -u
SRC=${SRC:-/home/jli199/boptim_scratch/tb_variants/valid}
POOL=${POOL:-/home/jli199/boptim_scratch/tb_variants/pool}
mkdir -p "$POOL"
n=0
for v in "$SRC"/*/; do
  [ -d "$v" ] || continue
  name=$(basename "$v")
  dst="$POOL/$name"
  rm -rf "$dst"; cp -r "$v" "$dst"
  # terminus-2 needs tmux in the container; LLM Dockerfiles omit it. Inject it.
  if ! grep -qi "install.*tmux" "$dst/Dockerfile" 2>/dev/null; then
    printf '\n# injected: terminus-2 requires tmux\nRUN (apt-get update && apt-get install -y --no-install-recommends tmux ca-certificates && rm -rf /var/lib/apt/lists/*) || (apk add --no-cache tmux) || true\n' >> "$dst/Dockerfile"
  fi
  # docker-compose.yaml
  cat > "$dst/docker-compose.yaml" <<'YML'
services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
YML
  cat > "$dst/run-tests.sh" <<'SH'
#!/bin/bash
source $TEST_DIR/setup.sh
bash $TEST_DIR/run.sh
SH
  mkdir -p "$dst/tests"
  cat > "$dst/tests/setup.sh" <<'SH'
#!/bin/bash
PY=$(command -v python3 || command -v python)
"$PY" -m pip install --break-system-packages -q pytest packaging 2>/dev/null || "$PY" -m pip install -q pytest packaging || true
SH
  cat > "$dst/tests/run.sh" <<'SH'
#!/bin/bash
PY=$(command -v python3 || command -v python)
"$PY" -m pytest "$TEST_DIR/test_outputs.py" -rA
SH
  printf 'pytest\npackaging\n' > "$dst/tests/test_requirements.txt"
  # fix task.yaml test_scripts -> setup.sh/run.sh
  python3 - "$dst/task.yaml" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"test_scripts:.*?(?=\n\w|\Z)", "test_scripts:\n  - setup.sh\n  - run.sh\n", s, flags=re.DOTALL, count=1)
if "test_scripts:" not in s: s += "\ntest_scripts:\n  - setup.sh\n  - run.sh\n"
open(p, "w").write(s)
PY
  n=$((n+1))
done
echo "scaffolded $n tasks -> $POOL"
