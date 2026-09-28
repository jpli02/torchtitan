"""Convert Nemotron-Terminal-Synthetic-Tasks (Harbor task.toml format) into
terminal-bench 0.2.18 task dirs so `tb run` can execute and grade them.

Why: to GENERATE verified trajectories in the missing categories we need
runnable tasks that are (a) not our 80, (b) verifiable, (c) in 0.2.18 format.
terminal-bench-core==0.1.1 is exactly our 80; TB-2 tasks are encrypted. The
Nemotron synthetic tasks are unencrypted, CC-BY-4.0, ~90k tasks over 9 skill
categories with pytest test suites and public base images -- the only source
that fits.

Harbor layout                          0.2.18 layout this writes
  task.toml                            task.yaml (instruction, category,
  instruction.md                         difficulty, parser_name: pytest,
  environment/Dockerfile                 timeouts, test_scripts)
  environment/files/*                  Dockerfile (COPY files/ /app/ works
  tests/test_outputs.py                  because we stage files/ at task root)
  tests/test_requirements.txt          files/*            (build context)
  tests/test.sh                        docker-compose.yaml (generic tb template)
                                       tests/test_outputs.py, test_requirements.txt
                                       tests/setup.sh, run.sh  (system-pip pytest
                                         -rA so tb's pytest parser reads it and
                                         test deps match the Dockerfile's python)
                                       run-tests.sh
                                       solution/  (empty; rejection sampling
                                         supplies the trajectory, not an oracle)

Usage: python convert_harbor.py --category security --n 40 --out DIR
"""
import argparse
import io
import os
import re
import tarfile

from huggingface_hub import hf_hub_download

REPO = "nvidia/Nemotron-Terminal-Synthetic-Tasks"
# skill-category -> (tar path inside repo, difficulty label)
TARS = {
    "security": "skill_based/mixed/security.tar.gz",
    "debugging": "skill_based/mixed/debugging.tar.gz",
    "data_science": "skill_based/mixed/data_science.tar.gz",
    "data_processing": "skill_based/mixed/data_processing.tar.gz",
    "file_operations": "skill_based/mixed/file_operations.tar.gz",
    "scientific_computing": "skill_based/mixed/scientific_computing.tar.gz",
}
# Nemotron skill category -> TB-80 category name (for the manifest / reweighting)
TO_TB80 = {
    "security": "security", "debugging": "debugging",
    "data_science": "data-science", "data_processing": "data-science",
    "file_operations": "file-operations",
    "scientific_computing": "scientific-computing",
}

COMPOSE = """services:
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
"""
# System-python pytest with -rA so tb's pytest parser reads PASSED/FAILED, and
# so tests import the same packages the Nemotron Dockerfile installed.
# Nemotron's ubuntu-24.04 base provides python3/pip3, not python/pip, so resolve
# the interpreter and drive pip through `python3 -m pip`. -rA guarantees the
# "short test summary info" section tb's pytest parser requires.
SETUP = """#!/bin/bash
PY=$(command -v python3 || command -v python)
"$PY" -m pip install --break-system-packages -q pytest 2>/dev/null || "$PY" -m pip install -q pytest || true
if [ -f "$TEST_DIR/test_requirements.txt" ]; then
  "$PY" -m pip install --break-system-packages -q -r "$TEST_DIR/test_requirements.txt" 2>/dev/null || \
    "$PY" -m pip install -q -r "$TEST_DIR/test_requirements.txt" || true
fi
"""
RUN = """#!/bin/bash
PY=$(command -v python3 || command -v python)
"$PY" -m pytest "$TEST_DIR/test_outputs.py" -rA
"""
RUN_TESTS = """#!/bin/bash
source $TEST_DIR/setup.sh
bash $TEST_DIR/run.sh
"""


def toml_get(txt, key, default=""):
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9.]+|\"[^\"]*\")", txt, re.M)
    if not m:
        return default
    v = m.group(1).strip().strip('"')
    return v


def yaml_escape_block(text):
    return "\n".join("  " + line for line in text.splitlines())


def convert_one(t, pre, cat, difficulty, out_root):
    files = [m for m in t.getmembers() if m.name.startswith(pre) and m.isfile()]
    rel = {m.name[len(pre):]: m for m in files}
    if "instruction.md" not in rel or "tests/test_outputs.py" not in rel or \
       "environment/Dockerfile" not in rel:
        return None
    name = pre.strip("/").split("/")[-1]
    d = os.path.join(out_root, name)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    os.makedirs(os.path.join(d, "files"), exist_ok=True)
    os.makedirs(os.path.join(d, "solution"), exist_ok=True)

    def read(k):
        return t.extractfile(rel[k]).read().decode(errors="ignore")

    instruction = read("instruction.md").strip()
    if "terminal-bench-canary" in instruction:  # never train the canary in
        instruction = "\n".join(l for l in instruction.splitlines()
                                if "terminal-bench-canary" not in l
                                and "BENCHMARK DATA" not in l).strip()
    toml = read("task.toml")
    atimeout = toml_get(toml, "timeout_sec", "600")
    try:
        atimeout = float(re.search(r"\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", toml, re.S).group(1))
    except Exception:
        atimeout = 600.0

    # Dockerfile: COPY files/ /app/ already refers to a build-context 'files/'.
    open(os.path.join(d, "Dockerfile"), "w").write(read("environment/Dockerfile"))
    # stage environment/files/* -> files/
    for k, m in rel.items():
        if k.startswith("environment/files/"):
            sub = k[len("environment/files/"):]
            dst = os.path.join(d, "files", sub)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(t.extractfile(m).read())
    open(os.path.join(d, "docker-compose.yaml"), "w").write(COMPOSE)
    open(os.path.join(d, "tests", "test_outputs.py"), "w").write(read("tests/test_outputs.py"))
    if "tests/test_requirements.txt" in rel:
        open(os.path.join(d, "tests", "test_requirements.txt"), "w").write(read("tests/test_requirements.txt"))
    open(os.path.join(d, "tests", "setup.sh"), "w").write(SETUP)
    open(os.path.join(d, "tests", "run.sh"), "w").write(RUN)
    open(os.path.join(d, "run-tests.sh"), "w").write(RUN_TESTS)

    tb80 = TO_TB80.get(cat, cat)
    task_yaml = (
        "instruction: |-\n" + yaml_escape_block(instruction) + "\n"
        f"author_email: nemotron-synthetic\n"
        f"difficulty: {difficulty}\n"
        f"category: {tb80}\n"
        "tags:\n  - " + tb80 + "\n  - nemotron-synthetic\n"
        "parser_name: pytest\n"
        f"max_agent_timeout_sec: {max(atimeout, 420.0)}\n"
        "max_test_timeout_sec: 300.0\n"
        "test_scripts:\n  - setup.sh\n  - run.sh\n"
        "run_tests_in_same_shell: false\n"
    )
    open(os.path.join(d, "task.yaml"), "w").write(task_yaml)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(TARS))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip", type=int, default=0, help="skip first K tasks (for disjoint batches)")
    ap.add_argument("--out", default="/home/jli199/boptim_scratch/gen_tasks")
    args = ap.parse_args()
    p = hf_hub_download(REPO, TARS[args.category], repo_type="dataset")
    difficulty = "hard" if args.category in ("security", "scientific_computing", "debugging") else "medium"
    out_root = os.path.join(args.out, args.category)
    os.makedirs(out_root, exist_ok=True)
    t = tarfile.open(p)
    tasks = sorted(set(m.name.split("/")[2] for m in t.getmembers()
                       if len(m.name.split("/")) > 3 and re.search(r"_task_\d+", m.name.split("/")[2])))
    picked = tasks[args.skip:args.skip + args.n]
    done = []
    for name in picked:
        pre = f"./{args.category}/{name}/"
        try:
            r = convert_one(t, pre, args.category, difficulty, out_root)
            if r:
                done.append(r)
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:80]}")
    print(f"converted {len(done)}/{len(picked)} {args.category} tasks -> {out_root}")
    for n in done[:5]:
        print("  ", n)


if __name__ == "__main__":
    main()
