#!/usr/bin/env python3
"""Synthesize a Terminal-Bench task VARIANT from a seed task: same skill/tools
and difficulty, DIFFERENT scenario (renamed files, changed values/goal) so it is
clearly not identical but very similar. A strong LLM (gpt-6-luna, high effort)
emits the full task; we write it as a TB 0.2.18 task dir. The emitted tests must
FAIL on the fresh container and PASS after the emitted solution.sh (validated
separately). Canary strings are intentionally dropped (these are new tasks).

Usage: python gen_task_variant.py --seed <tb_task_dir> --out <variant_dir> [--model gpt-6-luna]
"""
import argparse, json, os, re, glob

def load_keys(path=os.path.expanduser("~/.boptim_keys.env")):
    if not os.path.exists(path): return
    for line in open(path):
        line = line.strip()
        if line.startswith("export "): line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def read(p):
    try: return open(p, encoding="utf-8", errors="ignore").read()
    except Exception: return ""

def strip_canary(s):
    out = []
    for ln in s.splitlines():
        if "canary" in ln.lower() or "BENCHMARK DATA SHOULD NEVER" in ln: continue
        out.append(ln)
    return "\n".join(out)

def seed_context(seed):
    ty = read(os.path.join(seed, "task.yaml"))
    instr = ""
    m = re.search(r"instruction:\s*\|-?\n(.*?)(?:\n\w+:|\Z)", ty, re.DOTALL)
    if m: instr = re.sub(r"^\s{2}", "", m.group(1), flags=re.M)
    cat = (re.search(r"^category:\s*(.+)$", ty, re.M) or [None, "general"])[1].strip() if re.search(r"^category:", ty, re.M) else "general"
    diff = (re.search(r"^difficulty:\s*(.+)$", ty, re.M) or [None, "medium"])[1].strip() if re.search(r"^difficulty:", ty, re.M) else "medium"
    docker = read(os.path.join(seed, "Dockerfile"))
    test = read(glob.glob(os.path.join(seed, "tests", "test_*.py"))[0]) if glob.glob(os.path.join(seed, "tests", "test_*.py")) else ""
    # app files COPY'd into the image (best-effort: non-scaffold files at task root)
    scaffold = {"task.yaml","Dockerfile","docker-compose.yaml","solution.sh","run-tests.sh"}
    appfiles = {}
    for f in glob.glob(os.path.join(seed, "*")):
        b = os.path.basename(f)
        if os.path.isfile(f) and b not in scaffold and not b.endswith(".sh"):
            appfiles[b] = read(f)[:4000]
    return strip_canary(instr).strip(), cat, diff, strip_canary(docker), strip_canary(test), appfiles

PROMPT = """You are creating a NEW command-line task for an agent benchmark, as a VARIANT of a seed task.
The variant must test the SAME skill, tools, and difficulty, but with a DIFFERENT concrete scenario:
rename files/dirs, change values, data, and the specific goal, so it is clearly NOT the same task yet very similar.
It must be fully self-contained and runnable in a Linux container.

SEED TASK
category: {cat} | difficulty: {diff}
--- instruction ---
{instr}
--- Dockerfile ---
{docker}
--- tests/test_outputs.py ---
{test}
--- app files ---
{appfiles}

Produce the VARIANT as a STRICT JSON object (no markdown, no prose) with keys:
- "instruction": string, the new task instruction given to the agent (describe what to build/fix; do NOT reveal the solution).
- "dockerfile": string, a complete Dockerfile that sets up the starting environment (base image, deps, COPY app files, WORKDIR /app). The starting state must make the tests FAIL (task unsolved).
- "app_files": object mapping relative path -> file contents for any starting files COPY'd by the Dockerfile.
- "test_file": string, a pytest file (tests/test_outputs.py) that verifies the task is solved; it must FAIL on the fresh container and PASS once the task is correctly done.
- "solution_sh": string, a bash script that, run in /app of the fresh container, correctly solves the task so the tests pass. This is the reference solution used only for validation.

CRITICAL VALIDITY RULES (a task that violates these is useless):
1. FAIL-FRESH IS MANDATORY. In the STARTING state (Dockerfile + app_files only, before solution_sh), the pytest suite MUST FAIL. The single most common fatal mistake is shipping a start state that already passes — do NOT do this.
2. DESIGN FOR FAIL-FRESH using one of these patterns:
   (a) OUTPUT-ARTIFACT: the task requires the agent to PRODUCE an output (a file at a specific path, a value, a built binary). The tests assert that artifact exists & is correct. At start the artifact does NOT exist, so tests fail. solution_sh creates it.
   (b) BROKEN-CODE: ship app code that is genuinely broken (throws / wrong output / missing dep). Tests assert correct behavior; they fail at start; solution_sh fixes the code/env.
   Before finalizing, mentally RUN your tests against the starting files — if any test could pass with no agent work, redesign.
3. Running solution_sh in /app of the fresh container MUST make all tests PASS. Tests must actually REQUIRE that work (assert on concrete artifacts/values/behavior), be deterministic, and not import the agent's solution.
4. THE IMAGE MUST BUILD. Use ONE common base (python:3.11-slim-bookworm or ubuntu:22.04), pin pip versions, keep deps minimal and available. Avoid heavyweight/exotic builds (no kernel/qemu compiles, no large model downloads).
5. Common tools only (python3/pip, bash, standard CLI); self-contained; no canary strings.
Output ONLY the JSON object."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="high")
    a = ap.parse_args()
    load_keys()
    import openai
    client = openai.OpenAI()
    instr, cat, diff, docker, test, appfiles = seed_context(a.seed)
    appf = "\n".join(f"### {k}\n{v}" for k, v in list(appfiles.items())[:6]) or "(none)"
    prompt = PROMPT.format(cat=cat, diff=diff, instr=instr[:4000], docker=docker[:3000], test=test[:6000], appfiles=appf[:6000])
    r = client.chat.completions.create(model=a.model,
        messages=[{"role": "system", "content": "You output only strict JSON."},
                  {"role": "user", "content": prompt}],
        max_completion_tokens=16000, reasoning_effort=a.effort)
    txt = r.choices[0].message.content or ""
    txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        d = json.loads(txt)
    except Exception as e:
        print("PARSE_FAIL", str(e)[:120]); print(txt[:300]); return 1
    out = a.out; os.makedirs(os.path.join(out, "tests"), exist_ok=True)
    # task.yaml
    with open(os.path.join(out, "task.yaml"), "w") as w:
        w.write("instruction: |-\n")
        for ln in d["instruction"].splitlines(): w.write("  " + ln + "\n")
        w.write(f"author_email: synth\ndifficulty: {diff}\ncategory: {cat}\ntags: [synthetic, tb-variant]\n")
        w.write("parser_name: pytest\nmax_agent_timeout_sec: 420.0\nmax_test_timeout_sec: 120.0\n")
        w.write("test_scripts: [setup-uv-pytest.sh, run-uv-pytest.sh]\nrun_tests_in_same_shell: false\n")
    open(os.path.join(out, "Dockerfile"), "w").write(d["dockerfile"])
    open(os.path.join(out, "solution.sh"), "w").write(d["solution_sh"])
    open(os.path.join(out, "tests", "test_outputs.py"), "w").write(d["test_file"])
    for rel, content in (d.get("app_files") or {}).items():
        fp = os.path.join(out, rel.lstrip("/"))
        os.makedirs(os.path.dirname(fp) or out, exist_ok=True)
        open(fp, "w").write(content)
    print("WROTE", out, "| files:", len(d.get("app_files") or {}) + 4)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
