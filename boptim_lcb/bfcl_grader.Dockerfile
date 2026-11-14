# Sandboxed BFCL (Berkeley Function-Calling Leaderboard, ICML 2025) image: the official bfcl-eval package from PyPI, used offline for
# (1) building the prompt-mode prompts from the bundled test files and (2) the AST checker.
#   docker build -t bfcl-grader -f boptim_lcb/bfcl_grader.Dockerfile boptim_lcb
#   docker run --rm --network none -v /srv/scratch/jli199/bfcl_stage:/w bfcl-grader python /w/bfcl_tool.py prep /w/bfcl.jsonl
#   docker run --rm --network none -v /srv/scratch/jli199/bfcl_stage:/w bfcl-grader python /w/bfcl_tool.py grade /w/bfcl.jsonl /w/GEN.jsonl /w/OUT.json
FROM python:3.12-slim
RUN pip install --no-cache-dir bfcl-eval
RUN python -c "import bfcl_eval, os; print('bfcl_eval at', os.path.dirname(bfcl_eval.__file__)); print(sorted(os.listdir(os.path.dirname(bfcl_eval.__file__))))"
