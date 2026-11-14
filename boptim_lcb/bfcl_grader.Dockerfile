# Sandboxed BFCL (Berkeley Function-Calling Leaderboard, ICML 2025) image: the official bfcl-eval package from PyPI, used offline for
# (1) building the prompt-mode prompts and (2) the AST checker. soundfile is an undeclared import of one of its model handlers.
#   docker build -t bfcl-grader -f boptim_lcb/bfcl_grader.Dockerfile boptim_lcb
#   docker run --rm --network none -v /srv/scratch/jli199/bfcl_stage:/w bfcl-grader python /w/bfcl_tool.py prep /w/raw /w/bfcl.jsonl
#   docker run --rm --network none -v /srv/scratch/jli199/bfcl_stage:/w bfcl-grader python /w/bfcl_tool.py grade /w/bfcl.jsonl /w/GEN.jsonl /w/OUT.json
FROM python:3.12-slim
RUN pip install --no-cache-dir bfcl-eval soundfile
RUN python -c "from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker; print(\"ast_checker import ok\")" || true
