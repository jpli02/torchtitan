import re
p = "/home/jli199/terminal_bench_eval/.venv/lib/python3.12/site-packages/terminal_bench/llms/lite_llm.py"
src = open(p).read()
if "TB_MAX_TOKENS" in src:
    print("already patched")
else:
    # add an os import guard + inject max_tokens into the completion call
    anchor = "            response = litellm.completion(\n                model=self._model_name,\n                messages=messages,\n                temperature=self._temperature,\n"
    inject = anchor + '                max_tokens=(int(__import__("os").environ["TB_MAX_TOKENS"]) if __import__("os").environ.get("TB_MAX_TOKENS") else None),\n'
    if anchor not in src:
        print("ANCHOR NOT FOUND")
    else:
        src = src.replace(anchor, inject, 1)
        open(p, "w").write(src)
        print("patched: max_tokens honors TB_MAX_TOKENS env")
