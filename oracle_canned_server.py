"""Canned oracle "model" for terminus-2: an OpenAI-compatible server that replays a
task's reference solution through the REAL harness so the logged trajectory has
real terminal output (item 4: fix the oracle's synthetic turn-2 construction).

Turn 0 : write solution.sh via heredoc and run it (task_complete=false, so the
         harness returns the real terminal screen instead of the confirm prompt)
Turn k : if the last screen ends with a shell prompt -> declare task_complete
         else -> wait (empty keystrokes, 30 s), up to --max_waits times
Confirm: reply task_complete again with no commands.

Usage: python oracle_canned_server.py --task <task_id> --port 8160 [--tasks_dir DIR]
"""
import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

DELIM = "__ORACLE_EOF__"
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--port", type=int, default=8160)
ap.add_argument("--tasks_dir", default="/home/jli199/terminal_bench_eval/tb_tasks")
ap.add_argument("--duration", type=float, default=60.0, help="seconds to wait after launching the solution")
ap.add_argument("--max_waits", type=int, default=20)
args = ap.parse_args()
TB = args.tasks_dir


def instruction(task):
    y = open(f"{TB}/{task}/task.yaml", errors="ignore").read()
    m = re.search(r"^instruction:\s*\|-?\s*\n((?:[ \t]+.*\n?)+)", y, re.M)
    return " ".join(m.group(1).split()) if m else task


def commands_for(task):
    sh, yml = f"{TB}/{task}/solution.sh", f"{TB}/{task}/solution.yaml"
    if os.path.isfile(sh):
        lines = [l for l in open(sh, errors="ignore").read().splitlines() if "terminal-bench-canary" not in l]
        script = "\n".join(lines).strip("\n")
        assert DELIM not in script, task
        ks = f"cat > /tmp/oracle_solve.sh << '{DELIM}'\n{script}\n{DELIM}\nbash /tmp/oracle_solve.sh\n"
        return [{"keystrokes": ks, "duration": args.duration}]
    if os.path.isfile(yml):
        raw = [l for l in open(yml, errors="ignore").read().splitlines() if "terminal-bench-canary" not in l]
        out = []
        for s in yaml.safe_load("\n".join(raw)) or []:
            cmd = str(s.get("command", ""))
            if s.get("append_enter", True):
                cmd += "\n"
            out.append({"keystrokes": cmd, "duration": max(float(s.get("min_timeout_sec", 1.0)), 1.0)})
        return out
    raise SystemExit(f"{task}: no solution.sh or solution.yaml")


INSTR = instruction(args.task)
CMDS = commands_for(args.task)
PROMPT_RE = re.compile(r"[\w.@\-]+@[\w.\-]+:[^\n]*[#$] ?$")


def reply(messages):
    idx = sum(1 for m in messages if m.get("role") == "assistant")
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if isinstance(last_user, list):
        last_user = " ".join(p.get("text", "") for p in last_user if isinstance(p, dict))
    if idx == 0:
        body = {"analysis": f"The task is: {INSTR[:220]} I know the exact steps that solve this, so I will apply them directly.",
                "plan": "Write the solution script, run it, then check the terminal before marking the task complete.",
                "commands": CMDS, "task_complete": False}
        kind = "solve"
    elif "mark the task as complete" in last_user:
        body = {"analysis": "The solution has run and the terminal is back at the prompt.",
                "plan": "Confirm completion so the task can be graded.", "commands": [], "task_complete": True}
        kind = "confirm"
    else:
        tail = last_user.rstrip()
        last_line = tail.splitlines()[-1] if tail else ""
        prompt_back = bool(PROMPT_RE.search(last_line.strip()))
        waits = idx - 1
        if prompt_back or waits >= args.max_waits:
            body = {"analysis": "The solution script has finished and the shell prompt is back.",
                    "plan": "Mark the task complete.", "commands": [], "task_complete": True}
            kind = "done" if prompt_back else "done-maxwait"
        else:
            body = {"analysis": "The solution script is still running; the prompt has not returned yet.",
                    "plan": "Wait for it to finish before marking the task complete.",
                    "commands": [{"keystrokes": "", "duration": 30.0}], "task_complete": False}
            kind = "wait"
    print(f"[oracle-canned] {args.task} turn={idx} -> {kind}", flush=True)
    return json.dumps(body, indent=2)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True, "task": args.task})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "oracle_real", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        content = reply(req.get("messages", []))
        self._send(200, {"id": f"canned-{int(time.time()*1000)}", "object": "chat.completion", "created": int(time.time()),
                         "model": req.get("model", "oracle_real"),
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


print(f"[oracle-canned] serving task={args.task} on :{args.port} ({len(CMDS)} command(s))", flush=True)
HTTPServer(("127.0.0.1", args.port), H).serve_forever()
