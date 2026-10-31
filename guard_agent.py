"""terminus-2 with a terminal-state guard (plus the TerminusVerify completion gate).

Motivation (failure analysis of the 8 never-solved easy tasks, 2026-10-30): the one
broadly scaffold-fixable failure is TERMINAL DESYNC. Two forms:
  1. An unbalanced quote or unterminated heredoc leaves bash at the `>` continuation
     prompt; every later command is swallowed into it, the agent sees only its own
     echoed text, concludes "terminal issues" and gives up (fatal on create-bucket).
  2. The agent sends the next command while a blocking one is still running; commands
     queue up and echo without output (minio, git-workflow-hack, web-scraper).
Plus two budget sinks: the same keystrokes repeated many times, and declaring done
without looking (handled by TerminusVerify).

What the guard does, before sending each command batch:
  - keystrokes with an unbalanced quote (outside heredoc bodies) are NOT sent; the model
    gets a message naming the problem and re-plans;
  - if the pane's last line is a `>` continuation prompt, send Ctrl-C + Enter to reset
    and tell the model what happened;
  - if the pane's last line is not a shell prompt (previous command still running), wait
    up to GUARD_BUSY_WAIT_SEC for the prompt; if it never returns, do NOT send the new
    keystrokes: return the screen with a note that the command is still running, so the
    model can wait (empty command + duration) or interrupt deliberately;
  - identical keystrokes sent >= GUARD_REPEAT_N times in the recent window get a nudge
    appended to the terminal output.
All interventions are counted and logged; GUARD_MAX_PER_EPISODE caps the quote/busy
refusals per episode so a stubborn model cannot loop forever (then the keys go through).

Usage:
  PYTHONPATH=<this dir> tb run --agent-import-path guard_agent:TerminusGuard ...
Env: TB_GUARD=0 disables everything (-> TerminusVerify); TB_VERIFY_MAX_PUSHBACKS as before.
"""
import os
import re
import time

from terminal_bench.agents.terminus_2 import Terminus2

from verify_agent import TerminusVerify

GUARD = os.environ.get("TB_GUARD", "1") != "0"
GUARD_BUSY_WAIT_SEC = float(os.environ.get("TB_GUARD_BUSY_WAIT_SEC", "20"))
GUARD_REPEAT_N = int(os.environ.get("TB_GUARD_REPEAT_N", "3"))
GUARD_MAX_PER_EPISODE = int(os.environ.get("TB_GUARD_MAX_PER_EPISODE", "2"))

# A shell prompt ends the line with # or $ (the prompt may be glued to the previous command's
# output when that output had no trailing newline, e.g. 'archive/root@abc:/app#').
PROMPT_RE = re.compile(r"[#$] ?$")
PS2_RE = re.compile(r"(^|\n)(>|quote>|dquote>|heredoc>|bquote>|cmdsubst>|pipe>|cmdand>|cmdor>) ?$")
# Programs asking for input: the terminal is waiting for the agent, not busy.
INPUT_RE = re.compile(r"(\[y/n\]|\[yes/no\]|\(y/n\)|password|passphrase|\?\s*$|:\s*$|continue\??\s*$|press enter|more\]?--)", re.I)


def last_line(screen: str) -> str:
    lines = [l for l in screen.rstrip("\n").splitlines() if l.strip()]
    return lines[-1].rstrip() if lines else ""


def at_prompt(screen: str) -> bool:
    return bool(PROMPT_RE.search(last_line(screen)))


def at_continuation(screen: str) -> bool:
    return bool(PS2_RE.search(last_line(screen)))


def awaiting_input(screen: str) -> bool:
    """A program is asking the user something ([Y/n], password:, ...): not busy, type away."""
    return bool(INPUT_RE.search(last_line(screen)))


def strip_heredocs(text: str) -> str:
    """Remove heredoc bodies (<<DELIM ... DELIM) so their quotes are not counted."""
    out, i, lines = [], 0, text.split("\n")
    while i < len(lines):
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", lines[i])
        out.append(lines[i])
        i += 1
        if m:
            delim = m.group(1)
            while i < len(lines) and lines[i].strip() != delim:
                i += 1
            i += 1  # skip the delimiter line
    return "\n".join(out)


def unbalanced_quote(keystrokes: str):
    """Return "'" or '"' if a quote is left open (shell-style scan), else None."""
    s = strip_heredocs(keystrokes)
    in_sq = in_dq = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and not in_sq:
            i += 2
            continue
        if c == "'" and not in_dq:
            in_sq = not in_sq
        elif c == '"' and not in_sq:
            in_dq = not in_dq
        elif c == "#" and not in_sq and not in_dq and (i == 0 or s[i - 1] in " \t\n"):
            j = s.find("\n", i)
            i = len(s) if j < 0 else j
            continue
        i += 1
    if in_sq:
        return "'"
    if in_dq:
        return '"'
    return None


class TerminusGuard(TerminusVerify):
    @staticmethod
    def name() -> str:
        return "terminus-guard"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent = []
        self._stats = {"quote_refused": 0, "ps2_reset": 0, "busy_refused": 0, "busy_waited": 0, "repeat_nudged": 0}

    def _screen(self, session) -> str:
        try:
            return session.capture_pane(capture_entire=False)
        except Exception:
            return ""

    def _wait_prompt(self, session, max_sec: float):
        """Wait for the prompt. Returns (prompt_back, still_changing): if the screen keeps
        changing the command is still producing output; a static screen without a prompt is
        a program waiting for input, which the agent must be allowed to answer."""
        t0 = time.time()
        first = self._screen(session)
        last = first
        while time.time() - t0 < max_sec:
            if at_prompt(last) or awaiting_input(last):
                return True, False
            time.sleep(2.0)
            last = self._screen(session)
        if at_prompt(last) or awaiting_input(last):
            return True, False
        return False, last != first

    def _execute_commands(self, commands, session):
        if not GUARD or not commands:
            return super()._execute_commands(commands, session)
        notes = []
        refusals = 0
        to_run = []
        for cmd in commands:
            ks = cmd.keystrokes
            q = unbalanced_quote(ks)
            if q is not None and refusals < GUARD_MAX_PER_EPISODE:
                refusals += 1
                self._stats["quote_refused"] += 1
                notes.append(f"[guard] NOT SENT: these keystrokes leave a {q} quote open, which would hang bash at a "
                             f"continuation prompt: {ks[:160]!r}. Fix the quoting (prefer a quoted heredoc for JSON/scripts) and resend.")
                continue
            to_run.append(cmd)
        if to_run:
            screen = self._screen(session)
            if at_continuation(screen):
                self._stats["ps2_reset"] += 1
                session.send_keys(["C-c", "Enter"], block=False, min_timeout_sec=1.0)
                notes.append("[guard] bash was waiting at a continuation prompt (an earlier command had an unbalanced quote or "
                             "unterminated heredoc); sent Ctrl-C to reset it. Commands typed since then were NOT executed.")
            elif not at_prompt(screen) and not awaiting_input(screen):
                ready, changing = self._wait_prompt(session, GUARD_BUSY_WAIT_SEC)
                if ready:
                    self._stats["busy_waited"] += 1
                elif changing and refusals < GUARD_MAX_PER_EPISODE:
                    # still printing output after the wait: genuinely busy
                    refusals += 1
                    self._stats["busy_refused"] += 1
                    out = self._limit_output_length(session.get_incremental_output())
                    notes.append("[guard] the previous command is STILL RUNNING (output is still arriving, no shell prompt). Your new "
                                 "commands were NOT sent. Either wait for it (send an empty keystrokes command with a duration) or interrupt it with C-c.")
                    return False, "\n".join(notes) + "\n\n" + out
                else:
                    # static screen without a prompt: a program is waiting for input, let the agent answer
                    self._stats["busy_static_passed"] = self._stats.get("busy_static_passed", 0) + 1
        # repeat detector
        for cmd in to_run:
            self._recent.append(cmd.keystrokes.strip())
        self._recent = self._recent[-12:]
        rep = [k for k in set(self._recent) if self._recent.count(k) >= GUARD_REPEAT_N and any(c.keystrokes.strip() == k for c in to_run)]
        if rep:
            self._stats["repeat_nudged"] += 1
            notes.append(f"[guard] you have now run {rep[0][:80]!r} {self._recent.count(rep[0])} times. Repeating it will give the "
                         "same result; change the approach (read the error, try a different command, or verify an assumption).")
        timed_out, out = super()._execute_commands(to_run, session) if to_run else (False, self._limit_output_length(session.get_incremental_output()))
        if notes:
            out = "\n".join(notes) + "\n\n" + out
        return timed_out, out

    def perform_task(self, *args, **kwargs):
        try:
            return super().perform_task(*args, **kwargs)
        finally:
            self._logger.info(f"[guard] interventions: {self._stats}")
            try:
                ld = kwargs.get("logging_dir") or (args[2] if len(args) > 2 else None)
                if ld:
                    with open(os.path.join(str(ld), "guard_stats.txt"), "w") as f:
                        f.write(repr(self._stats) + "\n")
            except Exception:
                pass
