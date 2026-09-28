"""terminus-2 with a task_complete verification gate.

Measured motivation: across continue10k and batching5k, every one of the 8 trials
that stopped without timing out and without resolving set task_complete=true
while passing a mean 28-33% of the task's tests. batching5k's early quitters
stopped at a median 299s of the ~720s available -- they abandoned 58% of their
clock to declare a success that was not one.

terminus-2 already double-confirms, but the confirmation is toothless: the model
re-emits task_complete with no commands and the run ends. This gate makes the
confirmation turn earn it:

  1. The confirmation prompt demands concrete verification, naming the checks
     that actually decide the grade (run the tests, inspect the output files).
  2. A confirmation that executes NO commands is refused, up to MAX_PUSHBACKS
     times. The model must actually look before it claims.

The cap matters: without it a stubborn model burns its whole budget in a
confirm/refuse loop, trading one failure mode for a worse one. After
MAX_PUSHBACKS refusals the claim is honoured so the trial still terminates.

Usage:
  PYTHONPATH=<this dir> tb run --agent-import-path verify_agent:TerminusVerify ...

Set TB_VERIFY_MAX_PUSHBACKS to change the cap (default 2); 0 disables the gate
and falls back to stock terminus-2 behaviour.
"""

import os

from terminal_bench.agents.terminus_2 import Terminus2

MAX_PUSHBACKS = int(os.environ.get("TB_VERIFY_MAX_PUSHBACKS", "2"))


class TerminusVerify(Terminus2):
    """Terminus2 that requires evidence before accepting task completion."""

    @staticmethod
    def name() -> str:
        return "terminus-verify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pushbacks = 0

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Demand verification rather than a bare re-affirmation."""
        if self._parser_name != "json":
            return super()._get_completion_confirmation_message(terminal_output)
        return (
            f"Current terminal state:\n{terminal_output}\n\n"
            "Before this is graded, VERIFY your work instead of assuming it is "
            "correct. Run commands that would catch a mistake: execute the "
            "task's tests if there are any, cat the files you created and check "
            "their contents against what was asked, re-read the task "
            "requirements and confirm every one is met.\n\n"
            'Put those verification commands in your "commands" list now. '
            "If the checks reveal a problem, fix it and keep working. Only once "
            "you have seen the checks pass should you include "
            '"task_complete": true in a later response.'
        )

    def _handle_llm_interaction(self, *args, **kwargs):
        commands, is_task_complete, feedback = super()._handle_llm_interaction(
            *args, **kwargs
        )

        # Only police the confirmation turn: the first claim already draws the
        # verification prompt above, and this refuses a confirmation that did no
        # work. A confirmation carrying commands is accepted -- the model looked,
        # which is the whole point of the gate.
        if (
            MAX_PUSHBACKS > 0
            and is_task_complete
            and getattr(self, "_pending_completion", False)
            and not commands
            and self._pushbacks < MAX_PUSHBACKS
        ):
            self._pushbacks += 1
            return commands, False, feedback

        return commands, is_task_complete, feedback
