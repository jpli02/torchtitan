"""Adds reasoning_effort to every OpenAI chat-completions call when GPT_REASONING_EFFORT is set.

The boptim-agent checkout that saves trajectories (worktree cifar-save-trajectory) has an optim/chatgpt.py from before the
GPT_REASONING_EFFORT option; this hook gives its runs that option without touching that checkout. It is loaded only when
cifar_bo/pyhooks is on PYTHONPATH (cifar_bo/launch.sh does that for arms with a reasoning effort).
"""
import os

_effort = os.environ.get("GPT_REASONING_EFFORT")
if _effort:
    try:
        from openai.resources.chat import completions as _completions

        _original_create = _completions.Completions.create

        def _create_with_effort(self, *args, **kwargs):
            kwargs.setdefault("reasoning_effort", _effort)
            return _original_create(self, *args, **kwargs)

        _completions.Completions.create = _create_with_effort
    except Exception as e:  # never take a run down
        print(f"[sitecustomize] reasoning_effort hook not installed: {type(e).__name__}: {e}")
