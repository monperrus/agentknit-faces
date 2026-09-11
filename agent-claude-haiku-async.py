#!/usr/bin/env python3
"""Async agent — Claude Haiku backend.

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-claude-haiku-async "<task>"           # one-shot
    agent-claude-haiku-async                    # interactive REPL
    agent-claude-haiku-async --session <id>     # explicit session override
"""

from __future__ import annotations

import importlib.util
import os

MODEL = "run:///home/martin/bin/claude-haiku-completions.py"
# Claude Haiku's context window (llmprobe has no report for this wrapper;
# 200k is Anthropic's standard Haiku context size).
CONTEXT_WINDOW = 200000
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

_spec = importlib.util.spec_from_file_location(
    "_async_agent",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "agent-deepseek-v4-flash-async.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

if __name__ == "__main__":
    _mod.main(MODEL, CONTEXT_WINDOW)
