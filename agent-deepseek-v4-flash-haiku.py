#!/usr/bin/env python3
"""Async agent — DeepSeek V4 Flash via Claude Haiku backend.

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-deepseek-v4-flash-haiku "<task>"           # one-shot
    agent-deepseek-v4-flash-haiku                    # interactive REPL
    agent-deepseek-v4-flash-haiku --session <id>     # explicit session override
"""

from __future__ import annotations

import importlib.util
import os

MODEL = "run:///home/martin/bin/claude-haiku-completions.py"
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

_spec = importlib.util.spec_from_file_location(
    "_async_agent",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "agent-deepseek-v4-flash-async.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

if __name__ == "__main__":
    _mod.main(MODEL)
