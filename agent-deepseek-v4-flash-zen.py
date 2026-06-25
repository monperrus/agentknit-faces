#!/usr/bin/env python3
"""Async agent — DeepSeek V4 Flash via opencode.ai (Zen backend).

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-deepseek-v4-flash-zen "<task>"           # one-shot
    agent-deepseek-v4-flash-zen                    # interactive REPL
    agent-deepseek-v4-flash-zen --session <id>     # explicit session override
"""

from __future__ import annotations

import importlib.util
import os

# This backend MUST emit a single valid /completions JSON document on stdout.
# Any surfaced reasoning must stay inside that JSON payload, not be printed
# as extra text before or after it.
MODEL = "run:///home/martin/bin/opencode-free-deepseek-v4-flash-completions.py"
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
