#!/usr/bin/env python3
"""Wrapper — runs agent_probe against DeepSeek V4 Flash (free) via opencode.ai subprocess.

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-opencode-free-deepseek-v4-flash "<task>"           # one-shot
    agent-opencode-free-deepseek-v4-flash                    # interactive REPL
    agent-opencode-free-deepseek-v4-flash --session <id>     # explicit session override
    agent-opencode-free-deepseek-v4-flash --non-interactive  # disable ask_user_question tool
"""

from __future__ import annotations

import os
import re
import sys
import importlib.util

_URL_RE = re.compile(r"(https?://\S+)")


def _osc8(url: str) -> str:
    return f"\033]8;;{url}\033\\{url}\033]8;;\033\\"


class _Osc8Stdout:
    """Wrap stdout to rewrite bare URLs as OSC 8 terminal hyperlinks."""

    def __init__(self, wrapped):
        self._w = wrapped
        self._buf = ""

    def _rewrite(self, text: str) -> str:
        return _URL_RE.sub(lambda m: _osc8(m.group(1)), text)

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._w.write(self._rewrite(line) + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._w.write(self._rewrite(self._buf))
            self._buf = ""
        self._w.flush()

    def __getattr__(self, name):
        return getattr(self._w, name)


sys.stdout = _Osc8Stdout(sys.stdout)

MODEL = "run:///home/martin/bin/opencode-free-deepseek-v4-flash-completions.py"
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

AGENT_PROBE_DIR = "/home/martin/workspace/prototypes/probe-model-tools"
sys.path.insert(0, AGENT_PROBE_DIR)
script_path = os.path.join(AGENT_PROBE_DIR, "agent_probe.py")

spec = importlib.util.spec_from_file_location("agent_probe", script_path)
assert spec is not None and spec.loader is not None
agent_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_probe)

# Inject the model as argv[1] if not already present.
if len(sys.argv) < 2 or sys.argv[1] != MODEL:
    sys.argv.insert(1, MODEL)

agent_probe.main()
