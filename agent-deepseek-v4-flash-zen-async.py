#!/usr/bin/env python3
"""Agent-workflow wrap — DeepSeek V4 Flash via opencode.ai (Zen backend).

Usage:
    agent-deepseek-v4-flash-zen "<task>"           # one-shot
    agent-deepseek-v4-flash-zen                    # interactive REPL
    agent-deepseek-v4-flash-zen --session <id>     # explicit session override
"""

from __future__ import annotations

import importlib.util
import os

CONFIG_PATH = "/home/martin/workspace/prototypes/async-agent/agent-deepseek-v4-flash-zen.py"

# Let agentknit know which script to re-invoke to resume a session.
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

_spec = importlib.util.spec_from_file_location("_agent_zen_config", CONFIG_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# The config module sets AGENTKNIT_RESUME_COMMAND to its own path; override
# back to this wrapper so resume re-invokes the correct script.
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

if __name__ == "__main__":
    _mod._agent_main(_mod.MODEL, endpoint=getattr(_mod, "ENDPOINT", None))
