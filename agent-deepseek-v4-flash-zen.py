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

import os
import sys
import importlib.util

MODEL = "run:///home/martin/bin/opencode-free-deepseek-v4-flash-completions.py"

AGENT_PROBE_DIR = "/home/martin/workspace/prototypes/probe-model-tools"
sys.path.insert(0, AGENT_PROBE_DIR)
script_path = os.path.join(AGENT_PROBE_DIR, "agent_probe.py")

spec = importlib.util.spec_from_file_location("agent_probe", script_path)
agent_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_probe)

# Inject the model as argv[1] if not already present.
if len(sys.argv) < 2 or sys.argv[1] != MODEL:
    sys.argv.insert(1, MODEL)

agent_probe.main()
