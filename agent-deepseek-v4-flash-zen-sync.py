#!/usr/bin/env python3
"""Wrapper — runs agentknit against DeepSeek V4 Flash (free) via the Zen
endpoint (opencode.ai/zen) using a local subprocess script.

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-deepseek-v4-flash-zen "<task>"           # one-shot
    agent-deepseek-v4-flash-zen                    # interactive REPL
    agent-deepseek-v4-flash-zen --session <id>     # explicit session override
    agent-deepseek-v4-flash-zen --non-interactive  # disable ask_user_question tool
"""

import os
import sys

MODEL = "run:///home/martin/bin/opencode-free-deepseek-v4-flash-completions.py"
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

project_root = os.path.dirname(os.path.realpath(__file__))
# Only prepend project_root if it contains a local agentknit package — avoids
# shadowing an installed package when the script lives in $HOME next to the
# agentknit project directory (which has no __init__.py at its root).
if os.path.exists(os.path.join(project_root, "agentknit", "__init__.py")):
    sys.path.insert(0, project_root)


# Inject the model as argv[1] if not already present.
if len(sys.argv) < 2 or sys.argv[1] != MODEL:
    sys.argv.insert(1, MODEL)

import agentknit
agentknit.enable_rtk_rewrite()
agentknit.main()
