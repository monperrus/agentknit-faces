#!/usr/bin/env python3
"""Run agentknit against GitHub Copilot GPT-5.6 Luna via subprocess.

Usage:
    agent-gpt-5.6-luna-copilot.py "<task>"           # one-shot
    agent-gpt-5.6-luna-copilot.py                    # interactive REPL
    agent-gpt-5.6-luna-copilot.py --session <id>     # explicit session override
    agent-gpt-5.6-luna-copilot.py --non-interactive  # disable ask_user_question tool
"""

import os
import sys

COMPLETIONS_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "copilot-gpt-5.6-luna-gpt-5.6-luna.py",
)
MODEL = f"run://{COMPLETIONS_SCRIPT}"

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)

import agentknit


if len(sys.argv) > 1 and sys.argv[1] == COMPLETIONS_SCRIPT:
    sys.argv[1] = MODEL
elif len(sys.argv) < 2 or sys.argv[1] != MODEL:
    sys.argv.insert(1, MODEL)

agentknit.main()
