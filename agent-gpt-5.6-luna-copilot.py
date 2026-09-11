#!/usr/bin/env python3
"""Run agentknit against GitHub Copilot GPT-5.6 Luna via subprocess.

Usage:
    agent-gpt-5.6-luna-copilot.py "<task>"           # one-shot
    agent-gpt-5.6-luna-copilot.py                    # interactive REPL
    agent-gpt-5.6-luna-copilot.py --session <id>     # explicit session override
    agent-gpt-5.6-luna-copilot.py --non-interactive  # disable ask_user_question tool

GPT-5.6 Luna caches only eligible prefixes of at least 1,024 tokens. Strict
cache-hit enforcement is therefore disabled for this wrapper; cache accounting
is still recorded whenever Copilot serves a cached prefix.
"""

import json
import os
import sys
import tempfile

COMPLETIONS_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "copilot-gpt-5.6-luna.py",
)
MODEL = f"run://{COMPLETIONS_SCRIPT}"
# Max prompt tokens measured by llmprobe (reports/gpt-5.6-luna).
CONTEXT_WINDOW = 922000

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)

import agentknit


if len(sys.argv) > 1 and sys.argv[1] == COMPLETIONS_SCRIPT:
    sys.argv[1] = MODEL
elif len(sys.argv) < 2 or sys.argv[1] != MODEL:
    sys.argv.insert(1, MODEL)

if "--no-strict-cache-proof" not in sys.argv:
    sys.argv.insert(2, "--no-strict-cache-proof")

# agentknit's CLI has no --context-window flag; pass it via an explicit spec
# file (--spec-path skips all name-based lookup and probing). Mirror the
# in-memory default run:// spec (tool_specs + tools), with context_window added.
_spec = {
    "model": COMPLETIONS_SCRIPT,
    "endpoint": MODEL,
    "status": "default",
    "context_window": CONTEXT_WINDOW,
    "tool_specs": agentknit._core._DEFAULT_TOOL_SCHEMA,
    "tools": agentknit._core._DEFAULT_TOOLS,
    "behaviour": {"call_delivery_mode": "structured_tool_calls"},
}
_spec_file = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", prefix="agent_spec_gpt-5.6-luna_", delete=False
)
json.dump(_spec, _spec_file)
_spec_file.close()
sys.argv.insert(3, "--spec-path")
sys.argv.insert(4, _spec_file.name)

agentknit.main()
