#!/usr/bin/env python3
"""Agent wrapper for glm-5.3 via api.z.ai.

glm-5.3 is z.ai's flagship model, on the same coding endpoint as glm-5.2.
API key retrieved from keyring (service: z.ai, username: api_key).

Usage:
    agent-glm-5.3 "<task>"           # one-shot
    agent-glm-5.3                    # interactive REPL
    agent-glm-5.3 --non-interactive  # disable ask_user_question tool
"""

import os
import sys

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)

import agentknit

MODEL    = "glm-5.3"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

# Load spec and inject keyring config so agentknit resolves the API key
# via keyring directly, without any shared env-var convention.
schema = agentknit.load_specification(MODEL, ENDPOINT)
schema["keyring_service"]  = "z.ai"
schema["keyring_username"] = "api_key"

# Model-specific system prompt supplement to counter glm's tendency
# to assume HOME is /home/user.
_home = os.path.expanduser("~")
_cwd  = os.getcwd()
_SUPPLEMENT = (
    f"Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}."
)

# Parse simple flags we care about.
_non_interactive = "--non-interactive" in sys.argv
_session_id = None
if "--session" in sys.argv:
    idx = sys.argv.index("--session")
    if idx + 1 < len(sys.argv):
        _session_id = sys.argv[idx + 1]

# Collect task from CLI args or stdin.
# Skip known flags (--non-interactive is boolean; --session takes a value) so
# the session-id value is never mistaken for a task argument.
_FLAGS_WITH_VALUE = {"--session"}
_task_args: list[str] = []
_skip_next = False
for _arg in sys.argv[1:]:
    if _skip_next:
        _skip_next = False
        continue
    if _arg in _FLAGS_WITH_VALUE:
        _skip_next = True
        continue
    if _arg.startswith("--"):
        continue
    _task_args.append(_arg)
_task = " ".join(_task_args) if _task_args else None

if _task:
    agentknit.run_task(
        schema,
        _task,
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
    )
elif not sys.stdin.isatty():
    _task = sys.stdin.read().strip()
    if _task:
        agentknit.run_task(
            schema,
            _task,
            non_interactive=_non_interactive,
            session_id=_session_id,
            system_prompt_supplement=_SUPPLEMENT,
        )
else:
    agentknit.run_repl(
        schema,
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
    )
