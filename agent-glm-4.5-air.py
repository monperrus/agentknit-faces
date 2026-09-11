#!/usr/bin/env python3
"""Agent wrapper for glm-4.5-air via api.z.ai.

glm-4.5-air is z.ai's lightweight (flash/mini) model on the same endpoint
as glm-5.2. API key retrieved from keyring (service: z.ai, username: api_key).

Usage:
    agentknit-glm-4.5-air "<task>"           # one-shot
    agentknit-glm-4.5-air                    # interactive REPL
    agentknit-glm-4.5-air --non-interactive  # disable ask_user_question tool
"""

import os
import sys

import agentknit
from agentknit.async_toolkit import enable_nohup

MODEL    = "glm-4.5-air"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

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


def entry() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    # Resume hints must re-invoke this launcher, not the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    # Load spec and inject keyring config so agentknit resolves the API key
    # via keyring directly, without any shared env-var convention.
    schema = agentknit.load_specification(MODEL, ENDPOINT)
    schema["keyring_service"]  = "z.ai"
    schema["keyring_username"] = "api_key"
    # z.ai publishes no context-window figure for glm-4.5-air; llmprobe found
    # none either. GLM-4.5-Air ships with 131072 tokens.
    schema["context_window"] = 131072

    # Async shell tools ("nohup" / "nohup_query"): definitions and
    # implementations live in agentknit.async_toolkit; one call wires specs +
    # dispatch.
    enable_nohup(schema)

    # Parse simple flags we care about.
    non_interactive = "--non-interactive" in sys.argv
    session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            session_id = sys.argv[idx + 1]

    # Collect task from CLI args or stdin.
    # Skip known flags (--non-interactive is boolean; --session takes a value)
    # so the session-id value is never mistaken for a task argument.
    flags_with_value = {"--session"}
    task_args: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in flags_with_value:
            skip_next = True
            continue
        if arg.startswith("--"):
            continue
        task_args.append(arg)
    task = " ".join(task_args) if task_args else None

    common = dict(
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
    )
    if task:
        agentknit.run_task(schema, task, **common)
    elif not sys.stdin.isatty():
        stdin_task = sys.stdin.read().strip()
        if stdin_task:
            agentknit.run_task(schema, stdin_task, **common)
        else:
            agentknit.run_repl(schema, **common)
    else:
        agentknit.run_repl(schema, **common)


if __name__ == "__main__":
    entry()
