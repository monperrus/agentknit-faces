#!/usr/bin/env python3
"""Run agentknit with DeepSeek V4 Flash through the official DeepSeek API.

Endpoint: https://api.deepseek.com (OpenAI-compatible).  The API key is read
from the system keyring (service ``login2``, username ``deepseek_api_key``),
never from a source-controlled environment variable.

Usage:
    agentknit-deepseek-v4-flash "<task>"           # one-shot
    agentknit-deepseek-v4-flash                    # interactive REPL
    agentknit-deepseek-v4-flash --session <id>     # resume a previous session
    agentknit-deepseek-v4-flash --non-interactive  # disable ask_user_question
    echo "<task>" | agentknit-deepseek-v4-flash    # task on stdin
"""

from __future__ import annotations

import os
import sys

import agentknit
from agentknit.async_toolkit import enable_nohup

MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/v1"

_home = os.path.expanduser("~")
_cwd = os.getcwd()
_SUPPLEMENT = (
    "Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}.\n"
    "When creating git commits, use the author/committer email "
    "martin.monperrus@gnieh.org."
)


def main() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    # Resume hints must re-invoke this launcher, not the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, ENDPOINT)
    # load_specification() may return a cached probe whose "endpoint" is stale
    # (e.g. the Azure deployment probed earlier); pin it to the official API.
    schema["endpoint"] = ENDPOINT
    schema["keyring_service"] = "login2"
    schema["keyring_username"] = "deepseek_api_key"
    schema["display_name"] = "DeepSeek V4 Flash (official API)"
    # Context window measured by llmprobe (reports/deepseek-v4-flash).
    schema["context_window"] = 1048576

    # Async shell tools ("nohup" / "nohup_query"): definitions and
    # implementations live in agentknit.async_toolkit; one call wires specs +
    # dispatch.
    enable_nohup(schema)

    non_interactive = "--non-interactive" in sys.argv
    session_id = None
    if "--session" in sys.argv:
        index = sys.argv.index("--session")
        if index + 1 < len(sys.argv):
            session_id = sys.argv[index + 1]

    flags_with_value = {"--session"}
    task_args: list[str] = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in flags_with_value:
            skip_next = True
            continue
        if argument.startswith("--"):
            continue
        task_args.append(argument)

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
    main()
