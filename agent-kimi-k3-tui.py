#!/usr/bin/env python3
"""Textual TUI for Kimi K3 through the official Kimi Coding Plan API.

The API key is read from the system keyring (service ``login2``, username
``kimi_api_key``); it is never placed in this script, an environment variable,
or a command-line argument.

Usage:
    agent-kimi-k3-tui.py "<task>"           # open TUI with task prefilled
    agent-kimi-k3-tui.py                    # interactive TUI
    agent-kimi-k3-tui.py --session <id>     # resume a previous session
    agent-kimi-k3-tui.py --non-interactive  # disable ask_user_question
"""

import os
import sys


MODEL = "k3"
ENDPOINT = "https://api.kimi.com/coding/v1"

project_root = os.path.dirname(os.path.realpath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ensure generated resume commands return to this TUI launcher.
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

import agentknit
from agentknit import validate_schema
from agentknit.exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    AuthenticationError,
    PricingLimitExceededError,
    RateLimitError,
)
from agentknit_tui import AgentTUI
import agentknit_tui.app as agent_tui_app


schema = agentknit.load_specification(MODEL, ENDPOINT)
schema["keyring_service"] = "login2"
schema["keyring_username"] = "kimi_api_key"
schema["display_name"] = "Kimi K3 (Kimi Coding Plan)"

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


def _create_k3_client(_schema: dict | None = None):
    """Create a Coding Plan client with K3's mandatory temperature setting."""
    client = agentknit.create_client(schema)
    create = client.chat.completions.create

    def create_with_k3_temperature(*args, **kwargs):
        # K3's Coding Plan endpoint rejects agentknit's generic temperature=0.
        kwargs["temperature"] = 1
        return create(*args, **kwargs)

    client.chat.completions.create = create_with_k3_temperature
    return client


def main() -> int:
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
    task = " ".join(task_args)

    try:
        validate_schema(schema)
        agentknit.check_and_display_pricing(schema)
    except AgentSpecDisabledError as exc:
        print(f"Agent disabled: {exc.comment or exc}", file=sys.stderr)
        return 2
    except AgentSpecInvalidError as exc:
        print(exc, file=sys.stderr)
        return 2
    except PricingLimitExceededError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        return 2
    except RateLimitError as exc:
        print(f"Rate limited: {exc}", file=sys.stderr)
        return 2

    # AgentTUI imports create_client into its app module, so patch that local
    # binding before construction.  Its turn worker then receives our adapted
    # client without changing agentknit's generic behaviour for other models.
    agent_tui_app.create_client = _create_k3_client
    app = AgentTUI(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=task,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
