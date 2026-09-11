#!/usr/bin/env python3
"""Textual TUI for Kimi K3 through the official Kimi Coding Plan API.

The API key is read from the system keyring (service ``login2``, username
``kimi_api_key``); it is never placed in this script, an environment variable,
or a command-line argument.

Usage:
    agentknit-kimi-k3-tui "<task>"           # open TUI with task prefilled
    agentknit-kimi-k3-tui                    # interactive TUI
    agentknit-kimi-k3-tui --session <id>     # resume a previous session
    agentknit-kimi-k3-tui --non-interactive  # disable ask_user_question
"""

import os
import sys

import agentknit

MODEL = "k3"
ENDPOINT = "https://api.kimi.com/coding/v1"
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


def _create_k3_client(schema: dict | None = None):
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
    # Ensure generated resume commands return to this TUI launcher.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, ENDPOINT)
    schema["keyring_service"] = "login2"
    schema["keyring_username"] = "kimi_api_key"
    schema["display_name"] = "Kimi K3 (Kimi Coding Plan)"
    # Context window measured by llmprobe (reports/k3).
    schema["context_window"] = 1048576

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
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 1024 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=1024,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
