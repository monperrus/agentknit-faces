#!/usr/bin/env python3
"""Textual TUI agent for glm-5.3 via api.z.ai.

Same schema as agent-glm-5.3.py (keyring z.ai/api_key, coding endpoint)
but rendered with agentknit-tui: persistent conversation pane, multiline
prompt, inline tool calls, live status bar.

Usage:
    agentknit-glm-5.3-tui "<task>"           # start TUI with task prefilled
    agentknit-glm-5.3-tui                    # interactive TUI
    agentknit-glm-5.3-tui --session <id>     # resume a previous trajectory
    agentknit-glm-5.3-tui --non-interactive  # drop ask_user* tools
"""

import os
import sys

import agentknit
from agentknit import validate_schema
from agentknit.async_toolkit import enable_nohup
from agentknit.exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    AuthenticationError,
    PricingLimitExceededError,
    RateLimitError,
)
from agentknit_tui import AgentTUI

MODEL    = "glm-5.3"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

# z.ai caches prompt prefixes in 64-token blocks and reports
# ``cached_tokens: 0`` for any prompt too short to fill a whole block, so such
# a call is not a caching failure.  Measured against api.z.ai on 2026-09-12:
# glm-5.3 reports no cache hit at 63/64/65 prompt tokens and its first hit
# (64 cached tokens) at 66.  Without this floor, agentknit's strict cache-proof
# mode flags every sub-block prompt as "no cache hit after the first call".
MIN_CACHEABLE_TOKENS = 66

_home = os.path.expanduser("~")
_cwd  = os.getcwd()
_SUPPLEMENT = (
    f"Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}.\n"
    f"When creating git commits, use the author/committer email "
    f"martin.monperrus@gnieh.org."
)


def main() -> int:
    # Resume hints must point here, not at the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, ENDPOINT)
    schema["keyring_service"]  = "z.ai"
    schema["keyring_username"] = "api_key"
    schema["display_name"]     = f"agentknit-glm-5.3-tui ({MODEL})"
    # Context window measured by llmprobe (reports/glm-5.3).
    schema["context_window"]   = 1048576
    # z.ai's minimum cacheable prompt prefix (see MIN_CACHEABLE_TOKENS).
    schema["min_cacheable_tokens"] = MIN_CACHEABLE_TOKENS

    # Async shell tools ("nohup" / "nohup_query"): definitions and
    # implementations live in agentknit.async_toolkit; one call wires specs +
    # dispatch.
    enable_nohup(schema)

    _non_interactive = "--non-interactive" in sys.argv
    _session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            _session_id = sys.argv[idx + 1]

    # Collect task from CLI args (one-shot mode pre-fills the prompt); skip
    # known flags so the session-id value is never mistaken for a task.
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

    try:
        validate_schema(schema)
        agentknit.check_and_display_pricing(schema)
    except AgentSpecDisabledError as exc:
        print(f"Agent disabled: {exc.comment or exc}", file=sys.stderr)
        return 2
    except AgentSpecInvalidError as exc:
        print(f"{exc}", file=sys.stderr)
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

    app = AgentTUI(
        schema,
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=_task or "",
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
