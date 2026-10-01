#!/usr/bin/env python3
"""Textual TUI for DeepSeek Flash through the official DeepSeek API.

Same schema as agent-deepseek-flash.py (inference-db entry ``deepseek``)
but rendered with agentknit-tui:
persistent conversation pane, multiline prompt, inline tool calls, live
status bar.

Usage:
    agent-deepseek-flash-tui "<task>"           # start TUI with task prefilled
    agent-deepseek-flash-tui                    # interactive TUI
    agent-deepseek-flash-tui --session <id>     # resume a previous trajectory
    agent-deepseek-flash-tui --non-interactive  # disable ask_user_question
"""

from __future__ import annotations

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

MODEL = "deepseek-flash"
INFERENCE_DB = "deepseek"

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


def main() -> int:
    # Resume hints must re-invoke this TUI launcher, not the generic CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    # The inference-db entry pins the endpoint even when the cached probe
    # spec carries a stale one.
    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "DeepSeek Flash (official API, TUI)"
    # Context window measured by llmprobe (reports/deepseek-flash).
    schema["context_window"] = 1048576
    # deepseek-flash streams its reasoning trace as `reasoning_content` SSE
    # deltas whenever it actually reasons (measured against api.deepseek.com on
    # 2026-09-14).  agentknit only enables streaming — and with it the
    # reasoning_delta events the TUI renders as dim italic text — when the spec
    # declares the capability; the default in-memory spec does not.
    schema["provider_api_support"] = {"streaming": {"supported": True}}
    # DeepSeek's disk-backed prefix cache outlives KV caches: entries are
    # "automatically cleared, usually within a few hours to a few days"
    # (api-docs.deepseek.com/guides/kv_cache).  6h is the conservative end;
    # a resume past it is a cold resume (full re-write expected), not a
    # cache_proof_missing surprise.  Also feeds the TUI warmth countdown.
    schema["cache_ttl_seconds"] = 21600

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

    # Collect the task from CLI args (it prefills the prompt); skip known
    # flags so the session-id value is never mistaken for a task.
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

    try:
        app = AgentTUI(
            schema,
            non_interactive=non_interactive,
            session_id=session_id,
            system_prompt_supplement=_SUPPLEMENT,
            prefill=task,
        )
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        return 2

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
