#!/usr/bin/env python3
"""Textual TUI for Kimi K3 through the official Kimi Coding Plan API.

Endpoint, key source and K3's quirks (it only accepts ``temperature=1``) come
from the inference-db entry ``kimi-coding``.

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
INFERENCE_DB = "kimi-coding"
from agentknit import validate_schema
from agentknit.exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    AuthenticationError,
    PricingLimitExceededError,
    RateLimitError,
)
from agentknit_tui import AgentTUI


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
    # Ensure generated resume commands return to this TUI launcher.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "Kimi K3 (Kimi Coding Plan)"
    # Context window measured by llmprobe (reports/k3).
    schema["context_window"] = 1048576
    # Trigger compaction at 75 % of the window: the last observed prompt plus
    # one more assistant reply + tool result must still fit.
    schema["compaction_trigger_tokens"] = 786432
    # Kimi's implicit prefix cache carries a 5-minute TTL by default (see
    # agent-kimi-k3.py).  Declaring it classifies a resume after >5 min as a
    # cold resume (expected full cache re-write) instead of a spurious
    # cache_proof_missing warning, and lets the TUI count the warmth down.
    schema["cache_ttl_seconds"] = 300
    # k3 streams its reasoning trace as `reasoning_content` SSE deltas (every
    # call carries it — measured against api.kimi.com on 2026-09-12, 6/6).
    # agentknit only enables streaming — and with it the reasoning_delta
    # events the TUI renders as dim italic text — when the spec declares the
    # capability; the default in-memory spec does not.
    schema["provider_api_support"] = {"streaming": {"supported": True}}

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

    app = AgentTUI(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=task,
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 4096 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=4096,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
