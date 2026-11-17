#!/usr/bin/env python3
"""Run agentknit with GPT-5.6 Sol through Surplus Intelligence.

Endpoint and key source come from the inference-db entry ``surplus-min10``.
Surplus serves discounted capacity from changing upstream sellers; requests
whose discount falls below the key's minimum are rejected by the marketplace.

Usage:
    agentknit-gpt-5.6-sol-surplus "<task>"           # one-shot
    agentknit-gpt-5.6-sol-surplus                    # interactive REPL
    agentknit-gpt-5.6-sol-surplus --session <id>     # resume a previous session
    agentknit-gpt-5.6-sol-surplus --non-interactive  # disable ask_user_question
    echo "<task>" | agentknit-gpt-5.6-sol-surplus    # task on stdin
"""

from __future__ import annotations

import os
import sys

import agentknit
from agentknit.async_toolkit import enable_nohup

MODEL = "gpt-5.6-sol"
INFERENCE_DB = "surplus-min10"

# Both values come from Surplus's live /min10/v1/models catalogue entry,
# queried 2026-11-17. Leave 25% headroom for the final reply and tool result.
CONTEXT_WINDOW = 1_050_000
COMPACTION_TRIGGER_TOKENS = 787_500

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

    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "GPT-5.6 Sol (Surplus Intelligence)"
    schema["context_window"] = CONTEXT_WINDOW
    schema["compaction_trigger_tokens"] = COMPACTION_TRIGGER_TOKENS
    # The live model catalogue declares streaming support.
    schema["provider_api_support"] = {"streaming": {"supported": True}}

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
        # Surplus can route consecutive calls to different upstream sellers.
        # A cache miss can therefore reflect a route change rather than a
        # broken conversation, so do not make cache hits a hard requirement.
        strict_cache_proof=False,
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
