#!/usr/bin/env python3
"""Run agentknit with DeepSeek Flash through the official DeepSeek API.

Endpoint and key source come from the inference-db entry ``deepseek``.

Usage:
    agentknit-deepseek-flash "<task>"           # one-shot
    agentknit-deepseek-flash                    # interactive REPL
    agentknit-deepseek-flash --session <id>     # resume a previous session
    agentknit-deepseek-flash --non-interactive  # disable ask_user_question
    echo "<task>" | agentknit-deepseek-flash    # task on stdin
"""

from __future__ import annotations

import os
import sys

import agentknit
from agentknit.async_toolkit import enable_nohup

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


def main() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    # Resume hints must re-invoke this launcher, not the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    # The inference-db entry pins the endpoint even when the cached probe
    # spec carries a stale one (e.g. the Azure deployment probed earlier).
    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "DeepSeek Flash (official API)"
    # Context window measured by llmprobe (reports/deepseek-flash).
    schema["context_window"] = 1048576
    # deepseek-flash streams its reasoning trace as `reasoning_content`
    # SSE deltas whenever it actually reasons: trivially-easy prompts get 0
    # reasoning tokens (and report `reasoning_tokens: 0`), real ones stream
    # the trace before content (measured against api.deepseek.com on
    # 2026-09-14).  agentknit only enables streaming — and with it the
    # reasoning_delta events the TUI renders as dim italic text — when the
    # spec declares the capability; the default in-memory spec does not.
    schema["provider_api_support"] = {"streaming": {"supported": True}}
    # DeepSeek's Context Caching on Disk keeps entries "a few hours to a few
    # days" after last use (docs: "automatically cleared, usually within a
    # few hours to a few days"); 6h is the conservative end.  Declaring it
    # classifies a resume after that as a cold resume (expected full
    # re-write) instead of a spurious cache_proof_missing warning.
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
