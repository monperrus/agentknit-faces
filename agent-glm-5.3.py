#!/usr/bin/env python3
"""Agent wrapper for glm-5.3 via api.z.ai.

glm-5.3 is z.ai's flagship model, on the same coding endpoint as glm-5.2.
API key retrieved from keyring (service: z.ai, username: api_key).

Usage:
    agentknit-glm-5.3 "<task>"           # one-shot
    agentknit-glm-5.3                    # interactive REPL
    agentknit-glm-5.3 --non-interactive  # disable ask_user_question tool
"""

import os
import sys

import agentknit
from agentknit.async_toolkit import enable_nohup

MODEL    = "glm-5.3"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

# z.ai caches prompt prefixes in 64-token blocks and reports
# ``cached_tokens: 0`` for any prompt too short to fill a whole block, so such
# a call is not a caching failure.  Measured against api.z.ai on 2026-09-12:
# glm-5.3 reports no cache hit at 63/64/65 prompt tokens and its first hit
# (64 cached tokens) at 66.  Without this floor, agentknit's strict cache-proof
# mode flags every sub-block prompt as "no cache hit after the first call", and
# fails closed on the first call of a session whose response carries no cache
# accounting at all.
MIN_CACHEABLE_TOKENS = 66

# Model-specific system prompt supplement to counter glm's tendency
# to assume HOME is /home/user.
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


def entry() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    # Resume hints must re-invoke this launcher, not the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    # Load spec and inject keyring config so agentknit resolves the API key
    # via keyring directly, without any shared env-var convention.
    schema = agentknit.load_specification(MODEL, ENDPOINT)
    schema["keyring_service"]  = "z.ai"
    schema["keyring_username"] = "api_key"
    # Context window measured by llmprobe (reports/glm-5.3).
    schema["context_window"] = 1048576
    # z.ai's minimum cacheable prompt prefix (see MIN_CACHEABLE_TOKENS).
    schema["min_cacheable_tokens"] = MIN_CACHEABLE_TOKENS

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
