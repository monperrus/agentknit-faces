#!/usr/bin/env python3
"""Run agentknit with Kimi K3 through the official Kimi Coding Plan API.

The Coding Plan is distinct from Kimi's pay-as-you-go Open Platform.
Endpoint, key source and K3's quirks (it only accepts ``temperature=1``) come
from the inference-db entry ``kimi-coding``.

Usage:
    agentknit-kimi-k3 "<task>"           # one-shot
    agentknit-kimi-k3                    # interactive REPL
    agentknit-kimi-k3 --session <id>     # resume a previous session
    agentknit-kimi-k3 --non-interactive  # disable ask_user_question
    echo "<task>" | agentknit-kimi-k3    # task on stdin
"""

import os
import sys

import agentknit


MODEL = "k3"
INFERENCE_DB = "kimi-coding"

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
    schema["display_name"] = "Kimi K3 (Kimi Coding Plan)"
    # Context window measured by llmprobe (reports/k3).
    schema["context_window"] = 1048576
    # The compaction trigger must sit well below the window: the last
    # observed prompt (trigger) + one more assistant reply + one tool result
    # must still fit.  75 % leaves ~260k of headroom for the last turn.
    schema["compaction_trigger_tokens"] = 786432

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
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 4096 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=4096,
        # Kimi's implicit prefix cache carries a 5-minute TTL by default
        # (documented on the context-caching page; `prompt_cache_options`
        # can raise it to 1h but is a billing-affecting write control we
        # don't exercise).  Declaring it makes a resume after >5 min a
        # "cold resume" (expected full re-write) instead of a spurious
        # cache_proof_missing warning.
        cache_ttl_seconds=300,
    )
    if task:
        agentknit.run_task(schema, task, **common)
    elif not sys.stdin.isatty():
        task = sys.stdin.read().strip()
        if task:
            agentknit.run_task(schema, task, **common)
    else:
        agentknit.run_repl(schema, **common)


if __name__ == "__main__":
    main()
