#!/usr/bin/env python3
"""Run agentknit with Kimi K3 through the official Kimi Coding Plan API.

The Coding Plan is distinct from Kimi's pay-as-you-go Open Platform.  Its
credential is read from the system keyring (service ``login2``, username
``kimi_api_key``), never from a source-controlled environment variable.

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
ENDPOINT = "https://api.kimi.com/coding/v1"

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


def _create_k3_client(schema):
    """Return a client that meets K3's fixed-temperature API requirement.

    K3's Coding Plan endpoint currently accepts only ``temperature=1``, while
    agentknit's generic agent loop intentionally sends ``temperature=0``.
    Keep that provider adaptation local to this launcher rather than changing
    the generic loop for every other provider.
    """
    client = agentknit.create_client(schema)
    create = client.chat.completions.create

    def create_with_k3_temperature(*args, **kwargs):
        kwargs["temperature"] = 1
        return create(*args, **kwargs)

    client.chat.completions.create = create_with_k3_temperature
    return client


def main() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    # Resume hints must re-invoke this launcher, not the generic agentknit CLI.
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    # Kimi Coding Plan keys are separate from Kimi Open Platform keys.
    # Agentknit resolves this pair directly with keyring, so no key is put
    # into the process environment or command line.
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

    task = " ".join(task_args) if task_args else None
    common = dict(
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
        client=_create_k3_client(schema),
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 1024 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=1024,
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
