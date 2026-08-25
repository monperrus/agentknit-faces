#!/usr/bin/env python3
"""Agent wrapper for glm-5.3 via api.z.ai.

glm-5.3 is z.ai's flagship model, on the same coding endpoint as glm-5.2.
API key retrieved from keyring (service: z.ai, username: api_key).

Usage:
    agent-glm-5.3 "<task>"           # one-shot
    agent-glm-5.3                    # interactive REPL
    agent-glm-5.3 --non-interactive  # disable ask_user_question tool
"""

import os
import sys

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)

import agentknit

MODEL    = "glm-5.3"
ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

# Load spec and inject keyring config so agentknit resolves the API key
# via keyring directly, without any shared env-var convention.
schema = agentknit.load_specification(MODEL, ENDPOINT)
schema["keyring_service"]  = "z.ai"
schema["keyring_username"] = "api_key"

# Async shell tools ("nohup" / "nohup_query") backed by agentknit's
# t_execute_async / t_query_exec from the tool library. "nohup" wraps the
# command in `timeout(1)` to bound execution (default 10 minutes).
from agentknit import ASYNC_FAST_THRESHOLD_S, ASYNC_INLINE_MAX_BYTES
from agentknit import t_execute_async as _t_execute_async

_NOHUP_TIMEOUT_MIN = 10


def _nohup(command: str, timeout: int = _NOHUP_TIMEOUT_MIN) -> tuple[str, dict[str, object]]:
    """Bound the command with timeout(1) then hand off to t_execute_async."""
    return _t_execute_async(f"timeout {int(timeout) * 60} {command}")


_NOHUP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "nohup",
            "description": (
                "Start a shell command asynchronously, like nohup(1). Returns "
                "tool_exec_id and local file paths for stdin (FIFO), stdout, "
                "and stderr. Write to stdin_localfile to send input to the "
                "running process. Execution is bounded: the command is killed "
                f"after `timeout` minutes (default {_NOHUP_TIMEOUT_MIN}). If "
                f"the command finishes within {int(ASYNC_FAST_THRESHOLD_S * 1000)} ms "
                f"and both outputs are under {ASYNC_INLINE_MAX_BYTES} bytes, "
                "stdout/stderr are inlined immediately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum minutes the command may run before being "
                            f"killed (default {_NOHUP_TIMEOUT_MIN})."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nohup_query",
            "description": (
                "Poll a command started with nohup. When completed, includes "
                "returncode and inlines stdout/stderr if both are under "
                f"{ASYNC_INLINE_MAX_BYTES} bytes; otherwise reports file sizes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_exec_id": {"type": "string", "description": "The tool_exec_id returned by nohup."},
                },
                "required": ["tool_exec_id"],
            },
        },
    },
]

schema["tool_specs"] = list(schema.get("tool_specs") or schema.get("inferred_tool_schema") or [])
schema["tool_specs"].extend(_NOHUP_TOOLS)
schema["inferred_tool_schema"] = schema["tool_specs"]
# Register the local _nohup wrapper (t_query_exec lives in TOOL_LIBRARY
# already) and route the dispatch entries through it.
from agentknit.tool_library import TOOL_LIBRARY

TOOL_LIBRARY["_nohup"] = _nohup
if "tools" in schema:
    schema["tools"] = list(schema["tools"]) + ["_nohup", "t_query_exec"]
else:
    schema.setdefault("tool_dispatch", {})
    schema["tool_dispatch"].update({
        "nohup":       {"python_function": "_nohup",      "param_map": {}},
        "nohup_query": {"python_function": "t_query_exec", "param_map": {}},
    })

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

# Parse simple flags we care about.
_non_interactive = "--non-interactive" in sys.argv
_session_id = None
if "--session" in sys.argv:
    idx = sys.argv.index("--session")
    if idx + 1 < len(sys.argv):
        _session_id = sys.argv[idx + 1]

# Collect task from CLI args or stdin.
# Skip known flags (--non-interactive is boolean; --session takes a value) so
# the session-id value is never mistaken for a task argument.
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

if _task:
    agentknit.run_task(
        schema,
        _task,
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
    )
elif not sys.stdin.isatty():
    _task = sys.stdin.read().strip()
    if _task:
        agentknit.run_task(
            schema,
            _task,
            non_interactive=_non_interactive,
            session_id=_session_id,
            system_prompt_supplement=_SUPPLEMENT,
        )
else:
    agentknit.run_repl(
        schema,
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
    )
