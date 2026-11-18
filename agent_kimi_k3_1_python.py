#!/usr/bin/env python3
"""Run Kimi K3.1 Pro with only file-editing and Python tools.

The four available tools are ``read_file``, ``write_file``, ``str_replace``
and ``python``. Python code runs directly with the current interpreter; no
shell or background-command tool is exposed. Endpoint, credentials and Kimi's
``temperature=1`` requirement come from inference-db entry ``kimi-coding``.

Usage:
    agentknit-kimi-k3.1-python "<task>"           # one-shot
    agentknit-kimi-k3.1-python                    # interactive REPL
    agentknit-kimi-k3.1-python --session <id>     # resume a previous session
    agentknit-kimi-k3.1-python --non-interactive  # disable ask_user_question
    echo "<task>" | agentknit-kimi-k3.1-python    # task on stdin
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from typing import Any

import agentknit
from agentknit.tool_library import TOOL_LIBRARY

MODEL = "k3.1-pro"
INFERENCE_DB = "kimi-coding"
PYTHON_TIMEOUT_SECONDS = 120

_home = os.path.expanduser("~")
_cwd = os.getcwd()
_SUPPLEMENT = (
    "Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}.\n"
    "When creating git commits, use the author/committer email "
    "martin.monperrus@gnieh.org.\n"
    "Use the python tool for computation and program execution. No shell tool "
    "is available."
)


def t_python(code: str, timeout: int = PYTHON_TIMEOUT_SECONDS) -> tuple[str, dict[str, object]]:
    """Execute Python source and return its stdout, stderr and exit code as JSON."""
    try:
        timeout = max(1, min(int(timeout), 600))
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        result = {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"ERROR: Python timed out after {timeout} seconds.",
            "exit_code": None,
        }
    except Exception as exc:
        result = {"stdout": "", "stderr": f"ERROR: {type(exc).__name__}: {exc}", "exit_code": None}
    text = json.dumps(result)
    return text, {"result": result, "ok": result["exit_code"] == 0}


PYTHON_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python source with the current interpreter and return JSON "
            "containing stdout, stderr and exit_code. Code runs in the current "
            "working directory and is not sandboxed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (1–600; default 120).",
                },
            },
            "required": ["code"],
        },
    },
}


def configure_tools(schema: dict[str, Any]) -> None:
    """Replace the default shell tool with the Python-only tool set."""
    TOOL_LIBRARY["t_python"] = t_python
    file_tools = [
        tool for tool in schema["tool_specs"]
        if tool["function"]["name"] in {"read_file", "write_file", "str_replace"}
    ]
    schema["tool_specs"] = copy.deepcopy(file_tools) + [copy.deepcopy(PYTHON_TOOL_SPEC)]
    schema["tools"] = ["t_read", "t_write", "t_update", "t_python"]


def main() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "Kimi K3.1 Pro (Python tools)"
    schema["context_window"] = 1_048_576
    schema["compaction_trigger_tokens"] = 786_432
    configure_tools(schema)

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
        min_cacheable_tokens=4096,
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
