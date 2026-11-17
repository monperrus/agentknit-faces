#!/usr/bin/env python3
"""Run Kimi K3.1 Pro with one persistent Python REPL as its only tool.

The model reads, writes and edits files by writing Python itself. There are
no separate file, shell or background-command tools. Imports, variables,
functions and the working directory persist between calls; a timeout restarts
the interpreter. Execution is deliberately unsandboxed.

Endpoint, credentials and Kimi's ``temperature=1`` requirement come from
inference-db entry ``kimi-coding``.

Usage:
    agentknit-kimi-k3.1-python-repl "<task>"           # one-shot
    agentknit-kimi-k3.1-python-repl                    # interactive REPL
    agentknit-kimi-k3.1-python-repl --session <id>     # resume a session
    agentknit-kimi-k3.1-python-repl --non-interactive  # disable questions
    echo "<task>" | agentknit-kimi-k3.1-python-repl    # task on stdin
"""

from __future__ import annotations

import copy
import json
import os
import queue
import sys
import threading
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
    "Use the python tool for every action, including reading, writing and "
    "editing files. It is a persistent Python REPL: imports, variables, "
    "functions and working-directory changes survive between calls. No shell "
    "or separate file tools are available. Print the information you need to "
    "observe; expression results are displayed as in a normal Python REPL."
)


class _PythonExecutionError(Exception):
    """Internal signal that InteractiveConsole already displayed an error."""


class _PythonREPL:
    """Own a persistent interpreter that can be discarded after a timeout."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._execution_lock = threading.Lock()
        self._requests: queue.Queue[str | None] = queue.Queue()
        self._responses: queue.Queue[dict[str, object]] = queue.Queue()

    def _start(self) -> None:
        self._requests = queue.Queue()
        self._responses = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="python-repl")
        self._worker.start()

    @staticmethod
    def _show_traceback() -> None:
        """Keep InteractiveConsole's traceback while marking the call failed."""
        import traceback

        traceback.print_exception(*sys.exc_info())
        raise _PythonExecutionError from None

    def _run(self) -> None:
        import code
        import contextlib
        import io

        console = code.InteractiveConsole()
        console.showtraceback = self._show_traceback
        while True:
            source = self._requests.get()
            if source is None:
                return
            stdout = io.StringIO()
            stderr = io.StringIO()
            exit_code = 0
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    incomplete = console.runsource(source, symbol="single")
                    if incomplete:
                        print("SyntaxError: incomplete Python input", file=stderr)
                        exit_code = 1
                except _PythonExecutionError:
                    exit_code = 1
                except SystemExit as exc:
                    exit_code = exc.code if isinstance(exc.code, int) else 1
                    if exc.code not in (0, None) and not isinstance(exc.code, int):
                        print(exc.code, file=stderr)
                except BaseException:
                    import traceback

                    traceback.print_exc()
                    exit_code = 1
            self._responses.put({
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "exit_code": exit_code,
            })

    def execute(self, source: str, timeout: int) -> dict[str, object]:
        # Serialize execution as well as replacement, so an expired request
        # can never receive the next request's response from an old queue.
        with self._execution_lock:
            with self._lock:
                if self._worker is None:
                    self._start()
                self._requests.put(source)
            try:
                return self._responses.get(timeout=timeout)
            except queue.Empty:
                # A Python thread cannot be killed safely. Leave the expired
                # daemon behind and start clean; it cannot affect the new REPL.
                with self._lock:
                    self._worker = None
                return {
                    "stdout": "",
                    "stderr": (
                        f"ERROR: Python timed out after {timeout} seconds; "
                        "the persistent interpreter was restarted."
                    ),
                    "exit_code": None,
                }

    def reset(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._requests.put(None)
                self._worker.join(timeout=1)
            self._worker = None


_repl = _PythonREPL()


def _reset_interpreter() -> None:
    """Release interpreter resources; used by tests and embedding callers."""
    _repl.reset()


def t_python(code: str, timeout: int = PYTHON_TIMEOUT_SECONDS) -> tuple[str, dict[str, object]]:
    """Execute Python in the persistent REPL and return its captured output."""
    try:
        timeout = max(1, min(int(timeout), 600))
        result = _repl.execute(code, timeout)
    except Exception as exc:
        result = {"stdout": "", "stderr": f"ERROR: {type(exc).__name__}: {exc}", "exit_code": None}
    text = json.dumps(result)
    return text, {"result": result, "ok": result["exit_code"] == 0}


PYTHON_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python source in a persistent REPL. This is the only tool: "
            "use it for computation and program execution, and to read, write "
            "and edit files (for example with pathlib and string replacement). "
            "Imports, variables, functions and working-directory changes persist "
            "between calls. Expression results are displayed; print other output "
            "you need to observe. Returns JSON containing stdout, stderr and "
            "exit_code. Code runs in the current working directory and is not "
            "sandboxed. A timeout restarts the interpreter and clears its state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python source to execute, including any file "
                        "read/write/edit operations."
                    ),
                },
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
    """Replace every model-facing tool with the persistent Python REPL."""
    TOOL_LIBRARY["t_python"] = t_python
    schema["tool_specs"] = [copy.deepcopy(PYTHON_TOOL_SPEC)]
    schema["tools"] = ["t_python"]


def main() -> None:
    """Dispatch a one-shot task, stdin task, or interactive agent REPL."""
    os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

    schema = agentknit.load_specification(MODEL, inference_db=INFERENCE_DB)
    schema["display_name"] = "Kimi K3.1 Pro (Python REPL only)"
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
    try:
        if task:
            agentknit.run_task(schema, task, **common)
        elif not sys.stdin.isatty():
            task = sys.stdin.read().strip()
            if task:
                agentknit.run_task(schema, task, **common)
        else:
            agentknit.run_repl(schema, **common)
    finally:
        _reset_interpreter()


if __name__ == "__main__":
    main()
