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

Every session is maintained as a valid Jupyter notebook. New sessions use a
``.ipynb`` file beside the JSONL transcript; resumed sessions use
``<session_id>.ipynb`` beside agentknit's snapshot. Prompts and assistant
replies become Markdown cells; each Python call becomes a code cell with its
stdout/stderr. Use ``--notebook`` to print the path at startup, or
``--notebook <path>`` for an explicit location.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import re
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


def _splitlines(text: str) -> list[str]:
    """Notebook multiline strings keep line endings and never drop newlines."""
    return text.splitlines(keepends=True)


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _splitlines(source)}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _splitlines(source),
    }


class _NotebookSink:
    """Atomically mirror agentknit's durable event stream as a notebook."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *,
                 requested: bool = False, replay: list[dict[str, Any]] | None = None,
                 model: str = MODEL) -> None:
        self.path = os.fspath(path) if path is not None else None
        self.requested = requested
        self._model = model
        self._session_id: str | None = None
        self._cells: list[dict[str, Any]] = [_markdown(
            "# Agentknit coding session\n\n"
            "Automatically maintained from the live Python REPL session."
        )]
        self._execution_count = 0
        self._calls: dict[str, dict[str, Any]] = {}
        self._turn_open = False
        self._turn_needs_separator = False
        self._assistant_text: str | None = None
        self._metadata: dict[str, Any] = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "agentknit": {"session_id": None},
        }
        for record in replay or []:
            self._replay(record)
        if self.path is not None:
            self._write()
        self._bound = self.path is not None

    def _default_path(self, transcript_path: str | None, session_id: str) -> None:
        if not self._bound and transcript_path:
            self.path = os.path.splitext(transcript_path)[0] + ".ipynb"
            self._bound = True
            self._metadata["agentknit"]["session_id"] = session_id
            if self.requested:
                print(f"Notebook: {self.path}", flush=True)
            self._write()

    def _write(self) -> None:
        if self.path is None:
            return
        notebook = {"cells": self._cells, "metadata": self._metadata,
                    "nbformat": 4, "nbformat_minor": 5}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        temporary = f"{self.path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def _append_markdown(self, source: str) -> None:
        if not source:
            return
        if self._cells[-1]["cell_type"] == "markdown":
            previous = "".join(self._cells[-1]["source"])
            self._cells[-1]["source"] = _splitlines(previous + "\n\n" + source)
        else:
            self._cells.append(_markdown(source))

    def _start_turn(self, task: str) -> None:
        if self._turn_open:
            self._turn_end()
        self._turn_open = True
        self._append_markdown(f"## User\n\n{task}")
        self._turn_needs_separator = False
        # A tool-only turn needs no prose; its code cell follows directly.

    def _turn_end(self) -> None:
        if self._turn_open:
            if self._turn_needs_separator:
                self._cells.append(_code(""))
            self._turn_open = False

    def _tool_start(self, record: dict[str, Any]) -> None:
        args = record.get("args") or {}
        code = args.get("code", "") if isinstance(args, dict) else ""
        cell = _code(str(code))
        cell["metadata"] = {"agentknit": {
            "call_id": record.get("call_id"), "timeout": args.get("timeout")
            if isinstance(args, dict) else None}}
        self._cells.append(cell)
        self._turn_needs_separator = True
        if record.get("call_id"):
            self._calls[str(record["call_id"])] = cell

    def _tool_end(self, record: dict[str, Any]) -> None:
        cell = self._calls.get(str(record.get("call_id")))
        if cell is None or cell["execution_count"] is not None:
            return
        result: Any = record.get("result", "")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"stdout": "", "stderr": result, "exit_code": None}
        if not isinstance(result, dict):
            result = {"stdout": "", "stderr": str(result), "exit_code": None}
        outputs = []
        # Jupyter permits empty streams; retaining both makes the REPL result exact.
        for stream in ("stdout", "stderr"):
            outputs.append({"output_type": "stream", "name": stream,
                            "text": _splitlines(str(result.get(stream, "")))})
        cell["outputs"] = outputs
        cell["metadata"]["agentknit"]["exit_code"] = result.get("exit_code")
        self._execution_count += 1
        cell["execution_count"] = self._execution_count

    def _message(self, message: dict[str, Any]) -> None:
        role = message.get("role")
        if role == "user":
            content = str(message.get("content", ""))
            content = re.sub(r"\n\n<session_time>.*?</session_time>$", "", content,
                             flags=re.DOTALL)
            self._start_turn(content)
        elif role == "assistant":
            content = message.get("content")
            if content:
                self._assistant_text = str(content)
                self._append_markdown(f"## Assistant\n\n{content}")
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    args = json.loads(function.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {"code": function.get("arguments") or ""}
                self._tool_start({"call_id": call.get("id"),
                                  "name": function.get("name"), "args": args})
        elif role == "tool":
            self._tool_end({"call_id": message.get("tool_call_id"),
                            "result": message.get("content", "")})

    def _replay(self, record: dict[str, Any]) -> None:
        """Rebuild cells from a resumed message history without duplicating calls."""
        if record.get("type") == "message":
            self._message(record.get("msg") or {})

    def append(self, record: dict[str, Any]) -> None:
        kind = record.get("type")
        if kind == "message" and self.path is None:
            message = record.get("msg") or {}
            if message.get("role") == "system":
                match = re.search(r"Session: ([^\s]+)", str(message.get("content", "")))
                if match:
                    self._session_id = match.group(1)
                    self._bind_from_session_id(self._session_id)
        if kind == "message":
            self._message(record.get("msg") or {})
        elif kind == "tool_start" and record.get("name") == "t_python":
            self._tool_start(record)
        elif kind == "tool_end":
            self._tool_end(record)
        elif kind == "turn_end":
            self._turn_end()
        if kind == "event" and record.get("event_type") == "session_resumed":
            self._bind_from_session_id(str(record.get("session_id", "")))
        elif kind == "event" and record.get("event_type") == "session_time":
            self._bind_from_log()
        elif kind == "event" and record.get("event_type") == "final_answer":
            text = str(record.get("data", {}).get("text") or "")
            if text and text != self._assistant_text:
                self._append_markdown(f"## Assistant\n\n{text}")
                self._assistant_text = text
        if self.path is not None:
            self._write()

    def bind(self, transcript_path: str | None, session_id: str) -> None:
        """Bind a sink to agentknit's transcript-derived notebook location."""
        self._default_path(transcript_path, session_id)

    def _bind_from_log(self) -> None:
        """Compatibility fallback for transcripts that expose only an id."""
        if self._session_id:
            self._bind_from_session_id(self._session_id)

    def _bind_from_session_id(self, session_id: str) -> None:
        """The durable stream omits log_path; derive its canonical location."""
        self._session_id = session_id
        if not session_id:
            return
        from agentknit._core import LOG_BASE, safe_model_name
        candidates = list((LOG_BASE / safe_model_name(self._model)).glob(
            f"*/*_{session_id}.jsonl"))
        if not candidates and not self._bound:
            # The first prompt creates the log after session initialization.
            # Its name contains the id and today's local date.
            import datetime
            folder = LOG_BASE / safe_model_name(self._model) / datetime.datetime.now().strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            transcript = folder / f"{datetime.datetime.now().strftime('%H%M%S')}_{session_id}.jsonl"
            self._default_path(str(transcript), session_id)
            return
        if candidates:
            self._default_path(str(max(candidates, key=lambda p: p.stat().st_mtime)),
                               session_id)

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


def _notebook_arg(arguments: list[str]) -> str | bool | None:
    """Return no override, an explicit path, or True for the default path."""
    if "--notebook" not in arguments:
        return None
    index = arguments.index("--notebook")
    if (index + 1 < len(arguments) and not arguments[index + 1].startswith("--")
            and arguments[index + 1].endswith(".ipynb")):
        return arguments[index + 1]
    return True


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

    notebook = _notebook_arg(sys.argv[1:])
    replay: list[dict[str, Any]] | None = None
    if session_id:
        from agentknit._core import _load_messages_snapshot
        messages = _load_messages_snapshot(MODEL, session_id) or []
        replay = [{"type": "message", "msg": message} for message in messages]
    notebook_sink = _NotebookSink(notebook if isinstance(notebook, str) else None,
                                  requested=notebook is True, replay=replay)
    if session_id and not isinstance(notebook, str):
        from agentknit._core import _snapshot_path
        snapshot = _snapshot_path(MODEL, session_id)
        # Resume opens a new transcript, but the notebook remains the
        # single canonical document for the whole session id.
        notebook_sink.bind(os.path.join(os.path.dirname(str(snapshot)), session_id + ".ipynb"),
                           session_id)

    flags_with_value = {"--session"}
    flags_with_optional_value = {"--notebook"}
    task_args: list[str] = []
    skip_next = False
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if skip_next:
            skip_next = False
            continue
        if argument in flags_with_value:
            skip_next = True
            continue
        if argument in flags_with_optional_value:
            if (index + 1 < len(arguments)
                    and not arguments[index + 1].startswith("--")
                    and arguments[index + 1].endswith(".ipynb")):
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
        durable_sink=notebook_sink,
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
