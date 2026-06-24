#!/usr/bin/env python3
"""Abstract async agent loop — parametrised by model.

Callers must set os.environ["AGENTKNIT_RESUME_COMMAND"] to their own path
before importing this module, then call main(model).

Usage (direct):
    agent-deepseek-v4-flash-async --model <model-url> "<task>"
    agent-deepseek-v4-flash-async --model <model-url>       # REPL
"""

from __future__ import annotations

import atexit
import os
import sys

AGENT_PROBE_DIR = "/home/martin/workspace/prototypes/probe-model-tools"
if AGENT_PROBE_DIR not in sys.path:
    sys.path.insert(0, AGENT_PROBE_DIR)

import queue

from agentknit import (
    ASYNC_FAST_THRESHOLD_S,
    ASYNC_INLINE_MAX_BYTES,
    Tool,
    async_completion_queue,
    build_tool_spec,
    register_tools_in_library,
    t_execute_async,
)
from agentknit._core import (
    BOLD,
    DEFAULT_ENDPOINT,
    DIM,
    RESET,
    RL_BOLD,
    RL_RESET,
    _build_resume_cmd,
    _save_messages_snapshot,
    create_client,
    init_session,
    print_session_history,
    run_turn,
)
from agentknit.tool_library import (
    TOOL_LIBRARY,
    _async_last_lines,
    _async_try_inline,
    t_read,
    t_update,
    t_write,
)

_TOOLS = [
    Tool(
        "execute_shell_command",
        f"Start a shell command asynchronously. Returns tool_exec_id, cwd "
        f"(working directory), and local file paths for stdin (FIFO), stdout, "
        f"and stderr. Write to stdin_localfile to send input to the running "
        f"process. Optional `when` (integer minutes, default 0) delays the "
        f"start. If the command finishes within {int(ASYNC_FAST_THRESHOLD_S * 1000)} ms "
        f"and both outputs are under {ASYNC_INLINE_MAX_BYTES} bytes, stdout/stderr "
        f"are inlined immediately. When a background command finishes you will "
        f"be notified automatically with its output. In the meantime you can "
        f"read the stdout and stderr files (at the paths returned by this tool) "
        f"to see partial output while the command is still running.",
        t_execute_async,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "when": {
                    "type": "integer",
                    "description": "Minutes to wait before starting the command (default 0).",
                },
            },
            "required": ["command"],
        },
    ),
    Tool(
        "read_file",
        "Read the contents of a local file.",
        t_read,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
            },
            "required": ["path"],
        },
    ),
    Tool(
        "write_file",
        "Write content to a local file, creating parent directories as needed.",
        t_write,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    ),
    Tool(
        "edit",
        "Edit a file by replacing an existing substring with new content. "
        "Use this instead of write_file when you need to make a surgical change "
        "to an existing file. Provide the exact existing text (old_str) and "
        "the replacement text (new_str).",
        t_update,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_str": {
                    "type": "string",
                    "description": "The exact existing substring to replace.",
                },
                "new_str": {"type": "string", "description": "The replacement text."},
            },
            "required": ["path", "old_str", "new_str"],
        },
        param_map={"path": "path", "old_str": "old", "new_str": "new"},
    ),
]

_TOOL_SCHEMA, _TOOL_DISPATCH = build_tool_spec(_TOOLS)
register_tools_in_library(_TOOLS)

_SYSTEM_SUPPLEMENT = (
    "You are a fully asynchronous coding agent. Think asynchronously. Start shell commands with execute_shell_command — "
    "they run in the background.."
)


def _build_schema(model: str, endpoint: str) -> dict:
    return {
        "model": model,
        "endpoint": endpoint,
        "status": "default",
        "inferred_tool_schema": _TOOL_SCHEMA,
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
        "tool_dispatch": _TOOL_DISPATCH,
    }


def _completion_message(event: dict) -> str:
    """Format a completion event into a natural-language notification for the LLM."""
    exec_id = event["tool_exec_id"]
    rc = event["returncode"]
    duration = event["duration"]
    cwd = event.get("cwd", "")
    stdout_path = event["stdout_file"]
    stderr_path = event["stderr_file"]
    stdout = _async_try_inline(stdout_path)
    if stdout is None:
        tail = _async_last_lines(stdout_path, 3)
        stdout_text = f"(see {stdout_path})" + (
            f"\nlast 3 lines:\n{tail}" if tail else ""
        )
    else:
        stdout_text = stdout.rstrip() or "(empty)"
    stderr = _async_try_inline(stderr_path)
    if stderr is None:
        tail = _async_last_lines(stderr_path, 3)
        stderr_text = f"(see {stderr_path})" + (
            f"\nlast 3 lines:\n{tail}" if tail else ""
        )
    else:
        stderr_text = stderr.rstrip()
    parts = [
        f"Background command {exec_id} finished after {duration:.1f}s "
        f"(returncode={rc})."
    ]
    if cwd:
        parts.append(f"cwd: {cwd}")
    parts.append(f"stdout: {stdout_text}")
    if stderr_text.strip():
        parts.append(f"stderr: {stderr_text}")
    return "\n".join(parts)


def _drain_completion_queue() -> list[str]:
    """Return formatted messages for every completed async command."""
    msgs: list[str] = []
    while True:
        try:
            msgs.append(_completion_message(async_completion_queue.get_nowait()))
        except queue.Empty:
            break
    return msgs


def _repl(schema: dict, session_id: str | None, system_prompt_supplement: str) -> None:
    """Custom REPL that auto-triggers LLM turns when background commands finish."""
    import hashlib
    import readline  # noqa: F401 — enables arrow keys in input()
    import threading

    client = create_client(schema)
    session = init_session(
        schema,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
    )
    model = schema["model"]
    resume_cmd = _build_resume_cmd(model, session["session_id"])

    if session_id:
        print_session_history(session)

    from pathlib import Path

    _hist_dir = Path.home() / ".local/share/agent_probe/repl_history"
    _hist_dir.mkdir(parents=True, exist_ok=True)
    _hist_file = (
        _hist_dir / f"{hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:12]}.hist"
    )
    try:
        readline.read_history_file(_hist_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    print(f"{BOLD}async-agent {model}{RESET}  (type 'exit' to quit)\n")

    pending: list[str] = []

    _alert = threading.Event()
    _stop_monitor = threading.Event()

    def _monitor() -> None:
        while not _stop_monitor.is_set():
            try:
                async_completion_queue.get(timeout=0.25)
                _alert.set()
            except queue.Empty:
                pass

    _monitor_thread = threading.Thread(target=_monitor, daemon=True)
    _monitor_thread.start()
    atexit.register(lambda: _stop_monitor.set())

    def _run(task: str) -> None:
        try:
            run_turn(client, model, session, task)
        except KeyboardInterrupt:
            print(f"\n{DIM}[interrupted]{RESET}")
        finally:
            _save_messages_snapshot(session)

    def _check_alerts() -> None:
        if _alert.is_set():
            _alert.clear()
            pending.extend(_drain_completion_queue())

    try:
        while True:
            _check_alerts()

            if pending:
                task = pending.pop(0)
                print(f"{DIM}[completion] {task.splitlines()[0]}{RESET}")
                _run(task)
                continue

            try:
                user_input = input(f"{RL_BOLD}>{RL_RESET} ")
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break

            _check_alerts()
            if pending:
                pending.insert(0, user_input.strip())
                continue

            cmd = user_input.strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                break

            _run(user_input)

    finally:
        _stop_monitor.set()
        try:
            readline.write_history_file(_hist_file)
        except Exception:
            pass
        _save_messages_snapshot(session)
        print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


def _run_task_async(
    schema: dict,
    task: str,
    *,
    session_id: str | None,
    system_prompt_supplement: str,
    non_interactive: bool,
) -> None:
    """One-shot task runner that delivers async completion events back to the model.

    Unlike run_task(), this loops until all background commands (those that
    didn't finish within ASYNC_FAST_THRESHOLD_S) have completed and their
    events have been delivered as follow-up turns.
    """
    import json
    import threading

    _pending_count = 0
    _pending_lock = threading.Lock()

    _LIB_KEY = "_async_agent_counting_execute"

    def _counting_execute(command: str, when: int = 0) -> tuple[str, dict]:
        nonlocal _pending_count
        result_str, result_dict = t_execute_async(command, when=when)
        result = json.loads(result_str)
        # "completed" is only present when the command finished inline (fast
        # path); its absence means a background thread will push an event to
        # async_completion_queue.
        if not result.get("completed"):
            with _pending_lock:
                _pending_count += 1
        return result_str, result_dict

    # Register under a string name so agentknit's JSON logging can serialize
    # the dispatch entry (callable values aren't JSON-serializable).
    TOOL_LIBRARY[_LIB_KEY] = _counting_execute

    patched_schema = {
        **schema,
        "tool_dispatch": {
            **schema["tool_dispatch"],
            "execute_shell_command": {"python_function": _LIB_KEY, "param_map": {}},
        },
    }

    client = create_client(patched_schema)
    session = init_session(
        patched_schema,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
        non_interactive=non_interactive,
    )
    try:
        run_turn(client, patched_schema["model"], session, task)
        while True:
            with _pending_lock:
                if _pending_count == 0:
                    break
            try:
                event = async_completion_queue.get(timeout=300)
            except queue.Empty:
                break
            with _pending_lock:
                _pending_count -= 1
            msg = _completion_message(event)
            print(f"{DIM}[completion] {msg.splitlines()[0]}{RESET}")
            run_turn(client, patched_schema["model"], session, msg)
    finally:
        TOOL_LIBRARY.pop(_LIB_KEY, None)
        _save_messages_snapshot(session)


def main(model: str) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Async coding agent.")
    p.add_argument("task", nargs="*", help="One-shot task (omit for REPL)")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--session", metavar="SESSION_ID")
    p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable ask_user_question tool and REPL fallback",
    )
    args = p.parse_args()

    schema = _build_schema(model, args.endpoint)
    opts = dict(
        session_id=args.session,
        system_prompt_supplement=_SYSTEM_SUPPLEMENT,
        non_interactive=args.non_interactive,
    )
    if args.task:
        _run_task_async(schema, " ".join(args.task), **opts)
    else:
        _repl(schema, args.session, _SYSTEM_SUPPLEMENT)


if __name__ == "__main__":
    import argparse as _ap

    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--model", required=True)
    _known, _rest = _p.parse_known_args()
    os.environ.setdefault("AGENTKNIT_RESUME_COMMAND", os.path.realpath(__file__))
    sys.argv = [sys.argv[0]] + _rest
    main(_known.model)
