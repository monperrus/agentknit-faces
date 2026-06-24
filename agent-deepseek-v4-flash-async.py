#!/usr/bin/env python3
"""Abstract async agent loop — parametrised by model.

Callers must set os.environ["AGENTKNIT_RESUME_COMMAND"] to their own path
before importing this module, then call main(model).

Usage (direct):
    agent-deepseek-v4-flash-async --model <model-url> "<task>"
    agent-deepseek-v4-flash-async --model <model-url>       # REPL
"""

from __future__ import annotations

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

# Maps tool_exec_id → command string, populated by _t_execute_async_tracking.
_exec_commands: dict[str, str] = {}
_CMD_MAX = 60


def _t_execute_async_tracking(command: str, when: int = 0) -> tuple[str, dict]:
    import json
    result_str, result_dict = t_execute_async(command, when=when)
    exec_id = json.loads(result_str).get("tool_exec_id")
    if exec_id:
        _exec_commands[exec_id] = command
    return result_str, result_dict


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
        _t_execute_async_tracking,
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
    cmd = _exec_commands.pop(exec_id, None)
    cmd_repr = ""
    if cmd:
        truncated = cmd if len(cmd) <= _CMD_MAX else cmd[:_CMD_MAX] + "…"
        cmd_repr = f" `{truncated}`"
    parts = [
        f"Background command{cmd_repr} ({exec_id}) finished after {duration:.1f}s "
        f"(returncode={rc})."
    ]
    if cwd:
        parts.append(f"cwd: {cwd}")
    parts.append(f"stdout: {stdout_text}")
    if stderr_text.strip():
        parts.append(f"stderr: {stderr_text}")
    return "\n".join(parts)



def _repl(schema: dict, session_id: str | None, system_prompt_supplement: str) -> None:
    """Custom REPL that auto-triggers LLM turns when background commands finish.

    Uses the "input thread" pattern: input() runs in a daemon thread so it
    never needs to be interrupted externally.  Both user input and async
    completion events are routed through a shared event_queue; the main thread
    processes whichever arrives first.
    """
    import hashlib
    import readline  # noqa: F401 — enables arrow keys / history in input()
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

    # ("user", line) | ("completion", msg) | ("eof", None)
    event_queue: queue.Queue = queue.Queue()

    _stop = threading.Event()

    def _input_reader() -> None:
        while not _stop.is_set():
            try:
                line = input(f"{RL_BOLD}>{RL_RESET} ")
                event_queue.put(("user", line))
            except EOFError:
                event_queue.put(("eof", None))
                return
            except KeyboardInterrupt:
                print()  # newline after ^C; loop restarts → new prompt

    def _completion_watcher() -> None:
        while not _stop.is_set():
            try:
                event = async_completion_queue.get(timeout=0.25)
                event_queue.put(("completion", event))
            except queue.Empty:
                pass

    threading.Thread(target=_input_reader, daemon=True).start()
    threading.Thread(target=_completion_watcher, daemon=True).start()

    def _run(task: str) -> None:
        try:
            run_turn(client, model, session, task)
        except KeyboardInterrupt:
            print(f"\n{DIM}[interrupted]{RESET}")
        finally:
            _save_messages_snapshot(session)

    try:
        while True:
            try:
                kind, data = event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if kind == "eof":
                break
            elif kind == "user":
                cmd = data.strip()
                if not cmd:
                    continue
                if cmd.lower() in ("exit", "quit", "q"):
                    break
                _run(data)
            elif kind == "completion":
                msg = _completion_message(data)
                # Print on a fresh line so we don't clobber the user's current input.
                sys.stdout.write(f"\n{DIM}[completion] {msg.splitlines()[0]}{RESET}\n")
                sys.stdout.flush()
                _run(msg)

    finally:
        _stop.set()
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
