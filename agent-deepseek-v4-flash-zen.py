#!/usr/bin/env python3
"""Async wrapper — runs DeepSeek V4 Flash (free) via opencode.ai subprocess
with non-blocking shell execution.

Uses a stable per-directory session ID so each run from the same working
directory resumes the same conversation, letting the API reuse its
prefix cache.

Usage:
    agent-deepseek-v4-flash-zen "<task>"           # one-shot
    agent-deepseek-v4-flash-zen                    # interactive REPL
    agent-deepseek-v4-flash-zen --session <id>     # explicit session override
"""
from __future__ import annotations

import os
import sys

MODEL = "run:///home/martin/bin/opencode-free-deepseek-v4-flash-completions.py"
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

# Ensure agentknit is on sys.path
AGENT_PROBE_DIR = "/home/martin/workspace/prototypes/probe-model-tools"
if AGENT_PROBE_DIR not in sys.path:
    sys.path.insert(0, AGENT_PROBE_DIR)

from agentknit import Tool, build_tool_spec, register_tools_in_library
from agentknit import (
    t_execute_async, t_query_exec,
    ASYNC_FAST_THRESHOLD_S, ASYNC_INLINE_MAX_BYTES,
    async_completion_queue,
    run_task,
)
from agentknit.tool_library import t_read, t_write, t_update, _async_try_inline, _async_last_lines
import queue
import select

from agentknit._core import (
    create_client, init_session, run_turn, _save_messages_snapshot,
    DEFAULT_ENDPOINT,
    BOLD, DIM, RESET, RL_BOLD, RL_RESET,
    print_session_history, _build_resume_cmd,
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
        f"are inlined immediately. When a background command finishes the agent "
        f"is notified automatically.",
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
        "query_tool_exec",
        f"Poll a command started with execute_shell_command. When completed, "
        f"includes returncode, cwd (working directory), and inlines stdout/stderr "
        f"if both are under {ASYNC_INLINE_MAX_BYTES} bytes; otherwise reports "
        f"file sizes.",
        t_query_exec,
        parameters={
            "type": "object",
            "properties": {
                "tool_exec_id": {
                    "type": "string",
                    "description": "The ID returned by execute_shell_command.",
                },
            },
            "required": ["tool_exec_id"],
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
                "old_str": {"type": "string", "description": "The exact existing substring to replace."},
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
    "You are a coding agent. Start shell commands with execute_shell_command — "
    "they run in the background. Use query_tool_exec to poll status. Pass `when` "
    "to execute_shell_command to delay a command by N minutes. When a background "
    "command finishes you will be notified automatically with its output. "
    "Both execute_shell_command and query_tool_exec return the cwd (working "
    "directory) in the result."
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
    exec_id  = event["tool_exec_id"]
    rc       = event["returncode"]
    duration = event["duration"]
    cwd      = event.get("cwd", "")
    stdout_path = event["stdout_file"]
    stderr_path = event["stderr_file"]
    stdout = _async_try_inline(stdout_path)
    if stdout is None:
        tail = _async_last_lines(stdout_path, 3)
        stdout_text = f"(see {stdout_path})" + (f"\nlast 3 lines:\n{tail}" if tail else "")
    else:
        stdout_text = stdout.rstrip() or "(empty)"
    stderr = _async_try_inline(stderr_path)
    if stderr is None:
        tail = _async_last_lines(stderr_path, 3)
        stderr_text = f"(see {stderr_path})" + (f"\nlast 3 lines:\n{tail}" if tail else "")
    else:
        stderr_text = stderr.rstrip()
    parts = [f"Background command {exec_id} finished after {duration:.1f}s "
             f"(returncode={rc})."]
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
    import readline  # noqa: F401 — enables arrow keys in input()
    import hashlib

    client  = create_client(schema)
    session = init_session(
        schema,
        resumed_from=session_id,
        system_prompt_supplement=system_prompt_supplement,
    )
    model      = schema["model"]
    resume_cmd = _build_resume_cmd(model, session["session_id"])

    if session_id:
        print_session_history(session)

    # Per-directory readline history
    from pathlib import Path
    _hist_dir  = Path.home() / ".local/share/agent_probe/repl_history"
    _hist_dir.mkdir(parents=True, exist_ok=True)
    _hist_file = _hist_dir / f"{hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:12]}.hist"
    try:
        readline.read_history_file(_hist_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    print(f"{BOLD}async-agent {model}{RESET}  (type 'exit' to quit)\n")

    _POLL_INTERVAL = 0.25   # seconds between completion-queue checks at the prompt

    pending: list[str] = []

    def _run(task: str) -> None:
        try:
            run_turn(client, model, session, task)
        except KeyboardInterrupt:
            print(f"\n{DIM}[interrupted]{RESET}")
        finally:
            _save_messages_snapshot(session)

    try:
        while True:
            # Drain any completions that arrived during the last turn.
            for msg in _drain_completion_queue():
                pending.append(msg)

            if pending:
                task = pending.pop(0)
                print(f"{DIM}[completion] {task.splitlines()[0]}{RESET}")
                _run(task)
                continue

            # Wait for user input, polling the completion queue periodically.
            sys.stdout.write(f"{RL_BOLD}>{RL_RESET} ")
            sys.stdout.flush()
            user_task: str | None = None
            try:
                while user_task is None:
                    ready, _, _ = select.select([sys.stdin], [], [], _POLL_INTERVAL)
                    if ready:
                        line = sys.stdin.readline()
                        if not line:   # EOF
                            raise EOFError
                        user_task = line.rstrip("\n")
                    else:
                        for msg in _drain_completion_queue():
                            pending.append(msg)
                        if pending and not readline.get_line_buffer():
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            break   # leave user_task = None, loop will handle pending
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break

            if user_task is None:
                continue   # completion arrived while waiting; loop handles it

            cmd = user_task.strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                break

            _run(user_task)

    finally:
        try:
            readline.write_history_file(_hist_file)
        except Exception:
            pass
        _save_messages_snapshot(session)
        print(f"\n{DIM}Resume: {resume_cmd}{RESET}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Async DeepSeek V4 Flash coding agent.")
    p.add_argument("task", nargs="*", help="One-shot task (omit for REPL)")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--session", metavar="SESSION_ID")
    p.add_argument("--non-interactive", action="store_true",
                   help="Disable ask_user_question tool and REPL fallback")
    args = p.parse_args()

    schema = _build_schema(MODEL, args.endpoint)
    opts = dict(
        session_id=args.session,
        system_prompt_supplement=_SYSTEM_SUPPLEMENT,
        non_interactive=args.non_interactive,
    )
    if args.task:
        run_task(schema, " ".join(args.task), **opts)
    else:
        _repl(schema, args.session, _SYSTEM_SUPPLEMENT)


if __name__ == "__main__":
    main()
