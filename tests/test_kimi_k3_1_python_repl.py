"""Kimi K3.1 Python REPL face: one persistent tool for every action."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FACE = REPO_ROOT / "agent-kimi-k3.1-python-repl.py"


def load_face():
    spec = importlib.util.spec_from_file_location("face_kimi_k3_1_python_repl", FACE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def face(monkeypatch: pytest.MonkeyPatch):
    module = load_face()
    monkeypatch.setattr(module.agentknit, "run_repl", lambda *args, **kwargs: None)
    yield module
    module._reset_interpreter()


def capture_task(face, monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}
    monkeypatch.setattr(face.agentknit, "run_task", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs))
    return captured


def test_face_exposes_only_one_python_tool_for_all_file_operations(face, monkeypatch):
    captured = capture_task(face, monkeypatch)
    monkeypatch.setattr(sys, "argv", [FACE.name, "--non-interactive", "say", "ok"])

    face.main()

    schema = captured["args"][0]
    assert captured["args"][1] == "say ok"
    assert schema["model"] == "k3.1-pro"
    assert schema["inference_db"] == "kimi-coding"
    assert schema["tools"] == ["t_python"]
    assert len(schema["tool_specs"]) == 1
    tool = schema["tool_specs"][0]["function"]
    assert tool["name"] == "python"
    description = tool["description"].lower()
    assert "persistent" in description
    assert "read" in description
    assert "write" in description
    assert "edit" in description
    assert "files" in description
    assert captured["kwargs"]["min_cacheable_tokens"] == 4096
    assert captured["kwargs"]["cache_ttl_seconds"] == 300


def test_stdin_and_session_are_forwarded(face, monkeypatch):
    captured = capture_task(face, monkeypatch)
    monkeypatch.setattr(sys, "argv", [FACE.name, "--session", "session-123"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "task from stdin\n")

    face.main()

    assert captured["args"][1] == "task from stdin"
    assert captured["kwargs"]["session_id"] == "session-123"


def test_python_tool_reads_writes_and_edits_files_without_file_tools(face, tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("before\n")

    read = face.t_python(f"from pathlib import Path; print(Path({str(path)!r}).read_text())")
    assert json.loads(read[0])["stdout"] == "before\n\n"

    write = face.t_python(f"_ = Path({str(path)!r}).write_text('after\\n')")
    assert json.loads(write[0])["exit_code"] == 0

    edit = face.t_python(
        f"p = Path({str(path)!r}); _ = p.write_text(p.read_text().replace('after', 'edited'));"
        "print(p.read_text(), end='')"
    )
    result = json.loads(edit[0])
    assert result == {"stdout": "edited\n", "stderr": "", "exit_code": 0}
    assert path.read_text() == "edited\n"


def test_python_tool_preserves_state_and_displays_expressions(face):
    face.t_python("import math; answer = 6 * 7")

    text, metadata = face.t_python("answer")

    assert json.loads(text) == {"stdout": "42\n", "stderr": "", "exit_code": 0}
    assert metadata["ok"] is True


def test_python_tool_captures_errors_without_losing_state(face):
    face.t_python("kept = 'yes'")

    error, metadata = face.t_python("1 / 0")
    result = json.loads(error)
    assert result["exit_code"] == 1
    assert "ZeroDivisionError" in result["stderr"]
    assert metadata["ok"] is False

    recovered, _ = face.t_python("print(kept)")
    assert json.loads(recovered)["stdout"] == "yes\n"


def test_python_tool_timeout_restarts_a_clean_interpreter(face):
    face.t_python("discarded = True")

    text, metadata = face.t_python("import time; time.sleep(2)", timeout=1)
    result = json.loads(text)
    assert result["exit_code"] is None
    assert "timed out" in result["stderr"]
    assert metadata["ok"] is False

    restarted, _ = face.t_python("print('discarded' in globals())")
    assert json.loads(restarted)["stdout"] == "False\n"
