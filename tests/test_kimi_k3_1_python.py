"""Kimi K3.1 Python face: file editing plus Python, with no shell tool."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FACE = REPO_ROOT / "agent-kimi-k3.1-python.py"
EXPECTED_TOOLS = {"read_file", "write_file", "str_replace", "python"}


def load_face():
    spec = importlib.util.spec_from_file_location("face_kimi_k3_1_python", FACE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def face(monkeypatch: pytest.MonkeyPatch):
    module = load_face()
    monkeypatch.setattr(module.agentknit, "run_repl", lambda *args, **kwargs: None)
    return module


def capture_task(face, monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}
    monkeypatch.setattr(face.agentknit, "run_task", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs))
    return captured


def test_face_uses_kimi_3_1_and_only_the_four_requested_tools(face, monkeypatch):
    captured = capture_task(face, monkeypatch)
    monkeypatch.setattr(sys, "argv", [FACE.name, "--non-interactive", "say", "ok"])

    face.main()

    schema = captured["args"][0]
    assert captured["args"][1] == "say ok"
    assert schema["model"] == "k3.1-pro"
    assert schema["inference_db"] == "kimi-coding"
    assert schema["context_window"] == 1_048_576
    assert schema["compaction_trigger_tokens"] == 786_432
    assert schema["tools"] == ["t_read", "t_write", "t_update", "t_python"]
    assert {spec["function"]["name"] for spec in schema["tool_specs"]} == EXPECTED_TOOLS
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


def test_python_tool_runs_code_and_captures_stdout_and_stderr(face):
    text, metadata = face.t_python(
        "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"
    )

    result = json.loads(text)
    assert result == {"stdout": "out\n", "stderr": "err\n", "exit_code": 7}
    assert metadata["ok"] is False


def test_python_tool_success_and_timeout(face):
    text, metadata = face.t_python("print(6 * 7)")
    assert json.loads(text) == {"stdout": "42\n", "stderr": "", "exit_code": 0}
    assert metadata["ok"] is True

    text, metadata = face.t_python("import time; time.sleep(2)", timeout=0.01)
    result = json.loads(text)
    assert result["exit_code"] is None
    assert "timed out" in result["stderr"]
    assert metadata["ok"] is False
