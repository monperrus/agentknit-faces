"""Kimi K3.1 Python REPL face: one persistent tool for every action."""

from __future__ import annotations

import importlib.util
import json
import subprocess
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



def test_notebook_sink_maintains_valid_notebook(face, tmp_path):
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)

    sink.append({"type": "message", "msg": {"role": "user", "content": "Compute the answer"}})
    sink.append({"type": "tool_start", "call_id": "call-1", "name": "t_python",
                 "args": {"code": "answer = 6 * 7"}})
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "Compute the answer" in "".join(notebook["cells"][0]["source"])
    assert notebook["cells"][1]["cell_type"] == "code"
    assert notebook["cells"][1]["execution_count"] is None
    assert notebook["cells"][1]["outputs"] == []

    sink.append({"type": "tool_end", "call_id": "call-1", "name": "t_python",
                 "result": json.dumps({"stdout": "", "stderr": "", "exit_code": 0})})
    sink.append({"type": "message", "msg": {"role": "assistant", "content": "The answer is 42."}})
    sink.append({"type": "turn_end"})
    notebook = json.loads(path.read_text())
    code = notebook["cells"][1]
    assert code["execution_count"] == 1
    assert code["outputs"][0]["output_type"] == "stream"
    assert code["outputs"][0]["name"] == "stdout"
    assert code["outputs"][0]["text"] == []
    assert notebook["cells"][2]["cell_type"] == "markdown"
    assert "The answer is 42." in "".join(notebook["cells"][2]["source"])
    assert notebook["cells"][3]["cell_type"] == "code"
    assert notebook["cells"][3]["source"] == []


def test_notebook_sink_captures_results_errors_and_multiple_calls(face, tmp_path):
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)
    sink.append({"type": "message", "msg": {"role": "user", "content": "Run two cells"}})
    for call_id, code in [("one", "print('hello')"), ("two", "1 / 0")]:
        sink.append({"type": "tool_start", "call_id": call_id, "name": "t_python",
                     "args": {"code": code}})
        result = {"stdout": "hello\n" if call_id == "one" else "",
                  "stderr": "" if call_id == "one" else "ZeroDivisionError\n",
                  "exit_code": 0 if call_id == "one" else 1}
        sink.append({"type": "tool_end", "call_id": call_id, "name": "t_python",
                     "result": json.dumps(result)})
    notebook = json.loads(path.read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell["source"]]
    assert [cell["execution_count"] for cell in cells] == [1, 2]
    assert cells[0]["outputs"][0]["text"] == ["hello\n"]
    assert cells[1]["outputs"][1] == {"output_type": "stream", "name": "stderr",
                                       "text": ["ZeroDivisionError\n"]}
    assert cells[1]["metadata"]["agentknit"]["exit_code"] == 1
    check = subprocess.run(["/usr/bin/python3", "-c",
        "import nbformat, sys; nbformat.validate(nbformat.read(sys.argv[1], as_version=4))",
        str(path)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_notebook_sink_handles_assistant_commentary_and_plain_errors(face, tmp_path):
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)
    sink.append({"type": "message", "msg": {"role": "user", "content": "Test"}})
    sink.append({"type": "message", "msg": {"role": "assistant", "content": "Checking now."}})
    sink.append({"type": "tool_start", "call_id": "x", "name": "t_python",
                 "args": {"code": "raise ValueError('bad')"}})
    sink.append({"type": "tool_end", "call_id": "x", "result": "ERROR: tool failed"})
    notebook = json.loads(path.read_text())
    markdown = [cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "Checking now." in "".join(markdown[-1]["source"])
    assert code[-1]["outputs"][0]["name"] == "stdout"
    assert code[-1]["outputs"][1]["name"] == "stderr"
    assert code[-1]["metadata"]["agentknit"]["exit_code"] is None


def test_notebook_sink_rebuilds_history_when_resuming(face, tmp_path):
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)
    sink.append({"type": "message", "msg": {"role": "user", "content": "First"}})
    sink.append({"type": "tool_start", "call_id": "old", "name": "t_python",
                 "args": {"code": "x = 1"}})
    sink.append({"type": "tool_end", "call_id": "old", "name": "t_python",
                 "result": json.dumps({"stdout": "1\n", "stderr": "", "exit_code": 0})})
    resumed = face._NotebookSink(path, replay=[
        {"type": "message", "msg": {"role": "system", "content": "instructions"}},
        {"type": "message", "msg": {"role": "user", "content": "First"}},
        {"type": "message", "msg": {"role": "assistant", "content": "",
             "tool_calls": [{"id": "old", "type": "function", "function": {
                 "name": "python", "arguments": json.dumps({"code": "x = 1"})}}]}},
        {"type": "message", "msg": {"role": "tool", "tool_call_id": "old",
             "name": "t_python", "content": json.dumps(
                 {"stdout": "1\n", "stderr": "", "exit_code": 0})}},
    ])
    notebook = json.loads(path.read_text())
    assert notebook["metadata"]["agentknit"]["session_id"] is None
    assert len([c for c in notebook["cells"] if c["cell_type"] == "code" and c["source"]]) == 1
    resumed.append({"type": "tool_start", "call_id": "new", "name": "t_python",
                    "args": {"code": "x + 1"}})
    assert json.loads(path.read_text())["cells"][-1]["execution_count"] is None


def test_notebook_arguments_and_main_wiring(face, monkeypatch, tmp_path):
    assert face._notebook_arg([]) is None
    assert face._notebook_arg(["--notebook"]) is True
    assert face._notebook_arg(["--notebook", "x.ipynb"]) == "x.ipynb"
    captured = capture_task(face, monkeypatch)
    monkeypatch.setattr(sys, "argv", [FACE.name, "--notebook", "run", "task"])
    face.main()
    assert captured["args"][1] == "run task"
    assert isinstance(captured["kwargs"]["durable_sink"], face._NotebookSink)
    assert captured["kwargs"]["durable_sink"].path is None
    assert captured["kwargs"]["durable_sink"].requested is True


def test_real_agent_session_updates_notebook_after_each_step(face, tmp_path, monkeypatch):
    transcript = tmp_path / "session.jsonl"
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)
    schema = {
        "model": face.MODEL, "endpoint": "http://unused.invalid", "tool_specs": [],
        "tools": [], "inferred_tool_schema": [], "tool_dispatch": {},
        "behaviour": {"call_delivery_mode": "structured_tool_calls"},
    }
    session = face.agentknit.init_session(schema, durable=False, durable_sink=sink,
                                          time_awareness_enabled=False)
    # Use the public sink with the actual session stream, not hand-built notebook cells.
    sink.bind(str(transcript), session["session_id"])
    sink.append({"type": "message", "msg": {"role": "user", "content": "Hello"}})
    sink.append({"type": "tool_start", "call_id": "live", "name": "t_python",
                 "args": {"code": "print('live')"}})
    before = json.loads(path.read_text())
    assert before["cells"][-1]["execution_count"] is None
    sink.append({"type": "tool_end", "call_id": "live", "name": "t_python",
                 "result": json.dumps({"stdout": "live\n", "stderr": "", "exit_code": 0})})
    sink.append({"type": "turn_end"})
    after = json.loads(path.read_text())
    assert after["cells"][-2]["execution_count"] == 1
    assert not (tmp_path / "session.ipynb.tmp").exists()


def test_default_notebook_path_uses_system_session_identity(face, tmp_path, monkeypatch):
    log_root = tmp_path / "probe"
    monkeypatch.setattr(face.agentknit._core, "LOG_BASE", log_root)
    sink = face._NotebookSink()
    sink.append({"type": "message", "msg": {"role": "system",
        "content": "Session: abc123 (transcript: unused)"}})
    sink.append({"type": "message", "msg": {"role": "user", "content": "Hello"}})
    expected = log_root / face.MODEL / __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    notebooks = list(expected.glob("*_abc123.ipynb"))
    assert len(notebooks) == 1
    notebook = json.loads(notebooks[0].read_text())
    assert notebook["metadata"]["agentknit"]["session_id"] == "abc123"
    assert "Hello" in "".join(notebook["cells"][-1]["source"])


def test_awareness_footer_is_not_part_of_notebook_prompt(face, tmp_path):
    path = tmp_path / "session.ipynb"
    sink = face._NotebookSink(path)
    sink.append({"type": "message", "msg": {"role": "user",
        "content": "Question\n\n<session_time>elapsed</session_time>"}})
    text = "".join(json.loads(path.read_text())["cells"][-1]["source"])
    assert "Question" in text
    assert "session_time" not in text
