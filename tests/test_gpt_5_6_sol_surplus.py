"""The GPT-5.6 Sol face routes through Surplus Intelligence.

The live model catalogue (queried 2026-11-17) declares a 1,050,000-token
context window and streaming support. Surplus is a reseller whose upstream
seller can change between requests, so strict cache-hit enforcement is off;
cache accounting is still retained whenever the selected seller reports it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FACE = REPO_ROOT / "agent-gpt-5.6-sol-surplus.py"


def load_face():
    spec = importlib.util.spec_from_file_location("face_gpt_5_6_sol_surplus", FACE)
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


def test_one_shot_uses_the_surplus_route_and_model_limits(face, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(sys, "argv", [FACE.name, "--non-interactive", "say", "ok"])
    monkeypatch.setattr(face.agentknit, "run_task", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs))

    face.main()

    schema = captured["args"][0]
    assert captured["args"][1] == "say ok"
    assert schema["model"] == "gpt-5.6-sol"
    assert schema["endpoint"] == "https://api.surplusintelligence.ai/min10/v1"
    assert schema["inference_db"] == "surplus-min10"
    assert schema["context_window"] == 1_050_000
    assert schema["compaction_trigger_tokens"] == 787_500
    assert schema["provider_api_support"] == {"streaming": {"supported": True}}
    assert captured["kwargs"]["non_interactive"] is True
    assert captured["kwargs"]["strict_cache_proof"] is False
    assert captured["kwargs"]["session_id"] is None


def test_stdin_task_and_session_are_forwarded(face, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(sys, "argv", [FACE.name, "--session", "session-123"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdin, "read", lambda: "task from stdin\n")
    monkeypatch.setattr(face.agentknit, "run_task", lambda *args, **kwargs: captured.update(
        args=args, kwargs=kwargs))

    face.main()

    assert captured["args"][1] == "task from stdin"
    assert captured["kwargs"]["session_id"] == "session-123"
    assert captured["kwargs"]["strict_cache_proof"] is False
