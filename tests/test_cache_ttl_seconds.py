"""Faces talking to providers with a *documented* prefix-cache TTL must declare it.

Kimi's implicit Context Caching carries a 5-minute TTL by default
(platform.kimi.ai, "Choose a TTL"); DeepSeek's Context Caching on Disk keeps
entries "a few hours to a few days" (api-docs.deepseek.com/guides/kv_cache),
so the faces declare the conservative 6 h end.  With ``cache_ttl_seconds``
in the schema, a resume after the TTL is a *cold resume* — the expected full
cache re-write — instead of a spurious ``cache_proof_missing`` warning, and
the TUI can count the warmth down.  z.ai publishes no TTL, so its faces
deliberately declare nothing (agentknit's 3600 s default applies).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

FACES_WITH_TTL = (
    ("agent-kimi-k3.py", 300),
    ("agent-kimi-k3-tui.py", 300),
    ("agent-deepseek-flash.py", 21600),
    ("agent-deepseek-flash-tui.py", 21600),
    ("agent-deepseek-v4-flash.py", 21600),
)

ZAI_FACES = (
    "agent-glm-5.3.py",
    "agent-glm-5.3-tui.py",
    "agent-glm-5.3-flash-tui.py",
    "agent-glm-4.5-air.py",
)


def load_face(filename: str):
    """Import a face script (its name has dashes, so it is not a module name)."""
    path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(f"face_{filename[:-3].replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _capture_schema(monkeypatch: pytest.MonkeyPatch, filename: str) -> dict:
    captured: dict = {"kwargs": {}}

    def capture(schema, *args, **kwargs):
        captured["schema"] = schema
        captured["kwargs"] = kwargs
        # The TUI faces call ``app.run()`` on the constructor's return value.
        return SimpleNamespace(run=lambda: None)

    module = load_face(filename)
    monkeypatch.setattr(sys, "argv", [filename, "say ok"])
    if hasattr(module, "_gateway_token"):
        monkeypatch.setattr(module, "_gateway_token", lambda: "test-gateway-token")
    # The Kimi faces build their client eagerly, which resolves the API key
    # (keyring, then env): CI has neither, so hand them a stub client.
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: None))
    )
    monkeypatch.setattr(module.agentknit, "create_client", lambda *a, **k: fake_client)
    monkeypatch.setattr(module.agentknit, "check_and_display_pricing", lambda *a, **k: None)
    monkeypatch.setattr(module, "validate_schema", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module.agentknit, "run_task", capture)
    monkeypatch.setattr(module.agentknit, "run_repl", capture)
    monkeypatch.setattr(module, "AgentTUI", capture, raising=False)

    entry = getattr(module, "entry", None) or getattr(module, "main")
    entry()
    # Faces that pass knobs as run_task/run_repl kwargs (rather than schema
    # entries) deliver them alongside the schema: observe the effective
    # configuration either way.
    effective = dict(captured["kwargs"])
    effective.update(captured.get("schema") or {})
    return effective


@pytest.mark.parametrize(("filename", "ttl"), FACES_WITH_TTL)
def test_face_declares_documented_cache_ttl(filename, ttl, monkeypatch):
    if "tui" in filename:
        pytest.importorskip("agentknit_tui")
    schema = _capture_schema(monkeypatch, filename)
    assert schema.get("cache_ttl_seconds") == ttl


@pytest.mark.parametrize("filename", ZAI_FACES)
def test_zai_faces_do_not_invent_a_ttl(filename, monkeypatch):
    """z.ai publishes no cache TTL; inventing one would misclassify resumes."""
    if "tui" in filename:
        pytest.importorskip("agentknit_tui")
    schema = _capture_schema(monkeypatch, filename)
    assert "cache_ttl_seconds" not in schema
