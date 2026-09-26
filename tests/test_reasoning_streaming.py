"""Reasoning-capable z.ai faces must opt into streaming.

agentknit emits the ``reasoning_delta`` / ``reasoning_stream_end`` events — the
dim-italic thinking trace every front end renders — only when the spec
declares ``provider_api_support.streaming.supported``.  The default in-memory
spec does not, so a face that forgets the declaration silently shows no
reasoning at all, in the REPL and in the TUI alike.

``agent-glm-5.3.py`` and ``agent-glm-5.3-tui.py`` hit the same z.ai coding
endpoint, and glm-5.3 streams its trace as ``reasoning_content`` SSE deltas on
~90% of calls (measured against api.z.ai on 2026-09-12; the remaining calls are
routed to a backend variant that emits none).  Both faces must opt in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every glm-5.3 face whose front end renders the reasoning trace.
STREAMING_FACES = (
    "agent-glm-5.3.py",
    "agent-glm-5.3-tui.py",
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


@pytest.mark.parametrize("filename", STREAMING_FACES)
def test_face_declares_streaming(filename, monkeypatch):
    """The schema handed to the front end must opt into SSE streaming."""
    if "tui" in filename:
        pytest.importorskip("agentknit_tui")
    monkeypatch.setattr(sys, "argv", [filename, "say ok"])

    captured: dict = {}

    def capture(schema, *args, **kwargs):
        captured["schema"] = schema
        # The TUI faces call ``app.run()`` on the constructor's return value.
        return SimpleNamespace(run=lambda: None)

    module = load_face(filename)
    monkeypatch.setattr(module.agentknit, "check_and_display_pricing", lambda *a, **k: None)
    monkeypatch.setattr(module, "validate_schema", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(module.agentknit, "run_task", capture)
    monkeypatch.setattr(module.agentknit, "run_repl", capture)
    monkeypatch.setattr(module, "AgentTUI", capture, raising=False)

    entry = getattr(module, "entry", None) or getattr(module, "main")
    entry()

    support = captured["schema"]["provider_api_support"]
    assert support["streaming"]["supported"] is True, (
        "the face must declare provider_api_support.streaming.supported, "
        "otherwise agentknit never emits reasoning_delta and no trace is shown"
    )
