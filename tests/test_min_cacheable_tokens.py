"""The z.ai faces must declare z.ai's minimum cacheable prompt prefix.

z.ai caches prompt prefixes in 64-token blocks and reports
``cached_tokens: 0`` for any prompt too short to fill one block, so a
sub-block prompt legitimately shows no cache hit.  Without
``min_cacheable_tokens`` in the schema, agentknit's strict cache-proof mode
reads that as a caching failure (``cache_proof_missing`` warning — and a
fail-closed abort on the first call of a session whose response carries no
cache accounting), so the faces declare the measured floor.

Measured against api.z.ai on 2026-09-12: glm-5.3 reports no cache hit at
63/64/65 prompt tokens and its first hit (64 cached tokens) at 66;
glm-4.5-air (served as glm-5.3-flash) hits from 65.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every face that talks to z.ai / api.z.ai, plus the superlean gateway face
# that proxies to it, must carry the same floor.
ZAI_FACES = (
    "agent-glm-5.3.py",
    "agent-glm-5.3-tui.py",
    "agent-glm-4.5-air.py",
    "agent-glm-5.3-tui-superlean.py",
)

EXPECTED_MIN_CACHEABLE_TOKENS = 66


def load_face(filename: str):
    """Import a face script (its name has dashes, so it is not a module name)."""
    path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(f"face_{filename[:-3].replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", ZAI_FACES)
def test_face_declares_the_measured_floor(filename):
    module = load_face(filename)
    assert module.MIN_CACHEABLE_TOKENS == EXPECTED_MIN_CACHEABLE_TOKENS


@pytest.mark.parametrize("filename", ZAI_FACES)
def test_face_puts_the_floor_in_the_schema(filename, monkeypatch):
    """The floor must reach agentknit through the schema, whatever the front end."""
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

    assert captured["schema"]["min_cacheable_tokens"] == EXPECTED_MIN_CACHEABLE_TOKENS


def _cache_events(min_cacheable_tokens: int, prompt_tokens: int) -> list[tuple[str, dict]]:
    """Run agentknit's strict-cache check over a z.ai-shaped usage block."""
    from agentknit._core import _enforce_cache_proof

    events: list[tuple[str, dict]] = []
    session = {
        "strict_cache_proof": True,
        "llm_call_count": 2,  # past the first call: warns, never aborts
        "min_cacheable_tokens": min_cacheable_tokens,
        "on_event": lambda event_type, data: events.append((event_type, data)),
    }
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        # What z.ai returns for a prompt below one 64-token block.
        cached_tokens=0,
        cache_creation_tokens=0,
        has_cache_proof=True,
        completion_tokens=4,
    )
    _enforce_cache_proof(session, usage)
    return events


def test_sub_block_prompt_is_not_reported_as_a_cache_failure():
    events = _cache_events(EXPECTED_MIN_CACHEABLE_TOKENS, prompt_tokens=40)
    assert events, "the checker should still account for the call"
    assert events[0][0] == "cache_below_minimum"


def test_the_unset_floor_is_what_makes_the_same_call_look_broken():
    """Regression guard: this is the bug the floor fixes."""
    events = _cache_events(0, prompt_tokens=40)
    assert events[0][0] == "cache_proof_missing"


@pytest.mark.parametrize("prompt_tokens", (64, 65))
def test_the_floor_exempts_prompts_below_one_cache_block(prompt_tokens):
    """Measured: 64/65 prompt tokens still yield no hit on z.ai."""
    events = _cache_events(EXPECTED_MIN_CACHEABLE_TOKENS, prompt_tokens=prompt_tokens)
    assert events[0][0] == "cache_below_minimum"


def test_the_floor_does_not_mask_a_real_miss_above_the_block():
    events = _cache_events(EXPECTED_MIN_CACHEABLE_TOKENS, prompt_tokens=2000)
    assert events[0][0] == "cache_proof_missing"
