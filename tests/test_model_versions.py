"""The provider picker shows only the newest version of each model family.

A provider catalog lists every generation it still serves (gemini-2.5-flash,
gemini-3-flash-preview, ... gemini-3.8-flash); the picker keeps the newest of
a family and hides the rest, which stay reachable by typing their id in the
filter.

The hard part is telling a version from a size: ``llama-3.1-70b`` is version
3.1 at size 70b, ``granite-8b``/``ministral-14b`` carry a size and no version
at all, and a release stamp (``-2512``, ``-2024-11-20``) is not a component.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTKNIT = Path.home() / "workspace" / "prototypes" / "agentknit"
if str(AGENTKNIT) not in sys.path:
    sys.path.insert(0, str(AGENTKNIT))


@pytest.fixture(scope="module")
def face():
    path = REPO_ROOT / "agent-kimi-k3-tui-superlean.py"
    spec = importlib.util.spec_from_file_location("_face_versions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _split(face, model: str) -> tuple[str, tuple[int, ...], str, tuple[int, ...]]:
    """(prefix, version, variant, date) -- channel flag left out, it is plumbing."""
    parts = face._model_version(model)
    assert parts is not None, model
    return parts.prefix, parts.version, parts.variant, parts.date


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/gemini-3.8-flash", ("models/gemini-", (3, 8), "-flash", ())),
        # A release channel is not part of the family identity.
        ("models/gemini-3-flash-preview", ("models/gemini-", (3,), "-flash", ())),
        ("gpt-5-chat-latest", ("gpt-", (5,), "-chat", ())),
        # A size after the version stops the parse; it is not a component.
        ("meta-llama/llama-3.1-70b-instruct",
         ("meta-llama/llama-", (3, 1), "-70b-instruct", ())),
        ("google/gemma-3-27b-it", ("google/gemma-", (3,), "-27b-it", ())),
        # Release stamps are split off, in the shapes providers use.
        ("models/gemini-2.5-flash-native-audio-preview-12-2025",
         ("models/gemini-", (2, 5), "-flash-native-audio", (2025, 12, 0))),
        ("mistralai/mistral-large-2512", ("mistralai/mistral-large", (),
                                          "", (2025, 12, 0))),
        ("openai/gpt-4.1-2025-04-14", ("openai/gpt-", (4, 1), "", (2025, 4, 14))),
        # The openrouter marker for aliases is not part of the id.
        ("~anthropic/claude-opus-latest", None),
    ],
)
def test_model_version(face, model: str, expected) -> None:
    if expected is None:
        assert face._model_version(model) is None
        return
    assert _split(face, model) == expected


@pytest.mark.parametrize(
    "model",
    [
        # Sizes, not versions: these must never group with each other.
        "ibm/granite-34b-code-instruct",
        "mistralai/ministral-14b-2512",
        "models/gemini-flash-latest",
        "gpt-4o",  # the 4 is glued to a letter, so it is not a version
    ],
)
def test_unversioned(face, model: str) -> None:
    assert face._model_version(model) is None


def test_shadowed_models_keeps_the_newest(face) -> None:
    family = ["models/gemini-2.5-flash", "models/gemini-3-flash-preview",
              "models/gemini-3.8-flash"]
    assert face._shadowed_models(family) == {"models/gemini-2.5-flash",
                                             "models/gemini-3-flash-preview"}
    # Variants are separate families: -pro and -flash-lite survive.
    unrelated = ["models/gemini-3.8-flash", "models/gemini-3.8-flash-lite-tts",
                 "models/gemini-3.8-pro"]
    assert face._shadowed_models(unrelated) == set()
    # Sizes never collapse into each other.
    sizes = ["ibm/granite-8b-code-instruct", "ibm/granite-34b-code-instruct",
             "mistralai/ministral-3b-2512", "mistralai/ministral-14b-2512"]
    assert face._shadowed_models(sizes) == set()


def test_newest_release_stamp_wins(face) -> None:
    audio = ["models/gemini-2.5-flash-native-audio-preview-09-2025",
             "models/gemini-2.5-flash-native-audio-preview-12-2025"]
    assert face._shadowed_models(audio) == {audio[0]}
    # Within one version a plain id beats its own snapshots ...
    plain = ["gpt-4.1", "gpt-4.1-2025-04-14", "gpt-4.1-2025-08-06"]
    assert face._shadowed_models(plain) == set(plain[1:])
    # ... and the snapshots beat the floating -latest alias.
    floating = ["mistralai/mistral-large", "mistralai/mistral-large-latest",
                "mistralai/mistral-large-2512"]
    assert face._shadowed_models(floating) == {"mistralai/mistral-large",
                                               "mistralai/mistral-large-latest"}


def _route(model: str, *, shadowed: bool = False) -> dict:
    return {"kind": "model", "entry_id": "e", "model": model, "shadowed": shadowed}


def test_matching_routes_hides_shadowed_ids(face) -> None:
    routes = [_route("models/gemini-3.8-flash"),
              _route("models/gemini-2.5-flash", shadowed=True),
              _route("models/gemini-2.5-flash-native-audio", shadowed=False)]
    visible = face._matching_routes(routes, "", {})
    assert visible == [0, 2]
    # A full id reaches a hidden version even when it is a substring of a
    # visible one, which is how an older model stays pickable.
    exact = face._matching_routes(routes, "models/gemini-2.5-flash", {})
    assert exact == [1]
    # Fuzzy queries still drop the hidden ids ...
    assert face._matching_routes(routes, "gemini-2.5-flash", {}) == [2]
    # ... unless nothing else matches: the last resort is to show them.
    assert face._matching_routes(routes, "2.5-flash-na", {}) == [2]
    assert face._matching_routes([_route("m", shadowed=True)], "m", {}) == [0]
