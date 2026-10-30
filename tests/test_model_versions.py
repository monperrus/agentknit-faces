"""The provider picker shows only the newest version of each model family.

A provider catalog lists every generation it still serves (gemini-2.5-flash,
gemini-3-flash-preview, gemini-3.5-flash, ... gemini-3.8-flash); the picker
keeps the newest of a family and hides the rest, which stay reachable by
typing their id in the filter.

The hard part is telling a version from a size: ``llama-3.1-70b`` is version
3.1 at size 70b, ``granite-8b``/``ministral-14b`` carry a size and no version
at all, and a date stamp (``-2512``) is not a version component.
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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("models/gemini-3.8-flash", ("models/gemini-", (3, 8), "-flash")),
        # A release channel is not part of the family identity.
        ("models/gemini-3-flash-preview", ("models/gemini-", (3,), "-flash")),
        ("gpt-5-chat-latest", ("gpt-", (5,), "-chat")),
        ("deepseek/deepseek-v3.2-exp", ("deepseek/deepseek-v", (3, 2), "")),
        # A size after the version stops the parse; it is not a component.
        ("meta-llama/llama-3.1-70b-instruct", ("meta-llama/llama-", (3, 1), "-70b-instruct")),
        ("google/gemma-3-27b-it", ("google/gemma-", (3,), "-27b-it")),
        # A size instead of a version: no family, so no shadowing.
        ("ibm/granite-34b-code-instruct", None),
        ("mistralai/ministral-14b-2512", None),
        ("models/gemini-flash-latest", None),
    ],
)
def test_model_version(face, model: str, expected) -> None:
    assert face._model_version(model) == expected


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
