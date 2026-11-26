"""The provider picker shows only the newest generation of each model family.

A provider catalog lists every generation it still serves (gemini-2.5-flash,
gemini-3-flash-preview, ... gemini-3.8-flash, and gpt-4o/gpt-5-mini next to
gpt-6); the picker keeps the newest of a family and hides the rest, which stay
reachable by typing their id in the filter.

The hard part is telling a version from a size: ``llama-3.1-70b`` is version
3.1 at size 70b, ``granite-8b``/``ministral-14b`` carry a size and no version
at all (a unit letter ``b``/``k``/``m``/``x`` glued to the number), whereas
``gpt-4o`` is version 4 with an ``o`` variant; a release stamp (``-2512``,
``-2024-11-20``) is not a component either.
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
        "ibm/granite-8b",  # the 8 is glued to a size unit, not a version
        "mistralai/mixtral-8x7b",
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


def test_older_generation_is_hidden_across_variants(face) -> None:
    # gpt-3.5-turbo, gpt-4o and gpt-5-mini are superseded by gpt-6: a variant
    # suffix names a size or a modality, not another generation.
    catalog = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.1-mini", "gpt-5-mini",
               "gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol"]
    assert face._shadowed_models(catalog) == set(catalog[:-1])
    # Within the newest generation the variant rule alone applies.
    same = ["gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-sol-pro"]
    assert face._shadowed_models(same) == {"gpt-6-sol"}
    # Other prefixes are untouched: no gemini-6 exists to retire gemini-3.
    mixed = ["gpt-6-sol", "gemini-3-flash", "gemini-2.5-flash"]
    assert face._shadowed_models(mixed) == {"gemini-2.5-flash"}
    # A size is still not a version, so it never triggers the rule.
    sizes = ["granite-8b", "granite-34b-2512"]
    assert face._shadowed_models(sizes) == set()


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


@pytest.mark.parametrize(
    "model",
    [
        # known-positive: chat models of every shape the catalogs use.
        "models/gemini-3.8-flash",
        "claude-opus-5.5",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3-vl-32b-instruct",  # vision input, still a chat endpoint
        "perplexity/sonar-pro-search",
        "Mistral-Large-3-675B-Instruct-2512-NVFP4",
    ],
)
def test_chat_models_are_kept(face, model: str) -> None:
    assert face._model_chat_ok(model)


@pytest.mark.parametrize(
    "model",
    [
        # known-negative: one id per modality the /models catalogs mix in.
        "models/veo-3.1-generate-preview",
        "models/gemini-3-pro-image",
        "models/gemini-2.5-flash-preview-tts",
        "models/gemini-2.5-flash-native-audio-preview-12-2025",
        "models/gemini-3.8-live",
        "models/lyria-3.5",
        "models/aqa",
        "models/deep-research-pro-preview-12-2025",
        "models/gemini-2.5-computer-use-preview-10-2025",
        "models/gemini-embedding-2",
        "nvidia/nvclip",
        "text-embedding-3-small",
        "bge-m3",
        "meta/llama-guard-4-12b",
        "openai/gpt-5.5:batch",
        "parakeet-tdt-0.6b-v3",  # speech-to-text, served by inference.luj.fr
    ],
)
def test_non_chat_models_are_dropped(face, model: str) -> None:
    assert not face._model_chat_ok(model)


def test_route_choices_offer_chat_models_only(tmp_path, monkeypatch) -> None:
    """The picker offers no image/audio/embedding id; an all-non-chat entry
    keeps an entry-level route instead of vanishing."""
    import importlib

    db = tmp_path / "endpoints.toml"
    db.write_text(
        '[endpoints.g]\nurl = "https://x/v1"\ntype = "chat-completions"\n'
        'models = ["g-3.8-flash", "g-2.5-flash", "g-3-pro-image", "g-3.8-live"]\n'
        '[endpoints.media]\nurl = "https://y/v1"\ntype = "chat-completions"\n'
        'models = ["veo-3", "tts-1"]\n'
    )
    (tmp_path / "models.json").write_text("{}")
    monkeypatch.setenv("INFERENCE_DB", str(db))

    spec = importlib.util.spec_from_file_location(
        "_face_routes", REPO_ROOT / "agent-kimi-k3-tui-superlean.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    routes = module._route_choices()
    offered = {r["model"]: r["shadowed"] for r in routes if r.get("model")}
    # image and live ids are absent entirely; the older version is only hidden.
    assert offered == {"g-2.5-flash": True, "g-3.8-flash": False}
    entries = [r["entry_id"] for r in routes if r["kind"] == "entry"]
    assert entries == ["media"], "an entry with no chat model stays reachable"


def test_filter_matches_entry_name_and_host(tmp_path, monkeypatch) -> None:
    """Typing the provider's host or name finds its routes, not only its id:
    ``inference.luj.fr`` finds entry ``luj``."""
    import importlib

    db = tmp_path / "endpoints.toml"
    db.write_text(
        '[endpoints.luj]\nname = "inference.luj.fr (LiteLLM proxy)"\n'
        'url = "https://inference.luj.fr/v1"\ntype = "chat-completions"\n'
        'models = ["deepseek-v4-flash"]\n'
        '[endpoints.other]\nname = "Other"\nurl = "https://api.other.com/v1"\n'
        'type = "chat-completions"\nmodels = ["deepseek-v4-flash"]\n'
    )
    (tmp_path / "models.json").write_text("{}")
    monkeypatch.setenv("INFERENCE_DB", str(db))

    spec = importlib.util.spec_from_file_location(
        "_face_host", REPO_ROOT / "agent-kimi-k3-tui-superlean.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    routes = module._route_choices()

    def shown(query: str) -> list[str]:
        return [module._route_label(routes[i])
                for i in module._matching_routes(routes, query, {})]

    # known-positive: host, name word, and host + model terms together.
    assert shown("inference.luj.fr") == ["luj · deepseek-v4-flash"]
    assert shown("litellm") == ["luj · deepseek-v4-flash"]
    assert shown("luj.fr deepseek") == ["luj · deepseek-v4-flash"]
    # known-negative: a host that is nowhere matches nothing.
    assert shown("nowhere.example") == []
    # boundary: the model id alone still lists every provider serving it.
    assert shown("deepseek-v4-flash") == [
        "luj · deepseek-v4-flash", "other · deepseek-v4-flash"]
    # the gateway route is findable by its host too.
    assert shown("superleanai.com") == [module._route_label({"kind": "superleanai"})]
