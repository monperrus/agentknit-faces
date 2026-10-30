#!/usr/bin/env python3
"""Textual TUI for Kimi K3 routed through the superleanai middleware.

Same agent as agent-kimi-k3-tui.py, but instead of hitting
https://api.kimi.com/coding/v1 directly, every request goes through the
superleanai gateway's ``kimi`` profile:

    client --> <gateway>/kimi/v1/chat/completions
           --> https://api.kimi.com/coding/v1/chat/completions

Two gateway modes, selected by ``--local``:

* production (default) -- https://api.superleanai.com, which logs every
  request to the production Postgres database. Auth is a long-lived v2
  EdDSA JWT bound to an ``api_keys`` row created on
  https://dashboard.superleanai.com/keys (the Kimi Coding Plan credential
  is stored encrypted server-side under that row). The token lives in the
  system keyring (service ``login2``, username ``SUPERLEAN_API_KEY``; env
  var of the same name takes precedence). If you revoke/rotate the key on
  the dashboard, mint a replacement JWT server-side over the same jti and
  update the keyring entry.
* local (``--local``) -- spins up ``go run ./cmd/proxy`` from the local
  superleanai checkout on a free loopback port; traffic is only logged
  locally (if SUPERLEANAI_DATABASE_URL is set). Auth is a short-lived
  HS256 JWT minted locally (same as ``cmd/createjwt --byok``): its
  ``openrouter_key`` claim carries the real Kimi Coding Plan API key read
  from the system keyring (service ``login2``, username ``kimi_api_key``),
  which the gateway swaps into the upstream ``Authorization`` header.

Note: the production gateway no longer accepts those self-minted HS256
BYOK tokens -- it requires v2 EdDSA JWTs (``internal/auth``), which only
the dashboard signer (holder of the production Ed25519 private key) can
issue. That is why auth against production goes through a dashboard-bound
token while ``--local`` keeps minting BYOK tokens.

Neither key is ever placed in this script or on a command line.

Usage:
    agent-kimi-k3-tui-superlean.py "<task>"           # open TUI with task prefilled
    agent-kimi-k3-tui-superlean.py                    # interactive TUI
    agent-kimi-k3-tui-superlean.py --local            # use a local gateway instance
    agent-kimi-k3-tui-superlean.py --session <id>     # resume a previous session
    agent-kimi-k3-tui-superlean.py --non-interactive  # disable ask_user_question

The first screen asks which inference provider / model to use. Routes:

* superleanai (default) -- this script's usual Kimi-K3-over-gateway
  schema, with the superleanai endpoint chosen by --local.
* anything else -- an inference-db entry; the models offered come from
  inference-db's cached model list (~/.config/inference-db/models.json,
  refresh with `inference-db refresh-models`). Non-superleanai routes are
  plain agentknit loads, which only speak chat-completions -- entries typed
  "messages" (Anthropic) or "responses" (OpenAI Responses, ChatGPT Codex)
  are not offered at all, and models obviously unusable for chat
  (video/image/audio generators, :batch-only ids) are filtered out of
  large catalogs.
"""

import atexit
import json
import os
import secrets as _secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

MODEL = "k3"
# Production gateway endpoint and K3's temperature=1 quirk come from
# inference-db (upstream: entry kimi-coding); --local overrides the endpoint.
INFERENCE_DB = "superleanai-kimi"
PROFILE_NAME = "kimi"
PROD_GATEWAY = "https://api.superleanai.com"
KEYRING_SERVICE = "login2"
KEYRING_KIMI_KEY = "kimi_api_key"
KIMI_CONTEXT_WINDOW = 1048576

SUPERLEAN_ROOT = os.environ.get(
    "SUPERLEANAI_ROOT", os.path.expanduser("~/workspace/superleanai")
)


def _keyring_password(username: str, required: bool = True) -> str:
    import keyring

    value = keyring.get_password(KEYRING_SERVICE, username)
    if not value and required:
        raise RuntimeError(
            f"No '{username}' in system keyring (service '{KEYRING_SERVICE}')."
        )
    return value or ""


def _jwt_secret() -> str:
    return os.environ.get("JWT_SECRET") or _keyring_password("JWT_SECRET")


def _api_token() -> str:
    """Dashboard-issued v2 JWT for the production gateway (see module docstring)."""
    return os.environ.get("SUPERLEAN_API_KEY") or _keyring_password("SUPERLEAN_API_KEY")


def _mint_proxy_jwt(kimi_api_key: str, secret: str) -> str:
    """Mint the gateway auth token, same claims as cmd/createjwt --byok."""
    import jwt  # PyJWT, also used by superleanai's Python predecessor

    claims = {
        "email": "martin@monperrus.net",
        "openrouter_key": kimi_api_key,  # forwarded upstream as the Bearer key
        "jti": _secrets.token_hex(8),
        "iat": int(time.time()),
    }
    return jwt.encode(claims, secret, algorithm="HS256")


def _pick_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_local_proxy() -> tuple[subprocess.Popen, int]:
    """Start ``go run ./cmd/proxy`` on a free loopback port, wait until up."""
    if not os.path.isdir(os.path.join(SUPERLEAN_ROOT, "cmd", "proxy")):
        raise RuntimeError(
            f"superleanai checkout not found at {SUPERLEAN_ROOT} "
            "(set SUPERLEANAI_ROOT to override)."
        )
    profiles = os.path.join(SUPERLEAN_ROOT, "service", "profiles")
    if not os.path.isfile(os.path.join(profiles, f"{PROFILE_NAME}.json")):
        raise RuntimeError(
            f"Missing profile {os.path.join(profiles, PROFILE_NAME + '.json')} "
            f'(expected e.g. {{"target_url": "https://api.kimi.com/coding", '
            f'"tool_strategy": "debloat"}}).'
        )
    port = _pick_port()
    proc = subprocess.Popen(
        ["go", "run", "./cmd/proxy", "--port", str(port), "--profiles", profiles],
        cwd=SUPERLEAN_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    def _stop() -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    atexit.register(_stop)

    deadline = time.monotonic() + 60  # first `go run` may compile
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"superleanai proxy exited early (code {proc.returncode}).")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/", timeout=1
            ) as resp:
                if resp.status == 200:
                    return proc, port
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("superleanai proxy did not come up within 60s.")


def _wait_for_gateway(url: str, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/", timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Gateway {url} is not reachable.")


# Ensure generated resume commands return to this TUI launcher.
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

sys.path.insert(0, os.path.expanduser("~/workspace/prototypes/agentknit"))
sys.path.insert(0, os.path.expanduser("~/workspace/prototypes/agentknit-tui"))

import agentknit
from agentknit import validate_schema
from agentknit.exceptions import (
    AgentSpecDisabledError,
    AgentSpecInvalidError,
    AuthenticationError,
    PricingLimitExceededError,
    RateLimitError,
)
from agentknit_tui import AgentTUI


_home = os.path.expanduser("~")
_cwd = os.getcwd()
_SUPPLEMENT = (
    "Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}.\n"
    "When creating git commits, use the author/committer email "
    "martin.monperrus@gnieh.org."
)


def _patch_accept_encoding() -> None:
    """Make agentknit's requests-based client never send Accept-Encoding.

    The superleanai gateway forwards client headers verbatim upstream but
    strips Content-Encoding from the response, so an explicitly-requested
    gzip body would come back compressed yet undeclared and break the JSON
    parser. With ``identity`` requested, the upstream answers in plain JSON.
    """
    import requests

    original = requests.Session.request

    def request_without_accept_encoding(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        # requests prepares a default Accept-Encoding even when headers are
        # passed, so it must be overridden explicitly with identity.
        headers["Accept-Encoding"] = "identity"
        kwargs["headers"] = headers
        return original(self, method, url, **kwargs)

    requests.Session.request = request_without_accept_encoding


def _build_schema(gateway_base: str, token: str, model: str = MODEL) -> dict:
    schema = agentknit.load_specification(model, inference_db=INFERENCE_DB)
    if gateway_base != PROD_GATEWAY:
        schema["endpoint"] = f"{gateway_base}/{PROFILE_NAME}/v1"
    # Auth is the minted JWT, supplied via env var so it never appears on a
    # command line; an explicit key_env wins over the entry's key sources.
    schema.pop("keyring_service", None)
    schema.pop("keyring_username", None)
    schema.pop("auth", None)
    schema["key_env"] = "SUPERLEAN_PROXY_TOKEN"
    os.environ["SUPERLEAN_PROXY_TOKEN"] = token
    schema["display_name"] = "Kimi K3 via superleanai"
    # Context window measured by llmprobe (reports/k3); K3.1 ids have the
    # same 1 Mi window per the Coding Plan /models endpoint.
    schema["context_window"] = KIMI_CONTEXT_WINDOW
    # Trigger compaction at 75 % of the window: the last observed prompt plus
    # one more assistant reply + tool result must still fit.
    schema["compaction_trigger_tokens"] = int(KIMI_CONTEXT_WINDOW * 0.75)
    return schema


def _build_entry_schema(entry_id: str, model: str) -> dict:
    """Plain agentknit schema for an inference-db entry (no gateway)."""
    schema = agentknit.load_specification(model, inference_db=entry_id)
    schema["display_name"] = f"{model} via {entry_id}"
    return schema


# Model ids that can never carry a chat conversation. Provider catalogs mix
# generators and batch-only ids into /models; offering them in the picker is
# a guaranteed 400/404.
_BAD_MODEL_PARTS = (
    "veo", "sora", "imagen", "dall-e", "tts", "transcribe", "whisper",
    "embed", "bge-", "moderation", "guard",
)

# Entries whose whole cached catalog is filtered (not just curated subsets):
# above this count the list is a provider dump, not a choice the provider
# made for the user, so obviously non-chat ids are dropped.
_FILTER_CATALOG_ABOVE = 12


def _model_chat_ok(model_id: str) -> bool:
    """Keep model ids that can plausibly carry a chat conversation."""
    low = model_id.lower()
    if low.endswith((":batch", "-batch")):
        return False
    return not any(p in low for p in _BAD_MODEL_PARTS)


def _route_label(route: dict) -> str:
    if route["kind"] == "superleanai":
        return "superleanai → kimi-coding (logged gateway, Kimi K3)"
    if route["kind"] == "entry":
        return route["entry_id"]
    return f"{route['entry_id']} · {route['model']}"


def _route_choices() -> list[dict]:
    """First-screen routes: the superleanai gateway, then one per (entry, model)."""
    from agentknit import inference_db as idb

    import tomllib

    with open(idb.db_path(), "rb") as f:
        entries = tomllib.load(f).get("endpoints", {})
    cache_path = Path(idb.db_path()).parent / "models.json"
    cached: dict = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())

    routes: list[dict] = [{"kind": "superleanai"}]
    for entry_id in sorted(entries):
        if entry_id == INFERENCE_DB:
            continue
        entry = entries[entry_id]
        if not isinstance(entry, dict):
            continue  # sub-tables (headers, list_models, ...) share the prefix
        if entry.get("dormant"):
            continue  # exhausted/dead account, kept in the db for reference
        types = entry.get("type") or []
        if isinstance(types, str):
            types = [types]
        # agentknit only speaks chat-completions to the entry's url; it
        # cannot honour "messages" (Anthropic /v1/messages) or "responses"
        # (OpenAI Responses API) types and 400/404s on them. Offering those
        # routes only produced guaranteed-broken sessions.
        if "chat-completions" not in types:
            continue
        models = (cached.get(entry_id) or {}).get("models") or entry.get("models") or []
        if entry.get("list_models", {}).get("unsupported") and not models:
            continue
        if len(models) > _FILTER_CATALOG_ABOVE:
            models = [m for m in models if _model_chat_ok(m)]
        if models:
            routes.extend({"kind": "model", "entry_id": entry_id, "model": m}
                          for m in sorted(models))
        else:
            routes.append({"kind": "entry", "entry_id": entry_id})
    return routes


class ProviderPickerScreen(Screen[dict]):
    """First screen: pick the inference provider / model for this session."""

    CSS = """
    ProviderPickerScreen {
        align: center middle;
    }
    #picker {
        width: 72;
        height: auto;
        max-height: 90%;
        border: round $accent;
        padding: 1 2;
    }
    #picker-filter {
        height: auto;
        margin: 0 0 1 0;
    }
    #picker-list-box {
        height: auto;
        max-height: 30;
    }
    #picker-list-box ListView {
        height: auto;
        max-height: 30;
    }
    #picker-hint {
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "cancel", "Use default")]

    def __init__(self, routes: list[dict]) -> None:
        super().__init__()
        self._routes = routes
        self._visible = list(range(len(routes)))
        self._generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static("Which inference provider / model?", id="picker-title")
            yield Input(placeholder="filter (e.g. kimi, glm, opus)…", id="picker-filter")
            yield Vertical(id="picker-list-box")
            yield Static("type to filter · ↓/↑/enter: select · esc: default (superleanai)",
                         id="picker-hint")

    def on_mount(self) -> None:
        self._rebuild_list("")
        self.query_one("#picker-filter", Input).focus()

    def _rebuild_list(self, query: str) -> None:
        terms = query.strip().lower().split()
        self._visible = [i for i, r in enumerate(self._routes)
                         if all(t in _route_label(r).lower() for t in terms)]
        self._generation += 1
        view = ListView(*[
            # Fresh ids each rebuild: the previous ListView unmounts
            # asynchronously, so reusing ids would clash in the DOM registry.
            ListItem(Label(_route_label(self._routes[i])),
                     id=f"route-{self._generation}-{i}")
            for i in self._visible[:200]  # cap: ListView with 1000+ items is slow
        ])
        box = self.query_one("#picker-list-box", Vertical)
        box.remove_children()
        box.mount(view)

    @on(Input.Changed)
    def _filter(self, event: Input.Changed) -> None:
        self._rebuild_list(event.value)

    def on_key(self, event) -> None:
        if self.focused and self.focused.id == "picker-filter":
            if event.key == "down":
                self.query_one(ListView).focus()
                event.stop()
                return
            if event.key == "enter":
                # On a unique match, enter picks it directly; otherwise move
                # into the list (same as arrow down).
                event.stop()
                if len(self._visible) == 1:
                    self.dismiss(self._routes[self._visible[0]])
                elif self._visible:
                    self.query_one(ListView).focus()
                return

    @on(ListView.Selected)
    def _selected(self, event: ListView.Selected) -> None:
        index = int(str(event.item.id).rsplit("-", 1)[-1])
        self.dismiss(self._routes[index])

    def action_cancel(self) -> None:
        self.dismiss({"kind": "superleanai"})


def _run_route_picker(routes: list[dict]) -> dict:
    """Show the picker as a first screen; returns the chosen route."""
    from textual.app import App

    class PickerApp(App[dict]):
        def compose(self) -> ComposeResult:
            yield from ()

        def on_mount(self) -> None:
            self.push_screen(ProviderPickerScreen(routes),
                             lambda route: self.exit(route))

    app = PickerApp()
    app.run()
    return app.return_value or {"kind": "superleanai"}


def main() -> int:
    local = "--local" in sys.argv
    non_interactive = "--non-interactive" in sys.argv
    session_id = None
    if "--session" in sys.argv:
        index = sys.argv.index("--session")
        if index + 1 < len(sys.argv):
            session_id = sys.argv[index + 1]

    flags_with_value = {"--session"}
    task_args: list[str] = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in flags_with_value:
            skip_next = True
            continue
        if argument.startswith("--"):
            continue
        task_args.append(argument)
    task = " ".join(task_args)

    route = _run_route_picker(_route_choices())

    try:
        if route["kind"] == "superleanai":
            _patch_accept_encoding()
            if local:
                token = _mint_proxy_jwt(_keyring_password(KEYRING_KIMI_KEY), _jwt_secret())
            else:
                token = _api_token()
            if local:
                _proc, port = _start_local_proxy()
                gateway_base = f"http://127.0.0.1:{port}"
                print(f"superleanai middleware (local) on {gateway_base}/{PROFILE_NAME}/v1")
            else:
                gateway_base = PROD_GATEWAY
                _wait_for_gateway(gateway_base)
                print(f"superleanai middleware (production) on {gateway_base}/{PROFILE_NAME}/v1")
            schema = _build_schema(gateway_base, token)
        else:
            model = route.get("model") or input("model id: ").strip()
            schema = _build_entry_schema(route["entry_id"], model)

        validate_schema(schema)
        agentknit.check_and_display_pricing(schema)
    except AgentSpecDisabledError as exc:
        print(f"Agent disabled: {exc.comment or exc}", file=sys.stderr)
        return 2
    except AgentSpecInvalidError as exc:
        print(exc, file=sys.stderr)
        return 2
    except PricingLimitExceededError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    except AuthenticationError as exc:
        print(f"Authentication error: {exc}", file=sys.stderr)
        return 2
    except RateLimitError as exc:
        print(f"Rate limited: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    app = AgentTUI(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=task,
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 4096 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=4096,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
