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
"""

import atexit
import os
import secrets as _secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

MODEL = "k3"
UPSTREAM_ENDPOINT = "https://api.kimi.com/coding/v1"
PROFILE_NAME = "kimi"
PROD_GATEWAY = "https://api.superleanai.com"
KEYRING_SERVICE = "login2"
KEYRING_KIMI_KEY = "kimi_api_key"

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
import agentknit_tui.app as agent_tui_app


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


def _build_schema(gateway_base: str, token: str) -> dict:
    schema = agentknit.load_specification(MODEL, UPSTREAM_ENDPOINT)
    # Route through the middleware; auth is the minted JWT, supplied
    # via env var so it never appears on a command line.
    schema["endpoint"] = f"{gateway_base}/{PROFILE_NAME}/v1"
    schema.pop("keyring_service", None)
    schema.pop("keyring_username", None)
    schema.pop("auth", None)
    schema["key_env"] = "SUPERLEAN_PROXY_TOKEN"
    os.environ["SUPERLEAN_PROXY_TOKEN"] = token
    schema["display_name"] = "Kimi K3 via superleanai"
    # Context window measured by llmprobe (reports/k3).
    schema["context_window"] = 1048576
    return schema


def _create_k3_client_factory(schema: dict):
    def _create_k3_client(_schema: dict | None = None):
        """Create a Coding Plan client with K3's mandatory temperature setting."""
        client = agentknit.create_client(schema)
        create = client.chat.completions.create

        def create_with_k3_temperature(*args, **kwargs):
            # K3's Coding Plan endpoint rejects agentknit's generic temperature=0.
            kwargs["temperature"] = 1
            return create(*args, **kwargs)

        client.chat.completions.create = create_with_k3_temperature
        return client

    return _create_k3_client


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

    try:
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

    # AgentTUI imports create_client into its app module, so patch that local
    # binding before construction.  Its turn worker then receives our adapted
    # client without changing agentknit's generic behaviour for other models.
    agent_tui_app.create_client = _create_k3_client_factory(schema)
    app = AgentTUI(
        schema,
        non_interactive=non_interactive,
        session_id=session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=task,
        # Kimi's Coding Plan only reports cache accounting once the prompt
        # crosses its minimum cacheable prefix (observed: fields appear from
        # ~3k prompt tokens; the exact floor is not published, 1024 is a
        # conservative estimate).  Below that floor the first call exposes no
        # cache fields, which strict cache-proof mode would misread as broken.
        min_cacheable_tokens=1024,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
