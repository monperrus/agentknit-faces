#!/usr/bin/env python3
"""Textual TUI agent for glm-5.3 via api.z.ai, routed through superleanai.

Same agent as agent-glm-5.3-tui.py, but instead of hitting
https://api.z.ai/api/coding/paas/v4 directly, every request goes through a
local superleanai ``cmd/proxy`` instance (started on demand) configured
with a ``zai`` profile:

    client --> http://127.0.0.1:<port>/zai/v1/chat/completions
           --> https://api.z.ai/api/coding/paas/v4/chat/completions

so the middleware debloats/blocks tools and logs usage like for any
other profile.

The TUI authenticates to the proxy with a short-lived HS256 JWT minted
locally (same claims as ``cmd/createjwt --byok``): the JWT's
``openrouter_key`` claim carries the real z.ai API key, which the proxy
swaps into the upstream ``Authorization`` header.  The z.ai key is read
from the system keyring (service ``z.ai``, username ``api_key``) and the
signing secret from ``JWT_SECRET`` (env var or keyring service
``login2``); neither is ever placed in this script or on a command line.

Prerequisite: a ``zai`` profile in the superleanai profiles directory
(--profiles, or the profiles/ dir next to the repo checkout):

    {
      "target_url": "https://api.z.ai/api/coding/paas/v4",
      "path_strip_prefix": "/v1",
      "tool_strategy": "debloat"
    }

Usage:
    agent-glm-5.3-tui-superlean.py "<task>"          # start TUI with task prefilled
    agent-glm-5.3-tui-superlean.py                  # interactive TUI
    agent-glm-5.3-tui-superlean.py --session <id>   # resume a previous trajectory
    agent-glm-5.3-tui-superlean.py --non-interactive # drop ask_user* tools
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

MODEL = "glm-5.3"
UPSTREAM_ENDPOINT = "https://api.z.ai/api/coding/paas/v4"
PROFILE_NAME = "zai"
KEYRING_SERVICE = "z.ai"
KEYRING_USERNAME = "api_key"

SUPERLEAN_ROOT = os.environ.get(
    "SUPERLEANAI_ROOT", os.path.expanduser("~/workspace/superleanai")
)

project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)


def _keyring_password(service: str, username: str) -> str:
    import keyring

    value = keyring.get_password(service, username)
    if not value:
        raise RuntimeError(
            f"No '{username}' in system keyring (service '{service}')."
        )
    return value


def _jwt_secret() -> str:
    return os.environ.get("JWT_SECRET") or _keyring_password("login2", "JWT_SECRET")


def _mint_proxy_jwt(zai_api_key: str) -> str:
    """Mint the proxy auth token, same claims as cmd/createjwt --byok."""
    import jwt  # PyJWT, also used by superleanai's Python predecessor

    claims = {
        "email": "local-superlean",
        "openrouter_key": zai_api_key,  # forwarded upstream as the Bearer key
        "jti": _secrets.token_hex(8),
        "iat": int(time.time()),
    }
    return jwt.encode(claims, _jwt_secret(), algorithm="HS256")


def _pick_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_proxy() -> tuple[subprocess.Popen, int]:
    """Start ``go run ./cmd/proxy`` on a free loopback port, wait until up."""
    if not os.path.isdir(os.path.join(SUPERLEAN_ROOT, "cmd", "proxy")):
        raise RuntimeError(
            f"superleanai checkout not found at {SUPERLEAN_ROOT} "
            "(set SUPERLEANAI_ROOT to override)."
        )
    profiles = os.path.join(SUPERLEAN_ROOT, "profiles")
    if not os.path.isfile(os.path.join(profiles, f"{PROFILE_NAME}.json")):
        raise RuntimeError(
            f"Missing profile {os.path.join(profiles, PROFILE_NAME + '.json')} "
            f'(expected e.g. {{"target_url": "{UPSTREAM_ENDPOINT}", '
            f'"path_strip_prefix": "/v1", "tool_strategy": "debloat"}}).'
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


# Ensure generated resume commands return to this TUI launcher.
os.environ["AGENTKNIT_RESUME_COMMAND"] = os.path.realpath(__file__)

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

# Async shell tools ("nohup" / "nohup_query"): definitions and implementations
# live in agentknit.async_toolkit; one call wires specs + dispatch.
from agentknit.async_toolkit import enable_nohup

_home = os.path.expanduser("~")
_cwd = os.getcwd()
_SUPPLEMENT = (
    f"Environment paths (do NOT assume or guess these — use the values below):\n"
    f"- HOME: {_home}\n"
    f"- Current working directory: {_cwd}\n"
    f"Never assume the home directory is /home/user; it is {_home}.\n"
    f"When creating git commits, use the author/committer email "
    f"martin.monperrus@gnieh.org."
)


def _build_schema(port: int, token: str) -> dict:
    schema = agentknit.load_specification(MODEL, UPSTREAM_ENDPOINT)
    # Route through the local middleware; auth is the minted JWT, supplied
    # via env var so it never appears on a command line.
    schema["endpoint"] = f"http://127.0.0.1:{port}/{PROFILE_NAME}/v1"
    schema.pop("keyring_service", None)
    schema.pop("keyring_username", None)
    schema.pop("auth", None)
    schema["key_env"] = "SUPERLEAN_PROXY_TOKEN"
    os.environ["SUPERLEAN_PROXY_TOKEN"] = token
    schema["display_name"] = f"agent-glm-5.3-tui-superlean ({MODEL})"
    enable_nohup(schema)
    return schema


def main() -> int:
    _non_interactive = "--non-interactive" in sys.argv
    _session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session")
        if idx + 1 < len(sys.argv):
            _session_id = sys.argv[idx + 1]

    # Collect task from CLI args (one-shot mode pre-fills the prompt); skip
    # known flags so the session-id value is never mistaken for a task.
    _FLAGS_WITH_VALUE = {"--session"}
    _task_args: list[str] = []
    _skip_next = False
    for _arg in sys.argv[1:]:
        if _skip_next:
            _skip_next = False
            continue
        if _arg in _FLAGS_WITH_VALUE:
            _skip_next = True
            continue
        if _arg.startswith("--"):
            continue
        _task_args.append(_arg)
    _task = " ".join(_task_args) if _task_args else None

    try:
        token = _mint_proxy_jwt(
            _keyring_password(KEYRING_SERVICE, KEYRING_USERNAME)
        )
        _proc, port = _start_proxy()
        print(f"superleanai middleware on http://127.0.0.1:{port}/{PROFILE_NAME}/v1")
        schema = _build_schema(port, token)

        validate_schema(schema)
        agentknit.check_and_display_pricing(schema)
    except AgentSpecDisabledError as exc:
        print(f"Agent disabled: {exc.comment or exc}", file=sys.stderr)
        return 2
    except AgentSpecInvalidError as exc:
        print(f"{exc}", file=sys.stderr)
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
        non_interactive=_non_interactive,
        session_id=_session_id,
        system_prompt_supplement=_SUPPLEMENT,
        prefill=_task or "",
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
