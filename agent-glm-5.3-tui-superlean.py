#!/usr/bin/env python3
"""Textual TUI agent for glm-5.3 via api.z.ai, routed through the superleanai production middleware.

Same agent as agent-glm-5.3-tui.py, but instead of hitting
https://api.z.ai/api/coding/paas/v4 directly, every request goes through the
production superleanai gateway:

    client --> https://api.superleanai.com/zai/v1/chat/completions
           --> https://api.z.ai/api/coding/paas/v4/chat/completions

so the middleware debloats/blocks tools and logs usage in the production
Postgres (visible on the dashboard), like for any other profile.

The TUI authenticates to the gateway with a long-lived v2 EdDSA JWT bound
to an ``api_keys`` row in the production database (provider
``openrouter``, scope ``proxy:openrouter``; the billing bypass substitutes
the operator-configured upstream credential). The token lives in the
system keyring (service ``login2``, username ``SUPERLEAN_API_KEY_ZAI``;
env var of the same name takes precedence). If the row is revoked or
rotated, mint a replacement JWT server-side over the new jti and update
the keyring entry.

Note: the production gateway no longer accepts self-minted HS256 BYOK
tokens -- it requires v2 EdDSA JWTs (``internal/auth``), which only the
dashboard signer (holder of the production Ed25519 private key) can
issue.

No secret is ever placed in this script or on a command line.

Prerequisite: the ``zai`` profile in the production
``service/profiles/zai.json`` (deployed via the regular CD pipeline):

    {
      "target_url": "https://api.z.ai/api/coding/paas/v4",
      "path_strip_prefix": "/v1",
      "tool_strategy": "debloat"
    }

The ``agentknit`` profile (``service/profiles/agentknit.json``) is the
same upstream, restricted via ``allowed_tools`` to the agentknit toolset
(read_file, write_file, str_replace, exec_shell, nohup, nohup_query,
nohup_wait) with debloating served from the content-addressed
``tools-debloated`` cache. Select it client-side:

    SUPERLEAN_PROFILE=agentknit agent-glm-5.3-tui-superlean.py

Usage:
    agent-glm-5.3-tui-superlean.py "<task>"          # start TUI with task prefilled
    agent-glm-5.3-tui-superlean.py                  # interactive TUI
    agent-glm-5.3-tui-superlean.py --session <id>   # resume a previous trajectory
    agent-glm-5.3-tui-superlean.py --non-interactive # drop ask_user* tools
"""

import os
import sys

MODEL = "glm-5.3"
UPSTREAM_ENDPOINT = "https://api.z.ai/api/coding/paas/v4"
GATEWAY = os.environ.get("SUPERLEAN_ENDPOINT", "https://api.superleanai.com")
# The gateway profile is part of the client's routing decision, not server
# config: the profile name is a path segment of the gateway URL, so pointing
# the client at a different profile is purely a client-side change.
PROFILE_NAME = os.environ.get("SUPERLEAN_PROFILE", "zai")
KEYRING_SERVICE = "login2"
KEYRING_USERNAME = "SUPERLEAN_API_KEY_ZAI"

# z.ai caches prompt prefixes in 64-token blocks and reports
# ``cached_tokens: 0`` for any prompt too short to fill a whole block (and no
# cache field at all when the gateway answers before reaching the model), so
# declaring the floor keeps agentknit's strict cache-proof mode from failing
# closed on such responses.  Upstream value measured on 2026-09-12: glm-5.3
# reports no cache hit at 63/64/65 prompt tokens, its first hit at 66.
MIN_CACHEABLE_TOKENS = 66


def _keyring_password(service: str, username: str) -> str:
    import keyring

    value = keyring.get_password(service, username)
    if not value:
        raise RuntimeError(
            f"No '{username}' in system keyring (service '{service}')."
        )
    return value


def _gateway_token() -> str:
    """Dashboard-bound v2 JWT for the production gateway (see module docstring)."""
    override = os.environ.get(KEYRING_USERNAME)
    if override:
        return override
    return _keyring_password(KEYRING_SERVICE, KEYRING_USERNAME)


# Resume hints must point here, not at the generic agentknit CLI.
os.environ["AGENTKNIT_RESUME_COMMAND"] = sys.argv[0]

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


def _build_schema(token: str) -> dict:
    schema = agentknit.load_specification(MODEL, UPSTREAM_ENDPOINT)
    # Route through the production middleware; auth is the minted JWT,
    # supplied via env var so it never appears on a command line.
    schema["endpoint"] = f"{GATEWAY}/{PROFILE_NAME}/v1"
    schema.pop("keyring_service", None)
    schema.pop("keyring_username", None)
    schema.pop("auth", None)
    schema["key_env"] = "SUPERLEAN_PROXY_TOKEN"
    os.environ["SUPERLEAN_PROXY_TOKEN"] = token
    schema["display_name"] = f"agent-glm-5.3-tui-superlean ({MODEL})"
    # Context window measured by llmprobe (reports/glm-5.3).
    schema["context_window"] = 1048576
    # Upstream z.ai's minimum cacheable prefix (see MIN_CACHEABLE_TOKENS).
    schema["min_cacheable_tokens"] = MIN_CACHEABLE_TOKENS
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
        token = _gateway_token()
        schema = _build_schema(token)
        print(f"routed through {GATEWAY}/{PROFILE_NAME}/v1")

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
