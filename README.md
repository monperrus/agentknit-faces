# agentknit-faces

Thin launchers ("faces") that run [agentknit](https://pypi.org/project/agentknit/)
against a specific LLM, provider, or backend. Each face hardcodes a model
endpoint and forwards everything else (REPL, tool loop, session storage,
`--session`, ...) to agentknit itself.

## Install & run

```bash
uvx agentknit-glm-5.3 "<task>"          # one-shot, fresh conversation
uvx agentknit-glm-5.3                   # interactive REPL
```

`uvx` pulls the `agentknit-faces` package from PyPI on demand. TUI faces need
the `tui` extra:

```bash
uvx --from "agentknit-faces[tui]" agentknit-glm-5.3-tui
```

## Faces shipped on PyPI

| Command | Backend |
|---|---|
| `agentknit-glm-5.3` | glm-5.3 via api.z.ai |
| `agentknit-glm-5.3-tui` | glm-5.3 via api.z.ai, Textual TUI |
| `agentknit-glm-4.5-air` | glm-4.5-air via api.z.ai |
| `agentknit-deepseek-v4-flash` | DeepSeek V4 Flash via the official DeepSeek API (`api.deepseek.com`) |
| `agentknit-kimi-k3` | Kimi K3 via the official Kimi Coding Plan API |
| `agentknit-kimi-k3-tui` | Kimi K3 via the official Kimi Coding Plan API, Textual TUI |

API keys are resolved from the system keyring (never from source-controlled
env files):

| Face | keyring service | keyring username |
|---|---|---|
| glm faces | `z.ai` | `api_key` |
| deepseek faces | `login2` | `deepseek_api_key` |
| kimi faces | `login2` | `kimi_api_key` |

### Minimum cacheable prompt size

Providers only report cache accounting above a floor, and agentknit's strict
cache-proof mode needs to know it (`min_cacheable_tokens`) so that a
legitimately uncacheable short prompt is not reported as a caching failure.
The faces declare the floor in their schema:

| Backend | Floor | Basis |
|---|---|---|
| z.ai (`glm-5.3`, `glm-4.5-air`) | 66 prompt tokens | measured: prefixes are cached in 64-token blocks; prompt 63/64/65 → `cached_tokens: 0`, prompt 66 → `cached_tokens: 64` |
| Kimi Coding Plan (`k3`) | 1024 prompt tokens | conservative estimate (floor not published; fields appear from ~3k prompt tokens) |

See `tests/test_min_cacheable_tokens.py` for the measurement notes and the
regression tests.

## Usage

```bash
agentknit-glm-5.3 "<task>"                  # one-shot, fresh conversation
agentknit-glm-5.3                           # interactive REPL
agentknit-glm-5.3 --session <id> "<task>"   # resume a previous conversation
echo "<task>" | agentknit-glm-5.3           # task piped via stdin
```

## Local-only faces (not on PyPI)

The repo also hosts faces with machine-specific paths (local checkouts,
subprocess completion scripts, internal gateways). They stay out of the
wheel and are symlinked into `~/bin` locally:

| Script | Backend |
|---|---|
| `agent-glm-5.3-tui-superlean.py` | glm-5.3 TUI routed through the superleanai production middleware |
| `agent-kimi-k3-tui-superlean.py` | Kimi K3 TUI routed through the superleanai middleware |
| `agent-gpt-5.6-luna-copilot.py` | GitHub Copilot GPT-5.6 Luna (needs `~/bin/copilot-gpt-5.6-luna.py`) |
| `agent-deepseek-v4-flash-zen.py` / `-zen-sync.py` / `-zen-async.py` | DeepSeek V4 Flash (free) via the Zen backend |
| `agent-deepseek-v4-flash-async.py` | Abstract async agent loop, parametrised by model |
| `agent-claude-haiku-async.py` | Claude Haiku (via claude.ai OAuth), agent-workflow wrapped |

## Development

Face files use dashes (`agent-glm-5.3.py`) for direct execution; the wheel
packages underscore-named copies. Regenerate them before building:

```bash
./build-modules.sh
python -m build --wheel
```
