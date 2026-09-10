# agentknit-faces

Thin launcher scripts ("faces") that run [agentknit](https://pypi.org/project/agentknit/)
against a specific LLM, provider, or backend. Each script hardcodes a model
endpoint and forwards everything else (REPL, tool loop, session storage,
`--session`, `--reprobe`, ...) to agentknit itself.

Symlinked into `~/bin` so each launcher is runnable by name, e.g. `agent-glm-5.3 "<task>"`.

## Launchers

| Script | Backend |
|---|---|
| `agent-deepseek-v4-flash-async.py` | Abstract async agent loop, parametrised by model — imported by the other DeepSeek launchers |
| `agent-deepseek-v4-flash-haiku.py` | DeepSeek V4 Flash via a Claude Haiku backend |
| `agent-deepseek-v4-flash-zen.py` / `-zen-sync.py` | DeepSeek V4 Flash (free) via the Zen backend |
| `agent-deepseek-v4-flash-zen-async.py` | DeepSeek V4 Flash via opencode.ai (Zen backend), agent-workflow wrapped |
| `agent-glm-4.5-air.py` | glm-4.5-air via api.z.ai |
| `agent-glm-5.3.py` | glm-5.3 via api.z.ai |
| `agent-glm-5.3-tui.py` | glm-5.3 via api.z.ai, Textual TUI |
| `agent-glm-5.3-tui-superlean.py` | glm-5.3 via api.z.ai, TUI routed through the superleanai production middleware |
| `agent-gpt-5.6-luna-copilot.py` | GitHub Copilot GPT-5.6 Luna |
| `agent-kimi-k3.py` | Kimi K3 via the official Kimi Coding Plan API |
| `agent-kimi-k3-tui.py` | Kimi K3 via the official Kimi Coding Plan API, Textual TUI |
| `agent-kimi-k3-tui-superlean.py` | Kimi K3, TUI routed through the superleanai middleware |

Model wrapper `.py` files not built on agentknit (e.g. `agent-task.py`) live
elsewhere; launchers already symlinked from `~/bin` into a separate `agents/`
repo are also out of scope here.

## Usage

```bash
agent-glm-5.3 "<task>"                  # one-shot, fresh conversation
agent-glm-5.3                           # interactive REPL
agent-glm-5.3 --session <id> "<task>"   # resume a previous conversation
echo "<task>" | agent-glm-5.3           # task piped via stdin
```
