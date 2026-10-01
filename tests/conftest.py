"""Test isolation: faces read their endpoint from inference-db.

Point ``$INFERENCE_DB`` at a minimal fixture database so tests never read the
developer's ``~/.config/inference-db`` (and CI, which has none, still works).
Keys resolve from env vars that tests never set; faces under test get a stub
client anyway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DB = """
[endpoints.kimi-coding]
name = "Kimi Coding Plan"
url = "https://api.kimi.com/coding/v1"
type = "chat-completions"
[[endpoints.kimi-coding.key]]
source = "env"
var = "FACES_TEST_KEY"
[endpoints.kimi-coding.model_params.k3]
temperature = 1

[endpoints.deepseek]
name = "DeepSeek"
url = "https://api.deepseek.com/v1"
type = "chat-completions"
[[endpoints.deepseek.key]]
source = "env"
var = "FACES_TEST_KEY"

[endpoints.zai-coding]
name = "Z.ai coding"
url = "https://api.z.ai/api/coding/paas/v4"
type = "chat-completions"
[[endpoints.zai-coding.key]]
source = "env"
var = "FACES_TEST_KEY"

[endpoints.superleanai-zai]
name = "superleanai zai"
url = "https://api.superleanai.com/zai/v1"
type = "chat-completions"
[[endpoints.superleanai-zai.key]]
source = "env"
var = "FACES_TEST_KEY"

[endpoints.superleanai-kimi]
name = "superleanai kimi"
url = "https://api.superleanai.com/kimi/v1"
type = "chat-completions"
[[endpoints.superleanai-kimi.key]]
source = "env"
var = "FACES_TEST_KEY"
[endpoints.superleanai-kimi.model_params.k3]
temperature = 1
"""


@pytest.fixture(autouse=True)
def inference_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "endpoints.toml"
    path.write_text(FIXTURE_DB)
    monkeypatch.setenv("INFERENCE_DB", str(path))
    return path
