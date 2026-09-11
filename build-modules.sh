#!/bin/sh
# Regenerate the underscore-named module copies shipped in the wheel.
# Face files use dashes (agent-glm-5.3.py) for direct execution; Python
# module names cannot, so the wheel packages underscore copies.
set -e
cd "$(dirname "$0")"
for pair in \
    "agent-glm-5.3.py agent_glm_5_3.py" \
    "agent-glm-5.3-tui.py agent_glm_5_3_tui.py" \
    "agent-glm-4.5-air.py agent_glm_4_5_air.py" \
    "agent-deepseek-v4-flash.py agent_deepseek_v4_flash.py" \
    "agent-kimi-k3.py agent_kimi_k3.py" \
    "agent-kimi-k3-tui.py agent_kimi_k3_tui.py"
do
    set -- $pair
    cp "$1" "$2"
done
