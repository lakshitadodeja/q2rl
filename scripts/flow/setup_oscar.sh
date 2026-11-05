#!/bin/bash
# Run from the q2rl-flow project root on Oscar (not inside the old q2rl env).
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'This setup is for Oscar/Linux.'; exit 1; }
command -v uv >/dev/null || { echo 'uv is required; use the uv installed for the earlier experiment.'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.10 .venv
fi
uv pip install --python .venv/bin/python -r scripts/flow/requirements-oscar.txt -e ./robomimic
uv pip check --python .venv/bin/python
