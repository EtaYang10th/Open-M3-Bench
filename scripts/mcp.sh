#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

set -a
source "$ENV_FILE"
set +a

source ~/.bashrc
conda activate mcp_app


python "$SCRIPT_DIR/app_mm.py" --MODEL_PATH       gpt-5 \
                 --max_step         3 \
                 --max_concurrent   5 \
                 --TOP_TOOLS        400 \
                 --max_new_tokens   32768