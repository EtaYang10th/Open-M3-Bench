#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

set -a
source "$ENV_FILE"
set +a

if [[ "${M3_SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${M3_CONDA_ENV:-mcp_app}"
fi

cd "$SCRIPT_DIR"

python "app_mm.py" --MODEL_PATH       gpt-5 \
                 --max_step         3 \
                 --max_concurrent   5 \
                 --TOP_TOOLS        400 \
                 --max_new_tokens   32768