#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Activate the conda env. Override the name with M3_CONDA_ENV, skip with M3_SKIP_CONDA=1.
if [[ "${M3_SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${M3_CONDA_ENV:-mcp_app}"
fi

# Run from the repo root so all repo-relative paths below resolve.
cd "$SCRIPT_DIR"

# NOTE: do NOT wipe ./media here -- it holds the benchmark images themselves.
# To clear only the redundant working copies, use:
#   python tools/clean_media.py --dry-run   (then --apply --yes)

# Image root for the task images. Repo-relative by default; override with
#   M3_IMAGE_DIR=/path/to/images bash scripts/benchmark_fuzzy.sh
IMAGE_DIR="${M3_IMAGE_DIR:-media}"



# Model list for batch benchmarking (add/remove as needed)
experiment_names=(
  # "gemini-2.5-flash"
  # "claude-haiku-4-5"
  # "claude-sonnet-4-5"
  # "gemini-2.5-pro"
  # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  # "internvl3.5-latest"
  # "glm-4.5v"
  # "gpt-5"
  # "gpt-5-nano"
  # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
  "gemini-2.5-flash-lite"
  # "grok-4-fast-reasoning"
  # "grok-4-0709"
  # "Qwen/Qwen2.5-VL-72B-Instruct"
)

for experiment_name in "${experiment_names[@]}"; do
  base_model_name=$(basename "$experiment_name")
  echo "[BENCH START] model=$experiment_name"

  python "benchmark_pipeline.py" \
          --MODEL_PATH "$experiment_name" \
          --TOP_TOOLS 400 \
          --max_step  6 \
          --max_concurrent 10 \
          --num_client 5 \
          --max_new_tokens 32768 \
          --image_dir "$IMAGE_DIR" \
          --annotation_dir "json/test_mcp_fuzzy.json" \
          --OUTPUT_DIR ${base_model_name}_test_mcp_fuzzy.json \
          --fuzzy

  echo "[DONE] Output: results/${base_model_name}_test_mcp_fuzzy.json"
done
