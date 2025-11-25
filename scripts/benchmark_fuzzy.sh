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

source ~/.bashrc
conda activate mcp_app

find ./media -mindepth 1 -delete



# Model list for batch benchmarking (add/remove as needed)
experiment_names=(
  "gemini-2.5-flash"
  "claude-haiku-4-5"
  "claude-sonnet-4-5"
  "gemini-2.5-pro"
  "meta-llama/Llama-4-Scout-17B-16E-Instruct"
  "internvl3.5-latest"
  "glm-4.5v"
  "gpt-5"
  "gpt-5-nano"
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
  "gemini-2.5-flash-lite"
  "grok-4-fast-reasoning"
  "grok-4-0709"
  "Qwen/Qwen2.5-VL-72B-Instruct"
)

for experiment_name in "${experiment_names[@]}"; do
  base_model_name=$(basename "$experiment_name")
  echo "[BENCH START] model=$experiment_name"

  python "$SCRIPT_DIR/benchmark_pipeline.py" \
          --MODEL_PATH "$experiment_name" \
          --TOP_TOOLS 400 \
          --max_step  6 \
          --max_concurrent 10 \
          --num_client 5 \
          --max_new_tokens 32768 \
          --image_dir "$SCRIPT_DIR/datasets" \
          --annotation_dir "$SCRIPT_DIR/json/tasks_fuzzy.json" \
          --OUTPUT_DIR ${base_model_name}_test_mcp_fuzzy.json \
          --fuzzy

  echo "[DONE] Output: results/${base_model_name}_test_mcp_fuzzy.json"
done
