#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

set -a
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
set +a

if [[ "${M3_SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${M3_CONDA_ENV:-mcp_app}"
fi

cd "$SCRIPT_DIR"

# Model list (adjust as needed)
experiment_names=(
  "claude-haiku-4-5"
  "claude-sonnet-4-5"
  "gemini-2.5-pro"
  "gemini-2.5-flash"
  "gemini-2.5-flash-lite"
  "Llama-4-Scout-17B-16E-Instruct"
  "internvl3.5-latest"
  "glm-4.5v"
  "gpt-5"
  "gpt-5-nano"
  "gpt-5-mini"
  "grok-4-fast-reasoning"
  "grok-4-0709"
  "Qwen2.5-VL-72B-Instruct"
)


JUDGE_MODEL="gemini-2.5-flash-lite"

for experiment_name in "${experiment_names[@]}"; do
  PRED_PATH="results/${experiment_name}_test_mcp_fuzzy.json"
  OUT_DIR="results/${experiment_name}"
  OUT_PATH="${OUT_DIR}/callanalysis.json"

  mkdir -p "$OUT_DIR"

  if [ ! -f "$PRED_PATH" ]; then
    echo "[SKIP] Prediction file not found: $PRED_PATH"
    continue
  fi

  # If output file already exists, skip this run
  if [ -f "$OUT_PATH" ]; then
    echo "[SKIP] Output file already exists: $OUT_PATH"
    continue
  fi

  echo "[ANALYZE START] experiment_name=$experiment_name"
  python3 "$SCRIPT_DIR/evaluate_calls.py" \
    --pred "$PRED_PATH" \
    --out "$OUT_PATH" \
    --judge-model "$JUDGE_MODEL" \
    --num_client 10 \
    --max_new_tokens 32768
  echo "[DONE] Output: $OUT_PATH"
done


# Draw one donut per model (PDF under save/, plus a PNG for the README).
# Guarded so a missing plotting script never fails the whole evaluation run.
if [ -f "$SCRIPT_DIR/tools/plot_call_pies.py" ]; then
  echo "[SUMMARY] Plot pie charts for all models"
  python3 "$SCRIPT_DIR/tools/plot_call_pies.py" --results-root "results"
else
  echo "[SKIP] tools/plot_call_pies.py not found; skipping the summary figure."
fi


