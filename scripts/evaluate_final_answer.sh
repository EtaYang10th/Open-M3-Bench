SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

set -a
source "$ENV_FILE"
set +a


source ~/.bashrc
conda activate mcp_app

# Model list for batch evaluation (add/remove as needed)
experiment_names=(
  "gpt-5"
  "gpt-5-mini"
  "gpt-5-nano"
  "internvl3.5-latest"
  "gemini-2.5-pro"
  "gemini-2.5-flash"
  "gemini-2.5-flash-lite"
  "claude-haiku-4-5"
  "claude-sonnet-4-5"
  "Llama-4-Scout-17B-16E-Instruct"
  "Qwen2.5-VL-72B-Instruct"
  "glm-4.5v"
  "grok-4-fast-reasoning"
  "grok-4-0709"
)

# Fixed GT input path (repo-relative)
GT_PATH="$SCRIPT_DIR/json/test_mcp_GT.json"




JUDGE_MODELS_CSV="gpt-5-mini,gemini-2.5-flash,deepseek-chat,grok-4-fast-reasoning"

# Iterate model list and evaluate sequentially
for experiment_name in "${experiment_names[@]}"; do
  PRED_PATH="results/${experiment_name}_test_mcp_fuzzy.json"
  OUT_DIR="results/${experiment_name}"
  OUT_PATH="${OUT_DIR}/taskcompletion.json"

  mkdir -p "$OUT_DIR"

  if [ ! -f "$PRED_PATH" ]; then
    echo "[SKIP] Prediction file not found: $PRED_PATH"
    continue
  fi

  echo "[EVAL START] experiment_name=$experiment_name"
  python3 "$SCRIPT_DIR/evaluate_final.py" \
    --gt "$GT_PATH" \
    --pred "$PRED_PATH" \
    --judge-models "$JUDGE_MODELS_CSV" \
    --out "$OUT_PATH" \
    --num_client 5 \
    --max_new_tokens 32768
  echo "[DONE] Output: $OUT_PATH"
done