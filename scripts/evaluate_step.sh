
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"

source ~/.bashrc
conda activate mcp_app


# Model list (add/remove as needed)
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

GT_PATH="$SCRIPT_DIR/json/test_mcp_GT.json"

for experiment_name in "${experiment_names[@]}"; do
  OUT_DIR="results/${experiment_name}"
  PRED_PATH="results/${experiment_name}_test_mcp_fuzzy.json"

  mkdir -p "$OUT_DIR"

  if [ ! -f "$PRED_PATH" ]; then
    echo "[SKIP] Prediction file not found: $PRED_PATH"
    continue
  fi

  echo "[EVAL START] experiment_name=$experiment_name"
  python "$SCRIPT_DIR/evaluate_trajectories.py" \
    --gt "$GT_PATH" \
    --pred "$PRED_PATH" \
    --output-dir "$OUT_DIR" \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --tau-strong 0.8\
    --tau-weak 0.6
  echo "[DONE] Output directory: $OUT_DIR"
done


python "$SCRIPT_DIR/tools/fig_step_eval_result.py"