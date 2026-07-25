#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"

if [[ "${M3_SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${M3_CONDA_ENV:-mcp_app}"
fi


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

# Move to script dir root to ensure relative paths work
cd "$SCRIPT_DIR"

# TAU_WEAKS can be:
# - array: TAU_WEAKS=(0.95 1.0)
# - single value: TAU_WEAKS=0.6 (or TAU_WEAKS=(0.6))
TAU_WEAKS=(0.6)
num_taus="${#TAU_WEAKS[@]}"
METRICS_FILE="results_cv/step_eval_result.json"
TEMP_DIR="results_cv/temp_metrics"
mkdir -p "$TEMP_DIR"
rm -f "$TEMP_DIR"/*.json

mkdir -p "$(dirname "$METRICS_FILE")"

GT_PATH="json/test_mcp_GT.json"

for tau in "${TAU_WEAKS[@]}"; do
  for experiment_name in "${experiment_names[@]}"; do
    if (( num_taus > 1 )); then
      OUT_DIR="results_cv/${experiment_name}/tau_${tau}"
    else
      OUT_DIR="results_cv/${experiment_name}"
    fi
    # Assuming predictions are in the same location as original script
    PRED_PATH="results/${experiment_name}_test_mcp_fuzzy.json"
    TEMP_METRICS_FILE="${TEMP_DIR}/${experiment_name}_tau_${tau}.json"

    mkdir -p "$OUT_DIR"

    if [ ! -f "$PRED_PATH" ]; then
      echo "[SKIP] Prediction file not found: $PRED_PATH"
      continue
    fi

    echo "[EVAL START] experiment_name=$experiment_name, tau=$tau"
    
    python "evaluate_cv_issues.py" \
      --gt "$GT_PATH" \
      --pred "$PRED_PATH" \
      --output-dir "$OUT_DIR" \
      --model BAAI/bge-small-en-v1.5 \
      --tau-strong 0.8 \
      --tau-weak "$tau" \
      --experiment-name "$experiment_name" \
      --save-metrics-to "$TEMP_METRICS_FILE"
    echo "[DONE] Output directory: $OUT_DIR"
  done
done

echo "Merging results to $METRICS_FILE..."
python -c "import json, glob, os; 
files = glob.glob('${TEMP_DIR}/*.json'); 
data = []; 
for f in files: 
    try: 
        data.extend(json.load(open(f))) 
    except Exception as e: 
        print(f'Error reading {f}: {e}'); 
print(json.dumps(data, indent=2))" > "$METRICS_FILE"

echo "Evaluation complete. Results in $METRICS_FILE"
