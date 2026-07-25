import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

DISPLAY_NAME_MAP = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5": "GPT-5",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-haiku-4-5": "Claude 4.5 Haiku",
    "claude-sonnet-4-5": "Claude 4.5 Sonnet",
    "grok-4-0709": "Grok 4 (0709)",
    "grok-4-fast-reasoning": "Grok-4 Fast",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
    "internvl3.5-latest": "InternVL 3.5",
    "Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B16E",
}

def main():
    # Path relative to tools/ folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "../save/step_eval_result.json")
    output_pdf = os.path.join(base_dir, "../save/metrics_tauweak_mllm_eval.pdf")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

    if not data:
        print("No data in JSON file.")
        sys.exit(1)

    df = pd.DataFrame(data)
    
    # Calculate average score
    metrics = ["Recall", "Precision", "Argument Similarity", "Step Coherence", "Order Consistency", "Merge Purity"]
    
    # Check if columns exist
    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        print(f"Warning: Missing metrics columns: {missing_metrics}")
        # Proceed with available ones? Or error?
        # Assuming 0 for missing
        for m in missing_metrics:
            df[m] = 0.0

    # Ensure numeric
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0)
        
    df['Average Score'] = df[metrics].mean(axis=1)
    
    # Filter tau
    target_taus = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    # Filter for target taus (using simple float comparison tolerance if needed, but direct usually works for these values)
    # To be safe with float matching:
    df['tau_weak_rounded'] = df['tau_weak'].round(2)
    df_filtered = df[df['tau_weak_rounded'].isin(target_taus)]
    
    if df_filtered.empty:
        print("Warning: No data found for target taus (0.4-0.9). Plotting all available data.")
        df_filtered = df
        target_taus = sorted(df['tau_weak_rounded'].unique())

    # Sort
    df_filtered = df_filtered.sort_values('tau_weak')

    # Filter out excluded models
    df_filtered = df_filtered[df_filtered['Model'] != 'gemini-2.5-flash-lite']

    # Plot
    plt.figure(figsize=(6, 3))
    
    models = df_filtered['Model'].unique()
    
    # Calculate overall average score for each model to sort them
    model_avg_scores = {}
    for model in models:
        model_data = df_filtered[df_filtered['Model'] == model]
        model_avg_scores[model] = model_data['Average Score'].mean()
        
    # Sort models by average score descending
    sorted_models = sorted(models, key=lambda x: model_avg_scores[x], reverse=True)

    y_offsets = [0.005, 0.01, 0.002, 0.005, 0.004]
    
    for idx, model in enumerate(sorted_models):
        model_data = df_filtered[df_filtered['Model'] == model]
        label = DISPLAY_NAME_MAP.get(model, model)
        plt.plot(model_data['tau_weak'], model_data['Average Score'], marker='o', label=label)
        
        # Add value annotations
        # if idx == 0:
        #     for x, y in zip(model_data['tau_weak'], model_data['Average Score']):
        #         plt.text(x, y + y_offsets[idx], f"{y:.3f}", ha='center', va='bottom', fontsize=)
        
    plt.xlabel('τ')
    plt.ylabel('Average Score')
    # plt.title('Model Performance on different threshold τ values')
    # Move legend up from lower left
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), ncol=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(target_taus, [str(t).replace('0.', '.') for t in target_taus])
    plt.ylim(0.14, 0.525)

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf')
    print(f"Plot saved to {output_pdf}")

if __name__ == "__main__":
    main()
