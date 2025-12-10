#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, HPacker
import numpy as np
from PIL import Image

# ==== Configurable section ====
# Resolve to repo root: two levels up from this file
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = (REPO_ROOT / "results").resolve()
# Directory of brand logos
IMAGE_DIR = (REPO_ROOT / "image").resolve()
MODEL_NAMES = [
    'glm-4.5v',
    "Qwen2.5-VL-72B-Instruct",
    "internvl3.5-latest",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "gpt-5-nano",
    "Llama-4-Scout-17B-16E-Instruct",
    "grok-4-fast-reasoning",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gpt-5-mini",
    "grok-4-0709",
    "gemini-2.5-pro",
    "gpt-5",
]
# Metrics to extract (from aggregate)
METRICS = [
    "recall",
    "precision",
    "avg_sim_all_cov",
    #"avg_sim_strong_cov",
    "step_coherence_cov",
    "order_consistency_cov",
    "merge_purity_cov",
    # "completion_rate",
    # "information_grounding_rate",
]


PDF_OUT = Path("save/metrics_mllm_step_eval.pdf")
EFF_CSV_OUT = Path("save/avg_rounds_tool_calls.csv")
# ===================

# Fixed color palette (follow provided order)
PALETTE_HEX = [
    "#caf0f8",
    "#bce2ec",
    "#add4df",
    "#9fc6d3",
    "#91b7c6",
    "#82a9ba",
    "#749bad",
    "#668da1",
    "#577f95",
    "#497188",
    "#3a637c",
    "#2c546f",
    "#1e4663",
    "#0f3856",
    "#012a4a",
]

# Fixed parameters
LOGO_FIXED_WIDTH_PX = 16
LOGO_SUPER_SAMPLE_FACTOR = 4  # supersampling factor: load at higher res then shrink
LOGO_TEXT_GAP_PX = 3  # pixel gap between icon and text

# Plot tunables
FIG_WIDTH_IN = 5
BAR_HEIGHT_FACTOR = 0.35  # height per bar
FIG_HEIGHT_BASE = 1.5
TOP_K_HIGHLIGHT = 3
COLOR_TOP = "#0f3856"  # deep blue (top 3)
COLOR_OTHERS = "#bce2ec"  # light blue-gray (others)
VALUE_LABEL_OFFSET_RATIO = 0.01  # right offset ratio of value text relative to max_v
YLABEL_FONT_SIZE = 9
VALUE_FONT_SIZE = 8.5
XLABEL_FONT_SIZE = 10
TITLE_FONT_SIZE = 11
XGRID_ALPHA = 0.4 # horizontal offset relative to y-axis (points; positive = more left)
LABEL_X_OFFSET_PT = 6
SCORE_SIDE_THRESHOLD = 0.3  # avg threshold: <0.3 place name right/score left; otherwise reverse
LABEL_PAD_PT = 6  # pixel offset for name+icon relative to bar end
SCORE_PAD_RATIO = 0.01  # score text offset relative to max_v
SCORE_PRINT_THRESHOLD = 0.05  # do not print scores below this


# Map model name to brand keyword and image filename (substring match, case-insensitive)
BRAND_IMG_MAP = {
    "glm": "glm.png",
    "qwen": "qwen.png",
    "claude": "claude.png",
    "internvl": "internvl.png",
    "llama": "llama.png",
    "gemini": "gemini.png",
    "gpt": "gpt.png",
    "grok": "grok.png",
}

def find_logo_image_path(model_name: str) -> Path:
    """Match brand keyword by model name and return corresponding logo path; None if missing."""
    name_lower = model_name.lower()
    # Sort by keyword length desc to avoid short-key mismatches
    for key in sorted(BRAND_IMG_MAP.keys(), key=len, reverse=True):
        if key in name_lower:
            p = (IMAGE_DIR / BRAND_IMG_MAP[key]).resolve()
            if p.exists():
                return p
    return None

def _hex_to_rgb01(hex_color: str):
    s = hex_color.lstrip('#')
    return tuple(int(s[i:i+2], 16)/255.0 for i in (0, 2, 4))

def _rgb01_to_hex(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (
        int(max(0, min(1, r))*255),
        int(max(0, min(1, g))*255),
        int(max(0, min(1, b))*255),
    )

def lighten_color(hex_color: str, amount: float) -> str:
    """
    Mix color toward white by amount (0-1). Larger amount -> lighter.
    Ensure distinguishable from COLOR_OTHERS.
    """
    amount = max(0.0, min(1.0, amount))
    r, g, b = _hex_to_rgb01(hex_color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return _rgb01_to_hex((r, g, b))

def load_logo_resized_width(logo_path: Path, width: int = LOGO_FIXED_WIDTH_PX, super_sample_factor: int = LOGO_SUPER_SAMPLE_FACTOR) -> np.ndarray:
    """
    Resize proportionally to a fixed display width with supersampling:
    - First scale to (width * super_sample_factor) while keeping aspect ratio
    - Then zoom down to width at draw time to preserve details
    Uses PIL + LANCZOS.
    """
    img = Image.open(str(logo_path))
    if img.mode not in ("RGBA", "RGB"):
        img = img.convert("RGBA")
    w, h = img.size
    if w <= 0 or h <= 0:
        w, h = 1, 1
    # Upscale source image to higher pixels (supersampling)
    ss = max(1, int(super_sample_factor))
    new_w = int(width * ss)
    new_h = max(1, int(round(h * (new_w / float(w)))))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img)


# Model display name mapping (shorten or prettify)
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

def display_name_from_model(name: str) -> str:
    s = str(name)
    if s in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[s]
    # Default rule: replace hyphens with spaces and title-case; keep GPT/GLM uppercased
    t = s.replace("-", " ")
    t = re.sub(r"\bgpt\b", "GPT", t, flags=re.IGNORECASE)
    t = re.sub(r"\bglm\b", "GLM", t, flags=re.IGNORECASE)
    words = [w if w.isupper() else (w[:1].upper() + w[1:]) for w in t.split()]
    return " ".join(words)

def normalize_model_name(name: str) -> str:
    """
    Normalize full-width quotes/whitespace to avoid path inconsistencies.
    Keep original hyphens and dots unchanged.
    """
    s = name.strip()
    # Replace Chinese quotes with English
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Strip paired quotes if present
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    # Collapse consecutive whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def candidate_path_from_join(base_dir: Path, model: str) -> Path:
    filename = f"{model}/results_eval.json"
    return (base_dir / filename).resolve()

def search_result_file(base_dir: Path, model: str) -> Path:
    """
    If direct join missing, recursively search *{model}*results_eval.json under base_dir.
    Prefer longer path (more specific) or most recent mtime.
    """
    pattern = f"*{model}*results_eval.json"
    candidates = list(base_dir.rglob(pattern))
    if not candidates:
        return None
    # Prefer longer filenames, then newer mtime
    candidates.sort(key=lambda p: (len(str(p)), p.stat().st_mtime), reverse=True)
    return candidates[0].resolve()

def load_metrics_from_file(p: Path, metrics: List[str]) -> Dict[str, float]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    agg = data.get("aggregate", {})
    out = {}
    for m in metrics:
        val = agg.get(m, None)
        if isinstance(val, (int, float)):
            out[m] = float(val)
        else:
            out[m] = float("nan")
    return out

def load_metric_from_taskfile(base_dir: Path, model: str, key: str) -> float:
    """
    Read the given key from aggregate in base_dir/model/taskcompletion.json.
    Return NaN if missing or failed.
    """
    # First attempt the standard path
    candidate = (base_dir / model / "taskcompletion.json").resolve()
    path = None
    if candidate.exists():
        path = candidate
    else:
        # Then do a relaxed search
        pattern = f"*{model}*taskcompletion.json"
        candidates = list(base_dir.rglob(pattern))
        if candidates:
            candidates.sort(key=lambda p: (len(str(p)), p.stat().st_mtime), reverse=True)
            path = candidates[0].resolve()

    if path is None or not path.exists():
        return float("nan")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate", {})
        val = agg.get(key, None)
        return float(val) if isinstance(val, (int, float)) else float("nan")
    except Exception:
        return float("nan")

def main():
    rows = []
    missing = []
    # Also collect avg_Rounds and avg_tool_calls separately
    rows_eff = []

    for raw_name in MODEL_NAMES:
        model = normalize_model_name(raw_name)
        # Join by convention first
        path = candidate_path_from_join(BASE_DIR, model)
        print(path)
        if not path.exists():
            # Then try search
            path = search_result_file(BASE_DIR, model)

        if path is None or not path.exists():
            missing.append(model)
            continue

        try:
            mvals = load_metrics_from_file(path, METRICS)
            # Add completion_rate from taskcompletion.json
            mvals["completion_rate"] = load_metric_from_taskfile(BASE_DIR, model, "completion_rate")
            # Add information_grounding_rate from taskcompletion.json
            mvals["information_grounding_rate"] = load_metric_from_taskfile(BASE_DIR, model, "information_grounding_rate")
            rows.append({"model": model, **mvals, "_path": str(path)})

            # Read interaction cost metrics separately (not in main df, no path)
            eff_vals = load_metrics_from_file(path, ["avg_Rounds", "avg_tool_calls"])
            rows_eff.append({"model": model, **eff_vals})
        except Exception as e:
            print(f"[WARN] Failed to read: {model} @ {path} -> {e}", file=sys.stderr)
            missing.append(model)

    if not rows:
        print("[ERROR] No result files could be read. Please check BASE_DIR and file naming.")
        sys.exit(1)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Compute Average score (mean of all metrics)
    metrics_for_avg = [m for m in (METRICS + ["information_grounding_rate"]) if m in df.columns]
    if metrics_for_avg:
        df["Average score"] = df[metrics_for_avg].mean(axis=1)

    # Use mapped display names
    df["model"] = df["model"].astype(str).apply(display_name_from_model)

    # Sort by average ascending
    if "Average score" in df.columns:
        df = df.sort_values(by="Average score", ascending=True).reset_index(drop=True)

    # Save .CSV (including added info metrics and Average score)
    df.to_csv("save/step_eval_result.csv", index=False)

    # Generate separate CSV for interaction costs
    if rows_eff:
        df_eff = pd.DataFrame(rows_eff)
        # Keep desired columns only
        desired_cols = ["model", "avg_Rounds", "avg_tool_calls"]
        existing_cols = [c for c in desired_cols if c in df_eff.columns]
        df_eff = df_eff[existing_cols]
        # Use mapped display names
        df_eff["model"] = df_eff["model"].astype(str).apply(display_name_from_model)
        # Align model order to main table's average-sorted order (if exists)
        order_by_avg = df["model"].tolist()
        ordered_models_eff = [m for m in order_by_avg if m in set(df_eff["model"])]
        if ordered_models_eff:
            df_eff["model"] = pd.Categorical(df_eff["model"], categories=ordered_models_eff, ordered=True)
            df_eff = df_eff.sort_values(by="model").reset_index(drop=True)
        df_eff.to_csv(EFF_CSV_OUT, index=False)

    # Plot: show Average score only (horizontal bars, sorted by score)
    df_plot = df[["model", "Average score"]].dropna().copy()
    df_plot = df_plot.sort_values("Average score", ascending=True)
    models = df_plot["model"].tolist()
    values = df_plot["Average score"].tolist()

    fig_h = BAR_HEIGHT_FACTOR * len(models) + FIG_HEIGHT_BASE
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_h))

    # Colors: top-3 use lighter deep blue; others use light gray-blue
    colors = []
    top_start = max(0, len(models) - TOP_K_HIGHLIGHT)
    color_top_light = lighten_color(COLOR_TOP, 0.45)
    for idx in range(len(models)):
        colors.append(color_top_light if idx >= top_start else COLOR_OTHERS)

    y_pos = list(range(len(models)))
    bars = ax.barh(y_pos, values, color=colors)

    # Y-axis labels: use mapped display names; keep icon on the left
    display_names = [display_name_from_model(str(m)) for m in models]
    # Hide y-axis ticks and tick lines
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='both', left=False, right=False, length=0)

    max_v = max(values) if values else 0.0

    # Place score text based on threshold (<0.3 left side, >=0.3 right side)
    outside_dx = max_v * SCORE_PAD_RATIO
    inside_dx = max_v * SCORE_PAD_RATIO
    min_dx = max_v * 0.002
    for v, bar in zip(values, bars):
        if v < SCORE_PRINT_THRESHOLD:
            continue
        y = bar.get_y() + bar.get_height() / 2
        if v < SCORE_SIDE_THRESHOLD:
            # Score on the left side (inside)
            x_pos = max(v - inside_dx, min_dx)
            ax.text(
                x_pos,
                y,
                f"{v:.3f}",
                va="center",
                ha="right",
                fontsize=VALUE_FONT_SIZE,
                fontweight="bold",
            )
        else:
            # Score on the right side (outside)
            ax.text(
                v + outside_dx,
                y,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=VALUE_FONT_SIZE,
                fontweight="bold",
            )

    # Draw name and icon at bar end: side decided by threshold
    for i, (model, disp_name, v) in enumerate(zip(models, display_names, values)):
        y_center = bars[i].get_y() + bars[i].get_height() / 2
        logo_path = find_logo_image_path(str(model))
        try:
            image_box = None
            if logo_path is not None:
                img_arr = load_logo_resized_width(logo_path, width=LOGO_FIXED_WIDTH_PX, super_sample_factor=LOGO_SUPER_SAMPLE_FACTOR)
                display_zoom = float(LOGO_FIXED_WIDTH_PX) / float(max(1, img_arr.shape[1]))
                image_box = OffsetImage(img_arr, zoom=display_zoom)
            # Text alignment depends on side
            text_align = 'left' if v < SCORE_SIDE_THRESHOLD else 'right'
            text_area = TextArea(disp_name, textprops=dict(size=YLABEL_FONT_SIZE, ha=text_align, va='center'))
            if image_box is not None:
                if v < SCORE_SIDE_THRESHOLD:
                    # Icon left, text right (placed on right side of endpoint)
                    hbox = HPacker(children=[image_box, text_area], align="center", pad=0, sep=LOGO_TEXT_GAP_PX)
                else:
                    # On the left side, keep icon left and text right; pack then right-align
                    hbox = HPacker(children=[image_box, text_area], align="center", pad=0, sep=LOGO_TEXT_GAP_PX)
            else:
                hbox = text_area

            if v < SCORE_SIDE_THRESHOLD:
                # Name+icon on the right side of endpoint
                ab = AnnotationBbox(
                    hbox,
                    (v, y_center),
                    xycoords='data',
                    xybox=(LABEL_PAD_PT, 0),
                    boxcoords='offset points',
                    frameon=False,
                    box_alignment=(0.0, 0.5),
                    zorder=5,
                )
            else:
                # Name+icon on the left side of endpoint
                ab = AnnotationBbox(
                    hbox,
                    (v, y_center),
                    xycoords='data',
                    xybox=(-LABEL_PAD_PT, 0),
                    boxcoords='offset points',
                    frameon=False,
                    box_alignment=(1.0, 0.5),
                    zorder=5,
                )
            ab.set_clip_on(False)
            ax.add_artist(ab)
        except Exception:
            # Fallback: place text on the corresponding side at least
            if v < SCORE_SIDE_THRESHOLD:
                ax.text(
                    v + max_v * SCORE_PAD_RATIO,
                    y_center,
                    disp_name,
                    va='center',
                    ha='left',
                    fontsize=YLABEL_FONT_SIZE,
                    clip_on=False,
                    zorder=5,
                )
            else:
                ax.text(
                    max(v - max_v * SCORE_PAD_RATIO, max_v * 0.002),
                    y_center,
                    disp_name,
                    va='center',
                    ha='right',
                    fontsize=YLABEL_FONT_SIZE,
                    clip_on=False,
                    zorder=5,
                )

    # Styles
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # # Hide x-axis (axis, ticks, labels)
    # ax.spines["bottom"].set_visible(False)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # Draw x-direction grid only
    ax.grid(axis="x", linestyle="--", alpha=XGRID_ALPHA)
    # Hide x-axis title
    ax.set_xlabel("Average score", fontsize=XLABEL_FONT_SIZE)
    ax.set_xlim(0, 0.55)
    # ax.set_title("M³-Bench Step-level Evaluation (Avg. score)", fontsize=TITLE_FONT_SIZE, pad=10)

    plt.tight_layout()
    plt.savefig(PDF_OUT, dpi=300)

    if missing:
        print("[INFO] Models missing results_eval.json:", ", ".join(missing))

if __name__ == "__main__":
    main()
