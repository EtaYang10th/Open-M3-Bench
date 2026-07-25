#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose one donut chart per model from the call-level analysis outputs.

Reads ``<results-root>/<model>/callanalysis.json`` (produced by
``evaluate_calls.py``) and lays the models out in a grid, one donut each,
showing the distribution over the five MCP call outcomes.

Usage
-----
    python tools/plot_call_pies.py --results-root results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Order is fixed so colors mean the same thing in every subplot.
CATEGORIES = [
    ("success", "Success"),
    ("illegal_calling", "Illegal Calling"),
    ("unknown_tool", "Unknown Tool"),
    ("invalid_arguments", "Invalid Arguments"),
    ("resource_not_found", "Resource Not Found"),
]
COLORS = ["#59A14F", "#E15759", "#B07AA1", "#F28E2B", "#4E79A7"]

DISPLAY_NAME_MAP = {
    "gpt-5": "GPT-5",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5-nano": "GPT-5 Nano",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "claude-haiku-4-5": "Claude 4.5 Haiku",
    "claude-sonnet-4-5": "Claude 4.5 Sonnet",
    "grok-4-0709": "Grok 4 (0709)",
    "grok-4-fast-reasoning": "Grok-4 Fast",
    "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
    "internvl3.5-latest": "InternVL 3.5",
    "Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B16E",
    "glm-4.5v": "GLM-4.5V",
}


def collect(results_root: Path) -> list[tuple[str, list[int]]]:
    """Return [(model, counts-in-CATEGORIES-order)] for every readable model."""
    found = []
    for path in sorted(results_root.glob("*/callanalysis.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")
            continue
        counts = [int(data.get(key, 0) or 0) for key, _ in CATEGORIES]
        if sum(counts) <= 0:
            print(f"[WARN] skip {path}: all zero")
            continue
        found.append((path.parent.name, counts))
    return found


def plot(entries, out_pdf: Path, out_png: Path | None, ncols: int = 5) -> None:
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (model, counts) in zip(axes, entries):
        ax.pie(
            counts,
            colors=COLORS,
            startangle=90,
            counterclock=False,
            autopct=lambda pct: f"{pct:.0f}" if pct >= 5 else "",
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white"),
            textprops=dict(fontsize=8, color="white", fontweight="bold"),
        )
        ax.set_aspect("equal")
        ax.set_title(DISPLAY_NAME_MAP.get(model, model), fontsize=9, fontweight="bold")
        ax.text(0, 0, str(sum(counts)), ha="center", va="center", fontsize=9)

    for ax in axes[len(entries):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=label)
        for color, (_, label) in zip(COLORS, CATEGORIES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(CATEGORIES), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"[INFO] Saved {out_pdf}")
    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved {out_png}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory holding <model>/callanalysis.json")
    parser.add_argument("--out-pdf", default="save/call_pies.pdf")
    parser.add_argument("--out-png", default="", help="Optional PNG twin (dpi=300); empty to skip")
    parser.add_argument("--ncols", type=int, default=5)
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path)

    entries = collect(resolve(args.results_root))
    if not entries:
        print("[WARN] no callanalysis.json found; nothing to plot.")
        return
    plot(entries, resolve(args.out_pdf), resolve(args.out_png) if args.out_png else None, args.ncols)


if __name__ == "__main__":
    main()
