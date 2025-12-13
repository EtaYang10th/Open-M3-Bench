import json
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ===== Inset pie (donut) adjustable parameters =====
# Axes-fraction bounds: [x0, y0, width, height]
l=0.75
INSET_BOUNDS = [0.28, 0.2, l, l]
# Pie appearance
INSET_STARTANGLE = 80 # start angle
INSET_WEDGE_WIDTH = 0.5 # pie chart wedge width
INSET_LABELDISTANCE = 1.1 # label distance
INSET_PCTDISTANCE = 0.85 # percentage distance
INSET_LABEL_FONTSIZE = 12 # label font size
INSET_PCT_FONTSIZE = 10 # percentage font size

# ===== Axis/labels/ticks/font sizes (adjust here) =====
AXIS_LABEL_FONTSIZE = 12  # axis label font size
X_TICK_FONTSIZE = 12      # x axis tick font size
Y_TICK_FONTSIZE = 10      # y axis tick font size
BAR_VALUE_FONTSIZE = 10   # bar top value label font size

# Shared color palette (truncated to number of labels)
COLOR_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#9FD0FF", "#C0CA33", "#8E24AA", "#F06292", "#A1887F",
    "#26C6DA", "#FFB74D", "#D4E157", "#90A4AE", "#4DB6AC",
]


def extract_server_tool_counts_from_catalog(catalog_dict):
    """
    Count how many tool functions each MCP server exposes from a catalog-like dict:
      { server: { tool_fn: {...}, ... }, ... }
    """
    counts = {}

    for server, tool_impl in catalog_dict.items():
        if not isinstance(tool_impl, dict):
            continue
        tool_count = sum(1 for _, fn_info in tool_impl.items() if isinstance(fn_info, dict))
        counts[server] = tool_count
    return counts


def plot_tools_per_server_bar(counts, output_pdf, inset_counts=None, output_png=None, png_dpi=300):
    """
    Reproduce the bar chart style from the legacy script and optionally draw a
    donut pie as an inset in the top-right of the bar axes for higher info density.

    - X labels are server names with underscores -> spaces, Title Case
    - Rotate x labels 30 degrees, right-aligned
    - Grid on y-axis only, dashed
    - Value labels on top of bars
    - If inset_counts is provided, draw a donut pie in the upper-right
    - Export vector PDF
    """
    if not counts:
        raise ValueError("No servers/tools found in input JSON.")

    # Build a DataFrame sorted by count desc, then server asc
    df = pd.DataFrame(
        [(srv, counts[srv]) for srv in counts],
        columns=["server", "tool_count"],
    ).sort_values(by=["tool_count", "server"], ascending=[False, True]).reset_index(drop=True)

    # Nicely formatted label for plotting: underscores -> spaces, Title Case
    df["server_label"] = df["server"].apply(lambda s: s.replace("_", " ").title())

    # Plot (vector PDF)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["server_label"], df["tool_count"]) 
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=BAR_VALUE_FONTSIZE,
        )
    ax.set_xlabel("MCP Server", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Number of Tools", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    # Rotate x tick labels without altering categorical tick positions
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_fontsize(X_TICK_FONTSIZE)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(Y_TICK_FONTSIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    # Optional inset donut pie in the top-right of the bar plot
    if inset_counts:
        try:
            labels, sizes = zip(*sorted(inset_counts.items(), key=lambda kv: kv[1], reverse=True))
        except ValueError:
            labels, sizes = [], []

        if sizes:
            # Bounds are in axes fraction coordinates: [x0, y0, width, height]
            # Tuned to use typical empty space at the top-right of the bar chart
            inset_ax = ax.inset_axes(INSET_BOUNDS)

            colors = COLOR_PALETTE[: len(sizes)]

            wedges, text_labels, autotexts = inset_ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                startangle=INSET_STARTANGLE,
                counterclock=False,
                autopct="%1.1f",
                pctdistance=INSET_PCTDISTANCE,
                labeldistance=INSET_LABELDISTANCE,
                wedgeprops=dict(width=INSET_WEDGE_WIDTH, edgecolor="white"),
                textprops=dict(fontsize=INSET_LABEL_FONTSIZE),
            )

            # Make wedge labels bold
            for t in text_labels:
                t.set_fontweight("bold")

            for t in autotexts:
                t.set_color("white")
                t.set_fontweight("bold")
                t.set_fontsize(INSET_PCT_FONTSIZE)

            inset_ax.set_aspect("equal")
            # Subtle frame to separate inset from bars
            for spine in inset_ax.spines.values():
                spine.set_alpha(0.3)
            inset_ax.set_facecolor("white")

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")

    # Optional PNG export (bitmap) without affecting the PDF output
    if output_png:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, format="png", dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


def extract_category_counts(data_dict):
    """
    Tally categories across all tool functions.
    Looks for keys in priority order: 'catalogry', 'category', 'catalog', 'catalogue'.
    """
    counts = Counter()

    for _, tool_impl in data_dict.items():
        if not isinstance(tool_impl, dict):
            continue
        for _, fn_info in tool_impl.items():
            if not isinstance(fn_info, dict):
                continue
            cat = (
                fn_info.get("catalogry")
                or fn_info.get("category")
                or fn_info.get("catalog")
                or fn_info.get("catalogue")
            )
            if cat:
                counts[cat] += 1

    return counts


# Removed standalone pie chart plotting per request.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_json",
        default="save/mcp_tools_with_desc.json",
        help="Path to the catalog JSON file (with tool definitions and categories).",
    )
    parser.add_argument(
        "--out_bar",
        default="save/mcp_tools_per_server.pdf",
        help="Where to save the bar chart PDF (tools per MCP).",
    )
    parser.add_argument(
        "--out_png",
        default="save/mcp_tools_per_server.png",
        help="Where to save the bar chart PNG (dpi=300).",
    )
    # Removed standalone pie output argument per request
    args = parser.parse_args()

    # Load catalog JSON (authoritative source)
    with open(args.in_json, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
    if not isinstance(catalog_data, dict):
        raise ValueError("Catalog JSON must be a dict mapping servers to tool definitions.")

    # Counts per server and category tallies from the single catalog file
    catalog_counts = extract_server_tool_counts_from_catalog(catalog_data)
    cat_counts = extract_category_counts(catalog_data)

    # Bar chart with optional inset pie (from category counts)
    plot_tools_per_server_bar(
        catalog_counts,
        args.out_bar,
        inset_counts=cat_counts if cat_counts else None,
        output_png=args.out_png,
        png_dpi=300,
    )


if __name__ == "__main__":
    main()


