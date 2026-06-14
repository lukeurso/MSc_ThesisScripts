# -*- coding: utf-8 -*-
"""
plot_volume_sensitivity_2x2.py

2×2 small-multiples sensitivity figure for the Pope et al. (2016)
depth/volume model.

Layout:
          OST                        PTM
  ┌────────────────────┐   ┌────────────────────┐
  │ (a) Depth          │   │ (b) Depth          │   row 0
  └────────────────────┘   └────────────────────┘
  ┌────────────────────┐   ┌────────────────────┐
  │ (c) Volume         │   │ (d) Volume         │   row 1
  └────────────────────┘   └────────────────────┘

Each panel shows the four sensitivity curves (g, Ad_red, Ad_pan, Ad_both).
Separating AOIs into columns removes the solid/dashed encoding of the
combined figure and makes OST ↔ PTM comparison direct. Y-axes are shared
within each row so curve magnitudes are immediately comparable.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# ============================================================
# 1) FILE PATHS
# ============================================================

OST_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\volume_sensitivity"
    r"\ost_sensitivity_g_Ad.csv"
)
PTM_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\volume_sensitivity"
    r"\ptm_sensitivity_g_Ad.csv"
)

OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\volume_sensitivity")


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

PARAMS = [
    ("g",       "#1f77b4", r"$g$  (attenuation)"),
    ("Ad_red",  "#d62728", r"$A_{d,\mathrm{red}}$"),
    ("Ad_pan",  "#2ca02c", r"$A_{d,\mathrm{pan}}$"),
    ("Ad_both", "#ff7f0e", r"$A_{d,\mathrm{red}}$ & $A_{d,\mathrm{pan}}$"),
]

# Row order: (metric column, row title)
METRICS = [
    ("depth_pct_chg", "% change in mean lake depth"),
    ("vol_pct_chg",   "% change in total lake volume"),
]

# Column order: AOI keys
AOIS = ["OST", "PTM"]

# Panel labels in reading order (row-major)
PANEL_LABELS = [("a", "b"), ("c", "d")]


# ============================================================
# 3) FIGURE / STYLE PARAMETERS  (Nature two-column format)
# ============================================================

FIG_WIDTH_IN  = 7.087   # 180 mm — Nature two-column width
FIG_HEIGHT_IN = 6.8

DPI_SCREEN = 150
DPI_SAVE   = 300

HSPACE = 0.42
WSPACE = 0.28

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

LINE_WIDTH        = 1.5
MARKER_STYLE      = "o"
MARKER_SIZE       = 3.0
MARKER_EDGE_WIDTH = 0.4
MARKER_EDGE_COLOR = "white"

SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0

GRID_LINESTYLE = ":"
GRID_ALPHA     = 0.40
GRID_LINEWIDTH = 0.5

ZERO_LINE_COLOR = "#555555"
ZERO_LINE_WIDTH = 0.6
ZERO_LINE_STYLE = ":"

PANEL_LABEL_X = -0.14
PANEL_LABEL_Y =  1.03


# ============================================================
# 4) DATA LOADING & AGGREGATION
# ============================================================

def load_sensitivity_csvs() -> pd.DataFrame:
    dfs = []
    for aoi, path in (("OST", OST_CSV), ("PTM", PTM_CSV)):
        if not path.exists():
            raise FileNotFoundError(f"Sensitivity CSV not found:\n  {path}")
        df = pd.read_csv(path)
        df["aoi"] = aoi
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_mean_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby(["aoi", "param", "perturb_pct"])[["depth_pct_chg", "vol_pct_chg"]]
        .mean()
        .reset_index()
    )
    return avg.sort_values(["aoi", "param", "perturb_pct"])


# ============================================================
# 5) STYLE HELPERS
# ============================================================

def apply_rcparams() -> None:
    mpl.rcParams.update({
        "font.family":           FONT_FAMILY,
        "font.size":             BASE_FONT_SIZE,
        "axes.labelsize":        LABEL_FONT_SIZE,
        "axes.titlesize":        LABEL_FONT_SIZE,
        "xtick.labelsize":       TICK_FONT_SIZE,
        "ytick.labelsize":       TICK_FONT_SIZE,
        "legend.fontsize":       LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth":        SPINE_WIDTH,
        "xtick.major.width":     TICK_WIDTH,
        "ytick.major.width":     TICK_WIDTH,
        "xtick.major.size":      TICK_LENGTH,
        "ytick.major.size":      TICK_LENGTH,
        "xtick.minor.visible":   False,
        "ytick.minor.visible":   False,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "figure.dpi":            DPI_SCREEN,
        "savefig.dpi":           DPI_SAVE,
    })


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, f"({letter})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def configure_panel(ax: plt.Axes, x_range: float) -> None:
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH,
               linestyle=ZERO_LINE_STYLE, zorder=1)
    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH,
               linestyle=ZERO_LINE_STYLE, zorder=1)
    ax.set_xlim(-x_range, x_range)
    ax.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH, zorder=0)
    ax.grid(axis="x", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH, zorder=0)


# ============================================================
# 6) PLOTTING
# ============================================================

def plot_panel(ax: plt.Axes, avg: pd.DataFrame, aoi: str, metric: str) -> None:
    sub = avg[avg["aoi"] == aoi]
    for param, color, _ in PARAMS:
        data = sub[sub["param"] == param].sort_values("perturb_pct")
        if data.empty:
            continue
        ax.plot(
            data["perturb_pct"],
            data[metric],
            color=color,
            linewidth=LINE_WIDTH,
            marker=MARKER_STYLE,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markeredgecolor=MARKER_EDGE_COLOR,
            zorder=3,
        )


def build_legend_handles() -> list:
    return [
        mlines.Line2D(
            [], [],
            color=color,
            linewidth=LINE_WIDTH,
            marker=MARKER_STYLE,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markeredgecolor=MARKER_EDGE_COLOR,
            label=label,
        )
        for _, color, label in PARAMS
    ]


def make_figure(avg: pd.DataFrame, x_range: float) -> None:
    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    for row, (metric, ylabel) in enumerate(METRICS):
        for col, aoi in enumerate(AOIS):
            ax = axes[row, col]
            plot_panel(ax, avg, aoi, metric)
            configure_panel(ax, x_range)
            add_panel_label(ax, PANEL_LABELS[row][col])

            # Column header on top row only
            if row == 0:
                ax.set_title(
                    f"{aoi}  —  {AOI_NAMES[aoi]}",
                    fontsize=LABEL_FONT_SIZE,
                    pad=4,
                )

            # Y-axis label on left column only (shared axis hides right tick labels)
            if col == 0:
                ax.set_ylabel(ylabel)

            # X-axis label on bottom row only
            if row == len(METRICS) - 1:
                ax.set_xlabel("Parameter perturbation (%)")

    # Shared legend below the figure
    handles = build_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(PARAMS),
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=2.2,
        columnspacing=1.2,
    )

    fig.suptitle(
        "Depth and volume sensitivity to model parameter perturbation",
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=1.01,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "volume_sensitivity_2x2.png"
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 7) MAIN
# ============================================================

def main() -> None:
    apply_rcparams()

    print("Loading sensitivity CSVs …")
    df = load_sensitivity_csvs()
    print(f"  {len(df):,} rows  |  AOIs: {sorted(df['aoi'].unique())}  "
          f"|  params: {sorted(df['param'].unique())}")

    print("Averaging across lakes and images …")
    avg = compute_mean_sensitivity(df)

    x_range = float(df["perturb_pct"].abs().max())
    print(f"  Perturbation range: ±{x_range:.0f}%")

    print("Building figure …")
    make_figure(avg, x_range)

    print("Done.")


if __name__ == "__main__":
    main()
