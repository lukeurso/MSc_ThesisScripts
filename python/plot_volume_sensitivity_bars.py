# -*- coding: utf-8 -*-
"""
plot_volume_sensitivity_bars.py

Tornado-bar sensitivity figure for the Pope et al. (2016) depth/volume model.

For each parameter, the bar spans the full output range achieved when
perturbing that parameter from -MAX% to +MAX% (i.e., the bar width equals
the total sensitivity range).  Parameters are ranked top-to-bottom by
decreasing sensitivity range (averaged across the two AOIs), so the most
influential parameter always appears at the top.

Layout:
  ┌────────────────────┐   ┌────────────────────┐
  │ (a) Depth          │   │ (b) Volume         │
  └────────────────────┘   └────────────────────┘

Within each panel, two bars per parameter (one per AOI) are drawn as
horizontally offset sub-rows, coloured by AOI.  A vertical reference line
at x = 0 marks the baseline (no change).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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

# AOI bar colours (distinct from the param colours used in line figures)
AOI_COLOURS = {
    "OST": "#2166AC",   # blue
    "PTM": "#b2182b",   # red
}

AOI_ORDER = ["OST", "PTM"]

# Display labels for each parameter value in the CSV
PARAM_LABELS = {
    "g":       r"$g$  (attenuation)",
    "Ad_red":  r"$A_{d,\mathrm{red}}$",
    "Ad_pan":  r"$A_{d,\mathrm{pan}}$",
    "Ad_both": r"$A_{d,\mathrm{red}}$ & $A_{d,\mathrm{pan}}$",
}

METRICS = [
    ("depth_pct_chg", "% change in mean lake depth",    "(a)  Depth sensitivity"),
    ("vol_pct_chg",   "% change in total lake volume",  "(b)  Volume sensitivity"),
]


# ============================================================
# 3) FIGURE / STYLE PARAMETERS  (two-column format)
# ============================================================

FIG_WIDTH_IN  = 7.087
FIG_HEIGHT_IN = 4.2

DPI_SCREEN = 150
DPI_SAVE   = 300

WSPACE = 0.35

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

BAR_HEIGHT       = 0.28    # height of each individual AOI bar
BAR_AOI_OFFSET   = 0.31   # vertical distance between the two AOI bars per param
BAR_ALPHA        = 0.85
BAR_EDGE_WIDTH   = 0.5
BAR_EDGE_COLOR   = "white"

SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0

GRID_ALPHA     = 0.35
GRID_LINEWIDTH = 0.5
GRID_LINESTYLE = ":"

ZERO_LINE_COLOR = "#333333"
ZERO_LINE_WIDTH = 0.8
ZERO_LINE_STYLE = "-"

PANEL_LABEL_X = -0.02
PANEL_LABEL_Y =  1.04


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
    return (
        df.groupby(["aoi", "param", "perturb_pct"])[["depth_pct_chg", "vol_pct_chg"]]
        .mean()
        .reset_index()
        .sort_values(["aoi", "param", "perturb_pct"])
    )


def compute_extremes(avg: pd.DataFrame) -> pd.DataFrame:
    """
    For each (aoi, param), extract the output values at the most negative
    and most positive perturbation steps.  Returns a DataFrame with columns:
        aoi, param,
        depth_neg, depth_pos, depth_range,
        vol_neg,   vol_pos,   vol_range
    """
    def _row_extremes(g: pd.DataFrame) -> pd.Series:
        lo = g.loc[g["perturb_pct"].idxmin()]
        hi = g.loc[g["perturb_pct"].idxmax()]
        return pd.Series({
            "depth_neg":   lo["depth_pct_chg"],
            "depth_pos":   hi["depth_pct_chg"],
            "depth_range": abs(hi["depth_pct_chg"] - lo["depth_pct_chg"]),
            "vol_neg":     lo["vol_pct_chg"],
            "vol_pos":     hi["vol_pct_chg"],
            "vol_range":   abs(hi["vol_pct_chg"] - lo["vol_pct_chg"]),
        })

    return (
        avg.groupby(["aoi", "param"])
        .apply(_row_extremes)
        .reset_index()
    )


def rank_params(extremes: pd.DataFrame, range_col: str) -> list:
    """Return params ordered from least to most sensitive (bottom → top of chart)."""
    mean_range = (
        extremes.groupby("param")[range_col]
        .mean()
        .sort_values(ascending=True)   # ascending → least sensitive at index 0 (bottom)
    )
    return mean_range.index.tolist()


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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


# ============================================================
# 6) PLOTTING
# ============================================================

def plot_tornado_panel(
    ax: plt.Axes,
    extremes: pd.DataFrame,
    param_order: list,
    neg_col: str,
    pos_col: str,
) -> None:
    """
    Draw tornado bars onto ax.

    Parameters are placed along the y-axis in param_order (index 0 = bottom).
    Within each parameter row, two sub-rows (one per AOI) are drawn with a
    small vertical offset.  Each bar spans [neg_val, pos_val].
    """
    n_params = len(param_order)
    # Spacing: each param occupies 1 unit; AOI sub-rows are centred within it
    aoi_offsets = {
        AOI_ORDER[0]: +BAR_AOI_OFFSET / 2,
        AOI_ORDER[1]: -BAR_AOI_OFFSET / 2,
    }

    for p_idx, param in enumerate(param_order):
        y_base = p_idx  # centre of the param row

        for aoi in AOI_ORDER:
            row = extremes[(extremes["param"] == param) & (extremes["aoi"] == aoi)]
            if row.empty:
                continue
            neg_val = float(row[neg_col].iloc[0])
            pos_val = float(row[pos_col].iloc[0])
            left    = min(neg_val, pos_val)
            width   = abs(pos_val - neg_val)
            y_pos   = y_base + aoi_offsets[aoi]

            ax.barh(
                y_pos,
                width,
                left=left,
                height=BAR_HEIGHT,
                color=AOI_COLOURS[aoi],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )

            # Annotate bar ends with the actual value at ±MAX%
            offset_pts = 2
            for val, ha in ((neg_val, "right"), (pos_val, "left")):
                nudge = -offset_pts if ha == "right" else offset_pts
                ax.annotate(
                    f"{val:+.1f}%",
                    xy=(val, y_pos),
                    xytext=(nudge, 0),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=5,
                    color=AOI_COLOURS[aoi],
                )

    # Y-axis: param labels at integer positions
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(
        [PARAM_LABELS.get(p, p) for p in param_order],
        fontsize=LABEL_FONT_SIZE,
    )
    ax.set_ylim(-0.7, n_params - 0.3)

    # Zero baseline
    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH,
               linestyle=ZERO_LINE_STYLE, zorder=2)

    ax.grid(axis="x", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def make_figure(extremes: pd.DataFrame) -> None:
    # Use depth range for ranking (same order applied to both panels)
    param_order = rank_params(extremes, "depth_range")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
    )
    fig.subplots_adjust(wspace=WSPACE)

    for ax, (metric_neg, metric_pos, xlabel, panel_title) in zip(
        (ax_a, ax_b),
        [
            ("depth_neg", "depth_pos", "% change in mean lake depth",   METRICS[0][2]),
            ("vol_neg",   "vol_pos",   "% change in total lake volume",  METRICS[1][2]),
        ],
    ):
        plot_tornado_panel(ax, extremes, param_order, metric_neg, metric_pos)
        ax.set_xlabel(xlabel)
        add_panel_label(ax, panel_title)

    # Legend
    handles = [
        mpatches.Patch(
            color=AOI_COLOURS[aoi],
            alpha=BAR_ALPHA,
            label=f"{aoi}  ({AOI_NAMES[aoi]})",
        )
        for aoi in AOI_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.suptitle(
        "Sensitivity range at ±{pct:.0f}% parameter perturbation  "
        "(parameters ranked by mean sensitivity across AOIs)".format(
            pct=extremes["depth_range"].max()   # proxy for MAX_PERTURB_PCT
        ),
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=1.02,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "volume_sensitivity_bars.png"
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

    print("Computing sensitivity extremes …")
    extremes = compute_extremes(avg)

    print("Building figure …")
    make_figure(extremes)

    print("Done.")


if __name__ == "__main__":
    main()
