# -*- coding: utf-8 -*-
"""
plot_drainage_counts_per_year.py

Plots the annual count of rapid lake drainage events for both AOIs on a single
combined figure.

Source: drainage_events.csv produced by analyze_drainage_events.py.

Two panels:
  (a) Full drainage events per year      (volume loss > 80 %)
  (b) Partial drainage events per year   (volume loss 40–80 %)

Within each panel, OST and PTM are shown as grouped bars side by side to
allow direct inter-site comparison.  Years with no events show zero-height bars.

Season filter
-------------
  USE_JJA = False  →  full melt season  (DOY 121–273, May–Sep)
  USE_JJA = True   →  JJA only          (DOY 152–243, Jun–Aug)

Formatted to match the Nature two-column figure style used elsewhere in this
project.
"""

from itertools import product
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ============================================================
# 1) FILE PATHS
# ============================================================

DRAINAGE_EVENTS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\drainage_events.csv"
)

OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_drainage")


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]
AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

# First character of lake_id → AOI
LAKE_PREFIX_AOI = {"P": "PTM", "C": "OST"}

YEARS = list(range(2013, 2026))

# Set True to restrict to JJA; False for full melt season
USE_JJA = False

SEASON_DOY_START = 121   # 1 May
SEASON_DOY_END   = 273   # 30 Sep

JJA_DOY_START = 152   # 1 Jun
JJA_DOY_END   = 243   # 31 Aug


# ============================================================
# 3) FIGURE / STYLE PARAMETERS  (Nature two-column format)
# ============================================================

FIG_WIDTH_IN  = 7.087   # 180 mm
FIG_HEIGHT_IN = 6.0

DPI_SCREEN = 150
DPI_SAVE   = 300

HSPACE = 0.42

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

# Bar colours — one per AOI, consistent across both panels
COLOUR_OST = "#457355"
COLOUR_PTM = "#2c577f"

BAR_WIDTH     = 0.35
BAR_ALPHA     = 0.85
BAR_EDGECOLOR = "white"
BAR_LINEWIDTH = 0.4

SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0

YEAR_LABEL_ROTATION = 45

GRID_LINESTYLE = ":"
GRID_ALPHA     = 0.45
GRID_LINEWIDTH = 0.5

Y_HEADROOM = 1.15

PANEL_LABEL_X = -0.11
PANEL_LABEL_Y =  1.03


# ============================================================
# 4) HELPERS
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


def get_doy_range() -> tuple[int, int]:
    return (JJA_DOY_START, JJA_DOY_END) if USE_JJA else (SEASON_DOY_START, SEASON_DOY_END)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        va="bottom", ha="right", clip_on=False,
    )


def configure_y_axis(ax: plt.Axes, y_max: float) -> None:
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5, min_n_ticks=3))
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
        zorder=0,
    )


def configure_x_axis(ax: plt.Axes, x_pos: np.ndarray) -> None:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(y) for y in YEARS],
        rotation=YEAR_LABEL_ROTATION,
        ha="right",
    )
    ax.set_xlim(x_pos[0] - BAR_WIDTH * 2, x_pos[-1] + BAR_WIDTH * 2)


# ============================================================
# 5) DATA LOADING & PREPARATION
# ============================================================

def load_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["post_win_dt"] = pd.to_datetime(df["post_win_start"], errors="coerce")
    df["year"]        = df["post_win_dt"].dt.year
    df["doy"]         = df["post_win_dt"].dt.dayofyear
    df["aoi"]         = df["lake_id"].str[0].map(LAKE_PREFIX_AOI)
    return df


def build_annual_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count drainage events per (year, aoi, drain_type).

    All year × AOI × drain_type combinations within YEARS are present in the
    output, with zero counts for combinations with no events.
    """
    doy_min, doy_max = get_doy_range()
    df = df[df["doy"].between(doy_min, doy_max)].copy()

    raw_counts = (
        df.groupby(["year", "aoi", "drain_type"])
        .size()
        .reset_index(name="n_events")
    )

    # Zero-fill the full year × AOI × drain_type grid
    full_grid = pd.DataFrame(
        list(product(YEARS, AOIS, ["full", "partial"])),
        columns=["year", "aoi", "drain_type"],
    )
    merged = full_grid.merge(raw_counts, on=["year", "aoi", "drain_type"], how="left")
    merged["n_events"] = merged["n_events"].fillna(0).astype(int)
    return merged


# ============================================================
# 6) FIGURE
# ============================================================

def make_figure(counts: pd.DataFrame, out_path: Path) -> None:
    x_pos        = np.arange(len(YEARS))
    season_label = "JJA" if USE_JJA else "May–Sep"

    aoi_colours = {"OST": COLOUR_OST, "PTM": COLOUR_PTM}

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=False,
    )
    fig.subplots_adjust(hspace=HSPACE)

    panels = [
        (ax_a, "full",    "(a)", r"No. of full drainages ($\Delta_{Vol} \leq$ -80%)"),
        (ax_b, "partial", "(b)", r"No. of partial drainages ($\Delta_{Vol}=$  -40% to -80%)"),
    ]

    for ax, drain_type, panel_label, ylabel in panels:
        sub   = counts[counts["drain_type"] == drain_type]
        y_max = max(float(sub["n_events"].max()) * Y_HEADROOM, 1.0)

        for i, aoi in enumerate(AOIS):
            # i=0 (OST) → shift left; i=1 (PTM) → shift right
            bar_x = x_pos + (i - 0.5) * BAR_WIDTH
            aoi_counts = sub[sub["aoi"] == aoi].set_index("year")["n_events"]
            vals = [int(aoi_counts.get(yr, 0)) for yr in YEARS]

            ax.bar(
                bar_x, vals,
                width=BAR_WIDTH,
                color=aoi_colours[aoi],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGECOLOR,
                linewidth=BAR_LINEWIDTH,
                label=AOI_NAMES.get(aoi, aoi),
                zorder=2,
            )

        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel_label)
        configure_y_axis(ax, y_max)
        configure_x_axis(ax, x_pos)

        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
        )

    fig.suptitle(
        f"Annual rapid lake drainage event counts  ·  {season_label}",
        fontsize=BASE_FONT_SIZE + 1,
        y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 7) MAIN
# ============================================================

def main() -> None:
    if not DRAINAGE_EVENTS_CSV.exists():
        raise FileNotFoundError(f"Input file not found:\n  {DRAINAGE_EVENTS_CSV}")

    apply_rcparams()

    print("Loading drainage events …")
    raw    = load_events(DRAINAGE_EVENTS_CSV)
    counts = build_annual_counts(raw)

    season_label = "JJA" if USE_JJA else "full season"
    print(f"  {len(raw)} total events")

    for aoi in AOIS:
        sub = counts[counts["aoi"] == aoi]
        n_full    = int(sub[sub["drain_type"] == "full"]["n_events"].sum())
        n_partial = int(sub[sub["drain_type"] == "partial"]["n_events"].sum())
        print(f"  {aoi} [{season_label}]: {n_full} full, {n_partial} partial")

    suffix = "_jja" if USE_JJA else "_season"
    out = OUTPUT_DIR / f"drainage_counts_per_year{suffix}.png"
    make_figure(counts, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
