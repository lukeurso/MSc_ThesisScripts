# -*- coding: utf-8 -*-
"""
plot_drainage_volume_by_season.py

Plots the volume of lake water lost to rapid drainage events across the melt
season, one figure per AOI (OST, PTM).

Source: drainage_events.csv produced by analyze_drainage_events.py.

For each event:
    drained_volume = pre_volume_m3 − post_volume_m3

Events are aggregated to the post-drainage observation window (summed across
all lakes with events in the same window).  Two overlaid series per panel:

  Total  : full + partial drainage volume per window   (amber)
  Full   : full drainage volume only per window        (dark red)

The full series is plotted as a shorter stem on top of the total stem, so the
amber extension above the dark-red tip represents the partial component.

Outputs (toggled via MAKE_MULTI_PANEL / MAKE_SINGLE_PANEL):
  Multi-panel : 4×4 grid, one year per panel, stem-and-marker display
  Single-panel: all years overlaid as scatter, coloured by year;
                circle = window contains a full event,
                triangle = partial-only window

One figure pair per AOI.

Season filter
-------------
  USE_JJA = False  →  full melt season  (DOY 121–273, May–Sep)
  USE_JJA = True   →  JJA only          (DOY 152–243, Jun–Aug)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


# ============================================================
# 1) FILE PATHS
# ============================================================

DRAINAGE_EVENTS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\drainage_events.csv"
)
COVERAGE_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\select_by_meltzone_coverage.csv"
)

OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_drainage")


# ============================================================
# 2) OUTPUT TOGGLES
# ============================================================

MAKE_MULTI_PANEL  = False
MAKE_SINGLE_PANEL = True


# ============================================================
# 3) DATA & SEASON SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]
AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

# First character of lake_id → AOI
LAKE_PREFIX_AOI = {"P": "PTM", "C": "OST"}

YEARS = list(range(2014, 2026))

# Set True to restrict to JJA; False for full melt season
USE_JJA = False

# Minimum melt-zone footprint coverage to include a window (Method B)
COVERAGE_THRESHOLD = 0.1

SEASON_DOY_START = 121   # 1 May
SEASON_DOY_END   = 273   # 30 Sep

JJA_DOY_START = 152   # 1 Jun
JJA_DOY_END   = 243   # 31 Aug

VOL_UNIT    = "Mm³"
VOL_DIVISOR = 1e6


# ============================================================
# 4) PLOT PARAMETERS (COMMON)
# ============================================================

DPI_SCREEN = 150
DPI_SAVE   = 300

FONT_FAMILY     = "sans-serif"
BASE_FONT_SIZE  = 7
TICK_FONT_SIZE  = 6
LABEL_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6

GRID_LINESTYLE = "--"
GRID_ALPHA     = 0.35
GRID_ZORDER    = 0

Y_HEADROOM_FACTOR = 1.15

SUPTITLE_FONTSIZE = 9

# DOY tick marks — full season
SEASON_DOY_TICKS  = [121, 152, 182, 213, 244]
SEASON_DOY_LABELS = ["May", "Jun", "Jul", "Aug", "Sep"]

# DOY tick marks — JJA
JJA_DOY_TICKS  = [152, 182, 213, 244]
JJA_DOY_LABELS = ["Jun", "Jul", "Aug", "Sep"]


# ============================================================
# 5) PLOT PARAMETERS (MULTI-PANEL)
# ============================================================

FIG_N_COLS     = 4
FIG_N_ROWS     = 4
MULTI_FIG_SIZE = (14, 12)

TOTAL_COLOUR = "#E08A3C"   # amber  — full + partial combined
FULL_COLOUR  = "#A62B1F"   # dark red — full drainage only

TOTAL_LINEWIDTH = 1.5
FULL_LINEWIDTH  = 1.5

MULTI_SCATTER_S = 30       # scatter s= value for stem-top markers
LEGEND_MARKERSIZE = 5      # Line2D markersize for legend handles

TOTAL_MARKER = "o"
FULL_MARKER  = "D"

TOTAL_STEM_ZORDER   = 2
FULL_STEM_ZORDER    = 3
TOTAL_MARKER_ZORDER = 4
FULL_MARKER_ZORDER  = 5

PANEL_TITLE_FONTSIZE  = 9
AXIS_LABEL_FONTSIZE   = 8
XTICK_FONTSIZE        = 7
YTICK_FONTSIZE        = 7
LEGEND_FONTSIZE_MULTI = 7
LEGEND_LOC            = "upper right"
LEGEND_FRAMEALPHA     = 0.8

XTICK_ROTATION = 45
XTICK_HA       = "right"

N_LABEL_X        = 0.97
N_LABEL_Y        = 0.97
N_LABEL_COLOR    = "#555555"
N_LABEL_FONTSIZE = 7


# ============================================================
# 6) PLOT PARAMETERS (SINGLE-PANEL)
# ============================================================

SINGLE_FIG_SIZE = (14, 16)
CMAP_NAME       = "cubehelix"
CMAP_VMIN       = 0.1    # colormap sample start (0–1); raise to avoid light end
CMAP_VMAX       = 0.85    # colormap sample end   (0–1); lower to avoid dark end
# Per-year color overrides — any year listed here ignores the gradient above.
# Leave empty ({}) to use the gradient for all years.
YEAR_COLORS: dict[int, str] = {2014: "#1c4e72",
    # 2024: "#ff4500",
}

SINGLE_MARKER_FULL    = "o"   # circle    — window contains ≥1 full event
SINGLE_MARKER_PARTIAL = "^"   # triangle  — partial-only window

SINGLE_SCATTER_S  = 60
SINGLE_LINEWIDTH  = 0.5
SINGLE_EDGECOLOR  = "white"

SINGLE_XTICK_FONTSIZE    = 9
SINGLE_YTICK_FONTSIZE    = 9
XLABEL_FONTSIZE          = 10
YLABEL_FONTSIZE          = 10
SINGLE_SUPTITLE_FONTSIZE = 11

SINGLE_LEGEND_LOC           = "upper right"
SINGLE_LEGEND_FONTSIZE      = 8
SINGLE_LEGEND_TITLE_FONTSIZE = 9
SINGLE_LEGEND_NCOL          = 2
SINGLE_LEGEND_FRAMEALPHA    = 0.85
SINGLE_LEGEND_EDGECOLOR     = "#aaaaaa"

ROLLING_WINDOW_SIZE = 3     # number of windows for the running mean
MAKE_ROLLING_MEAN   = True

# When True: rolling mean lines are overlaid on the scatter figure (thin, faint,
# right-side axis) and no separate rolling-mean figure is produced.
# When False: two separate figures are saved as before.
OVERLAY_ROLLING_MEAN = False
OVERLAY_ALPHA        = 0.0   # alpha for individual year rolling mean lines

# Mean-of-all-years rolling mean line (overlaid on scatter, right axis)
OVERLAY_MEAN_COLOR = "#ba00da"   # near-black; stands out against colourmap lines
OVERLAY_MEAN_LW    = 1.2
OVERLAY_MEAN_ALPHA = 0.0

# LOWESS smooth — pooled across all years, plotted on primary (left) axis
MAKE_LOWESS          = False
LOWESS_FRAC          = 0.3   # bandwidth: fraction of all points used per local fit
LOWESS_COLOR         = "#880320"
LOWESS_LW            = 1.0
LOWESS_ALPHA         = 0.0
LOWESS_WEIGHT_BY_VOL = True   # replicate points proportional to volume so high
LOWESS_WEIGHT_MAX    = 3     # values pull the fit up; max repeats per point
LOWESS_IQR_FENCE     = 6.75   # exclude points above Q3 + N*IQR before fitting
# ===========================================================
# 7) HELPERS
# ============================================================

def apply_rcparams() -> None:
    mpl.rcParams.update({
        "font.family":         FONT_FAMILY,
        "font.size":           BASE_FONT_SIZE,
        "axes.labelsize":      LABEL_FONT_SIZE,
        "axes.titlesize":      LABEL_FONT_SIZE,
        "xtick.labelsize":     TICK_FONT_SIZE,
        "ytick.labelsize":     TICK_FONT_SIZE,
        "legend.fontsize":     LEGEND_FONT_SIZE,
        "axes.linewidth":      0.6,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "figure.dpi":          DPI_SCREEN,
        "savefig.dpi":         DPI_SAVE,
    })


def get_doy_range() -> tuple[int, int]:
    return (JJA_DOY_START, JJA_DOY_END) if USE_JJA else (SEASON_DOY_START, SEASON_DOY_END)


def get_doy_ticks() -> tuple[list, list]:
    return (JJA_DOY_TICKS, JJA_DOY_LABELS) if USE_JJA else (SEASON_DOY_TICKS, SEASON_DOY_LABELS)


def _year_color(i: int, yr: int):
    if yr in YEAR_COLORS:
        return YEAR_COLORS[yr]
    cmap = plt.colormaps[CMAP_NAME]
    n    = max(len(YEARS) - 1, 1)
    t    = CMAP_VMIN + (CMAP_VMAX - CMAP_VMIN) * i / n
    return cmap(t)


# ============================================================
# 8) DATA LOADING & PREPARATION
# ============================================================

def load_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["post_win_dt"]    = pd.to_datetime(df["post_win_start"], errors="coerce")
    df["year"]           = df["post_win_dt"].dt.year
    df["doy"]            = df["post_win_dt"].dt.dayofyear
    df["aoi"]            = df["lake_id"].str[0].map(LAKE_PREFIX_AOI)
    df["drained_vol_m3"] = df["pre_volume_m3"] - df["post_volume_m3"]
    return df


def filter_by_doy(df: pd.DataFrame) -> pd.DataFrame:
    doy_min, doy_max = get_doy_range()
    return df[df["doy"].between(doy_min, doy_max)].copy()


def build_valid_windows(cov_csv: Path) -> dict[str, set]:
    """Return {aoi: set_of_valid_win_start_date_strings} above COVERAGE_THRESHOLD."""
    cov = pd.read_csv(cov_csv)
    cov["win_start"] = (
        pd.to_datetime(cov["win_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    )
    aoi_col = {"PTM": "PTM_proportion", "OST": "OST_proportion"}
    valid: dict[str, set] = {}
    for aoi, col in aoi_col.items():
        if col in cov.columns:
            mask = pd.to_numeric(cov[col], errors="coerce").fillna(0) > COVERAGE_THRESHOLD
            valid[aoi] = set(cov.loc[mask, "win_start"])
    return valid


def filter_by_coverage(df: pd.DataFrame, valid_windows: dict[str, set]) -> pd.DataFrame:
    """Keep only events whose post_win_start falls in a valid window for their AOI."""
    df = df.copy()
    df["_date"] = (
        pd.to_datetime(df["post_win_start"], errors="coerce").dt.strftime("%Y-%m-%d")
    )
    mask = df.apply(
        lambda r: r["_date"] in valid_windows.get(r["aoi"], set()), axis=1
    )
    return df[mask].drop(columns=["_date"])


def aggregate_to_windows(df_aoi: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-lake events to per-window totals for one AOI.

    Returns one row per unique post_win_start with:
      total_vol_mm3  : sum of all drained volume in that window (Mm³)
      full_vol_mm3   : sum of full-drainage-only volume in that window (Mm³)
      has_full       : True if any full event occurred in that window
      has_partial    : True if any partial event occurred in that window
      n_events       : total event count in that window
    """
    meta = df_aoi.groupby("post_win_start").agg(
        year        = ("year",          "first"),
        doy         = ("doy",           "first"),
        total_vol   = ("drained_vol_m3", "sum"),
        has_full    = ("drain_type",    lambda x: "full" in x.values),
        has_partial = ("drain_type",    lambda x: "partial" in x.values),
        n_events    = ("lake_id",       "count"),
    ).reset_index()

    full_vol = (
        df_aoi[df_aoi["drain_type"] == "full"]
        .groupby("post_win_start")["drained_vol_m3"]
        .sum()
        .rename("full_vol")
        .reset_index()
    )

    win = meta.merge(full_vol, on="post_win_start", how="left")
    win["full_vol"] = win["full_vol"].fillna(0.0)

    win["total_vol_mm3"] = win["total_vol"] / VOL_DIVISOR
    win["full_vol_mm3"]  = win["full_vol"]  / VOL_DIVISOR
    return win


def global_y_max(win_df: pd.DataFrame) -> float:
    peak = win_df["total_vol_mm3"].max()
    if not np.isfinite(peak) or peak <= 0:
        return 1.0
    return float(peak) * Y_HEADROOM_FACTOR


# ============================================================
# 9) PLOTTERS
# ============================================================

def plot_multi_panel(win_df: pd.DataFrame, aoi: str, y_max: float, out_path: Path) -> None:
    doy_min, doy_max   = get_doy_range()
    doy_ticks, doy_labels = get_doy_ticks()
    season_label       = "JJA" if USE_JJA else "May–Sep"

    fig, axes = plt.subplots(FIG_N_ROWS, FIG_N_COLS, figsize=MULTI_FIG_SIZE, squeeze=False)
    flat = axes.flatten()

    for idx, yr in enumerate(YEARS):
        ax  = flat[idx]
        sub = win_df[win_df["year"] == yr].sort_values("doy")

        if not sub.empty:
            # Total (amber) — drawn first so full (dark red) sits on top
            ax.vlines(
                sub["doy"], 0, sub["total_vol_mm3"],
                color=TOTAL_COLOUR, linewidth=TOTAL_LINEWIDTH, zorder=TOTAL_STEM_ZORDER,
            )
            ax.scatter(
                sub["doy"], sub["total_vol_mm3"],
                color=TOTAL_COLOUR, marker=TOTAL_MARKER, s=MULTI_SCATTER_S,
                zorder=TOTAL_MARKER_ZORDER,
            )

            # Full-only (dark red) — overlaid; shorter where partial also present
            sub_full = sub[sub["has_full"]]
            if not sub_full.empty:
                ax.vlines(
                    sub_full["doy"], 0, sub_full["full_vol_mm3"],
                    color=FULL_COLOUR, linewidth=FULL_LINEWIDTH, zorder=FULL_STEM_ZORDER,
                )
                ax.scatter(
                    sub_full["doy"], sub_full["full_vol_mm3"],
                    color=FULL_COLOUR, marker=FULL_MARKER, s=MULTI_SCATTER_S,
                    zorder=FULL_MARKER_ZORDER,
                )

        ax.set_title(str(yr), fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlim(doy_min - 2, doy_max + 2)
        ax.set_ylim(0, y_max)
        ax.set_xticks(doy_ticks)
        ax.set_xticklabels(
            doy_labels, fontsize=XTICK_FONTSIZE,
            rotation=XTICK_ROTATION, ha=XTICK_HA,
        )
        ax.yaxis.set_tick_params(labelsize=YTICK_FONTSIZE)

        if idx % FIG_N_COLS == 0:
            ax.set_ylabel(f"Volume drained ({VOL_UNIT})", fontsize=AXIS_LABEL_FONTSIZE)

        n_ev = int(sub["n_events"].sum()) if not sub.empty else 0
        ax.text(
            N_LABEL_X, N_LABEL_Y, f"n={n_ev}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=N_LABEL_FONTSIZE, color=N_LABEL_COLOR,
        )

    h_total = Line2D(
        [0], [0], color=TOTAL_COLOUR, linewidth=TOTAL_LINEWIDTH,
        marker=TOTAL_MARKER, markersize=LEGEND_MARKERSIZE,
        label="Total (full + partial)",
    )
    h_full = Line2D(
        [0], [0], color=FULL_COLOUR, linewidth=FULL_LINEWIDTH,
        marker=FULL_MARKER, markersize=LEGEND_MARKERSIZE,
        label="Full drainage only",
    )
    flat[0].legend(
        handles=[h_total, h_full],
        fontsize=LEGEND_FONTSIZE_MULTI, loc=LEGEND_LOC, framealpha=LEGEND_FRAMEALPHA,
    )

    for idx in range(len(YEARS), FIG_N_ROWS * FIG_N_COLS):
        flat[idx].set_visible(False)

    fig.suptitle(
        f"{AOI_NAMES.get(aoi, aoi)}  ·  {season_label}",
        fontsize=SUPTITLE_FONTSIZE,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


def plot_single_panel(win_df: pd.DataFrame, aoi: str, y_max: float, out_path: Path) -> None:
    doy_min, doy_max      = get_doy_range()
    doy_ticks, doy_labels = get_doy_ticks()
    season_label          = "JJA" if USE_JJA else "May–Sep"

    cmap = plt.colormaps[CMAP_NAME].resampled(len(YEARS))
    fig, ax = plt.subplots(figsize=SINGLE_FIG_SIZE)

    year_handles = []
    for i, yr in enumerate(YEARS):
        sub   = win_df[win_df["year"] == yr]
        color = cmap(i)
        n_ev  = int(sub["n_events"].sum()) if not sub.empty else 0

        # Proxy handle for year legend (always created so legend is complete)
        year_handles.append(
            ax.scatter([], [], color=color, marker=SINGLE_MARKER_FULL,
                       s=SINGLE_SCATTER_S, linewidths=SINGLE_LINEWIDTH,
                       edgecolors=SINGLE_EDGECOLOR,
                       label=f"{yr}  (n={n_ev})")
        )

        if sub.empty:
            continue

        # Windows with ≥1 full event
        sub_full = sub[sub["has_full"]]
        if not sub_full.empty:
            ax.scatter(
                sub_full["doy"], sub_full["total_vol_mm3"],
                color=color, marker=SINGLE_MARKER_FULL, s=SINGLE_SCATTER_S,
                linewidths=SINGLE_LINEWIDTH, edgecolors=SINGLE_EDGECOLOR,
                zorder=3,
            )

        # Windows with partial events only
        sub_partial = sub[~sub["has_full"]]
        if not sub_partial.empty:
            ax.scatter(
                sub_partial["doy"], sub_partial["total_vol_mm3"],
                color=color, marker=SINGLE_MARKER_PARTIAL, s=SINGLE_SCATTER_S,
                linewidths=SINGLE_LINEWIDTH, edgecolors=SINGLE_EDGECOLOR,
                zorder=3,
            )

    ax.set_xlim(doy_min - 2, doy_max + 2)
    ax.set_ylim(0, y_max)
    ax.set_xticks(doy_ticks)
    ax.set_xticklabels(doy_labels, fontsize=SINGLE_XTICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=SINGLE_YTICK_FONTSIZE)
    ax.set_xlabel("Day of year", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel(f"Volume drained per window ({VOL_UNIT})", fontsize=YLABEL_FONTSIZE)

    # Marker-type legend (shape meaning)
    h_full_marker = ax.scatter(
        [], [], color="grey", marker=SINGLE_MARKER_FULL, s=SINGLE_SCATTER_S,
        label="Window contains full event",
    )
    h_partial_marker = ax.scatter(
        [], [], color="grey", marker=SINGLE_MARKER_PARTIAL, s=SINGLE_SCATTER_S,
        label="Partial-only window",
    )

    leg_years = ax.legend(
        handles=year_handles,
        title="Year  (total events)",
        loc=SINGLE_LEGEND_LOC,
        fontsize=SINGLE_LEGEND_FONTSIZE,
        title_fontsize=SINGLE_LEGEND_TITLE_FONTSIZE,
        ncol=SINGLE_LEGEND_NCOL,
        framealpha=SINGLE_LEGEND_FRAMEALPHA,
        edgecolor=SINGLE_LEGEND_EDGECOLOR,
    )
    ax.add_artist(leg_years)
    ax.legend(
        handles=[h_full_marker, h_partial_marker],
        loc="upper left",
        fontsize=SINGLE_LEGEND_FONTSIZE,
        framealpha=SINGLE_LEGEND_FRAMEALPHA,
        edgecolor=SINGLE_LEGEND_EDGECOLOR,
    )

    fig.suptitle(
        f"{AOI_NAMES.get(aoi, aoi)}  ·  {season_label}",
        fontsize=SINGLE_SUPTITLE_FONTSIZE, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 10) COMBINED SINGLE-PANEL (both AOIs side by side)
# ============================================================

def plot_combined_single_panel(win_dfs: dict, shared_y_max: float, out_path: Path) -> None:
    doy_min, doy_max      = get_doy_range()
    doy_ticks, doy_labels = get_doy_ticks()
    season_label          = "JJA" if USE_JJA else "May–Sep"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    year_win_counts = {yr: 0 for yr in YEARS}

    for row, aoi in enumerate(["OST", "PTM"]):
        ax     = axes[row]
        win_df = win_dfs[aoi]

        # Strip outliers from scatter and LOWESS using the same IQR fence
        _pos  = win_df["total_vol_mm3"][win_df["total_vol_mm3"] > 0]
        if not _pos.empty:
            q1, q3  = np.percentile(_pos, [25, 75])
            fence   = q3 + LOWESS_IQR_FENCE * (q3 - q1)
            win_df  = win_df[win_df["total_vol_mm3"] <= fence]

        for yr in YEARS:
            year_win_counts[yr] += len(win_df[win_df["year"] == yr])

        for i, yr in enumerate(YEARS):
            sub   = win_df[win_df["year"] == yr]
            color = _year_color(i, yr)
            if sub.empty:
                continue
            sub_full = sub[sub["has_full"]]
            if not sub_full.empty:
                ax.scatter(
                    sub_full["doy"], sub_full["total_vol_mm3"],
                    color=color, marker=SINGLE_MARKER_FULL, s=SINGLE_SCATTER_S,
                    linewidths=SINGLE_LINEWIDTH, edgecolors=SINGLE_EDGECOLOR, zorder=3,
                )
            sub_partial = sub[~sub["has_full"]]
            if not sub_partial.empty:
                ax.scatter(
                    sub_partial["doy"], sub_partial["total_vol_mm3"],
                    color=color, marker=SINGLE_MARKER_PARTIAL, s=SINGLE_SCATTER_S,
                    linewidths=SINGLE_LINEWIDTH, edgecolors=SINGLE_EDGECOLOR, zorder=3,
                )

        if OVERLAY_ROLLING_MEAN:
            ax2 = ax.twinx()
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.patch.set_visible(False)
            ax2.spines["right"].set_visible(True)
            ax2.spines["top"].set_visible(False)

            doy_grid = np.arange(doy_min, doy_max + 1, dtype=float)
            all_interp: list[np.ndarray] = []
            for i, yr in enumerate(YEARS):
                sub_r = win_df[win_df["year"] == yr].sort_values("doy")
                if sub_r.empty:
                    continue
                color_r = _year_color(i, yr)
                rolled = (
                    sub_r["total_vol_mm3"]
                    .rolling(ROLLING_WINDOW_SIZE, center=True, min_periods=1)
                    .mean()
                )
                ax2.plot(sub_r["doy"], rolled, color=color_r, lw=0.8, alpha=OVERLAY_ALPHA, zorder=1)
                interp = np.interp(
                    doy_grid, sub_r["doy"].values, rolled.values,
                    left=np.nan, right=np.nan,
                )
                all_interp.append(interp)

            if all_interp:
                mean_curve = np.nanmean(all_interp, axis=0)
                valid = np.isfinite(mean_curve)
                ax2.plot(
                    doy_grid[valid], mean_curve[valid],
                    color=OVERLAY_MEAN_COLOR, lw=OVERLAY_MEAN_LW,
                    alpha=OVERLAY_MEAN_ALPHA, zorder=2,
                    label=f"Mean of {ROLLING_WINDOW_SIZE}-win rolling means",
                )

            ax2.set_ylim(0, None)
            ax2.set_ylabel(
                f"{ROLLING_WINDOW_SIZE}-win mean ({VOL_UNIT})",
                fontsize=YLABEL_FONTSIZE - 1,
                color="#888888",
            )
            ax2.tick_params(axis="y", labelsize=SINGLE_YTICK_FONTSIZE - 1, colors="#888888")

        if MAKE_LOWESS and not win_df.empty:
            _pool    = win_df[win_df["total_vol_mm3"] > 0]
            pool_doy = _pool["doy"].values
            pool_vol = _pool["total_vol_mm3"].values
            sort_idx = np.argsort(pool_doy)
            doy_s    = pool_doy[sort_idx]
            vol_s    = pool_vol[sort_idx]

            if LOWESS_WEIGHT_BY_VOL and vol_s.max() > 0:
                repeats = np.round(vol_s / vol_s.max() * LOWESS_WEIGHT_MAX).astype(int).clip(1)
                doy_fit = np.repeat(doy_s, repeats)
                vol_fit = np.repeat(vol_s, repeats)
            else:
                doy_fit, vol_fit = doy_s, vol_s

            doy_eval   = np.arange(doy_s.min(), doy_s.max() + 1, dtype=float)
            smoothed_y = sm_lowess(vol_fit, doy_fit, frac=LOWESS_FRAC, xvals=doy_eval)
            ax.plot(
                doy_eval, smoothed_y,
                color=LOWESS_COLOR, lw=LOWESS_LW, alpha=LOWESS_ALPHA, zorder=4,
            )

        ax.set_xlim(doy_min - 2, doy_max + 2)
        ax.set_ylim(0, global_y_max(win_df))
        ax.tick_params(axis="y", labelsize=SINGLE_YTICK_FONTSIZE)
        ax.set_ylabel(f"Volume drained per window ({VOL_UNIT})", fontsize=YLABEL_FONTSIZE)

        ax.text(0.01, 0.97, f"({chr(ord('a') + row)})", transform=ax.transAxes,
                fontsize=SINGLE_XTICK_FONTSIZE, va="top", ha="left")
        ax.set_title(AOI_NAMES.get(aoi, aoi), fontsize=XLABEL_FONTSIZE)

    # x-axis labels only on bottom panel
    axes[1].set_xticks(doy_ticks)
    axes[1].set_xticklabels(doy_labels, fontsize=SINGLE_XTICK_FONTSIZE)
    axes[1].set_xlabel("Day of year", fontsize=XLABEL_FONTSIZE)

    # Marker legend — top panel, upper left
    h_full    = Line2D([0], [0], marker=SINGLE_MARKER_FULL, color="w",
                       markerfacecolor="grey", markersize=6, label="Window with full event")
    h_partial = Line2D([0], [0], marker=SINGLE_MARKER_PARTIAL, color="w",
                       markerfacecolor="grey", markersize=6, label="Partial-only window")
    marker_handles = [h_full, h_partial]
    if MAKE_LOWESS:
        marker_handles.append(
            Line2D([0], [0], color=LOWESS_COLOR, lw=LOWESS_LW,
                   label=f"LOWESS (frac={LOWESS_FRAC})")
        )
    leg_markers = axes[0].legend(
        handles=marker_handles, loc="upper left",
        bbox_to_anchor=(0.01, 0.88), bbox_transform=axes[0].transAxes,
        fontsize=SINGLE_LEGEND_FONTSIZE, framealpha=SINGLE_LEGEND_FRAMEALPHA,
        edgecolor=SINGLE_LEGEND_EDGECOLOR,
    )
    axes[0].add_artist(leg_markers)

    # Year legend — top panel, upper right
    year_handles = [
        Line2D([0], [0], marker=SINGLE_MARKER_FULL, color="w",
               markerfacecolor=_year_color(i, yr), markersize=6,
               label=f"{yr}  (n={year_win_counts[yr]})")
        for i, yr in enumerate(YEARS)
    ]
    axes[0].legend(handles=year_handles, title="Year  (n windows)",
                   loc=SINGLE_LEGEND_LOC, fontsize=SINGLE_LEGEND_FONTSIZE,
                   title_fontsize=SINGLE_LEGEND_TITLE_FONTSIZE,
                   ncol=SINGLE_LEGEND_NCOL, framealpha=SINGLE_LEGEND_FRAMEALPHA,
                   edgecolor=SINGLE_LEGEND_EDGECOLOR)

    fig.suptitle(f"Rapid drainage volume per window  ·  {season_label}",
                 fontsize=SINGLE_SUPTITLE_FONTSIZE, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 11) ROLLING-MEAN COMBINED PANEL
# ============================================================

def plot_rolling_mean_combined(win_dfs: dict, out_path: Path) -> None:
    doy_min, doy_max      = get_doy_range()
    doy_ticks, doy_labels = get_doy_ticks()
    season_label          = "JJA" if USE_JJA else "May–Sep"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for row, aoi in enumerate(["OST", "PTM"]):
        ax     = axes[row]
        win_df = win_dfs[aoi]

        for i, yr in enumerate(YEARS):
            sub = win_df[win_df["year"] == yr].sort_values("doy")
            if sub.empty:
                continue
            color  = _year_color(i, yr)
            rolled = (
                sub["total_vol_mm3"]
                .rolling(ROLLING_WINDOW_SIZE, center=True, min_periods=1)
                .mean()
            )
            ax.plot(sub["doy"], rolled, color=color, lw=1.5, alpha=0.85)

        ax.set_xlim(doy_min - 2, doy_max + 2)
        ax.set_ylim(0, None)
        ax.tick_params(axis="y", labelsize=SINGLE_YTICK_FONTSIZE)
        ax.set_ylabel(
            f"{ROLLING_WINDOW_SIZE}-window mean vol. ({VOL_UNIT})",
            fontsize=YLABEL_FONTSIZE,
        )
        ax.text(0.01, 0.97, f"({chr(ord('a') + row)})", transform=ax.transAxes,
                fontsize=SINGLE_XTICK_FONTSIZE, va="top", ha="left")
        ax.set_title(AOI_NAMES.get(aoi, aoi), fontsize=XLABEL_FONTSIZE)

    axes[1].set_xticks(doy_ticks)
    axes[1].set_xticklabels(doy_labels, fontsize=SINGLE_XTICK_FONTSIZE)
    axes[1].set_xlabel("Day of year", fontsize=XLABEL_FONTSIZE)

    # Year legend — top panel only
    year_handles = [
        Line2D([0], [0], color=_year_color(i, yr), lw=1.5, label=str(yr))
        for i, yr in enumerate(YEARS)
    ]
    axes[0].legend(
        handles=year_handles, title="Year",
        loc=SINGLE_LEGEND_LOC, fontsize=SINGLE_LEGEND_FONTSIZE,
        title_fontsize=SINGLE_LEGEND_TITLE_FONTSIZE,
        ncol=SINGLE_LEGEND_NCOL, framealpha=SINGLE_LEGEND_FRAMEALPHA,
        edgecolor=SINGLE_LEGEND_EDGECOLOR,
    )

    fig.suptitle(
        f"Running {ROLLING_WINDOW_SIZE}-window mean drainage volume  ·  {season_label}",
        fontsize=SINGLE_SUPTITLE_FONTSIZE, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 12) MAIN
# ============================================================

def main() -> None:
    if not MAKE_MULTI_PANEL and not MAKE_SINGLE_PANEL:
        raise ValueError("Both MAKE_MULTI_PANEL and MAKE_SINGLE_PANEL are False — nothing to plot.")

    if not DRAINAGE_EVENTS_CSV.exists():
        raise FileNotFoundError(f"Input file not found:\n  {DRAINAGE_EVENTS_CSV}")

    apply_rcparams()

    print("Loading drainage events …")
    raw           = load_events(DRAINAGE_EVENTS_CSV)
    valid_windows = build_valid_windows(COVERAGE_CSV)
    filt          = filter_by_coverage(filter_by_doy(raw), valid_windows)
    season_label  = "JJA" if USE_JJA else "full season"
    print(f"  {len(raw)} total events  →  {len(filt)} after {season_label} DOY + "
          f">{COVERAGE_THRESHOLD:.0%} coverage filter")

    suffix  = "_jja" if USE_JJA else "_season"
    win_dfs = {}

    for aoi in AOIS:
        print(f"\n{'=' * 60}\n  AOI: {aoi}  ({AOI_NAMES.get(aoi, '')})\n{'=' * 60}")
        df_aoi = filt[filt["aoi"] == aoi]
        n_full    = int((df_aoi["drain_type"] == "full").sum())
        n_partial = int((df_aoi["drain_type"] == "partial").sum())
        print(f"  {len(df_aoi)} events — {n_full} full, {n_partial} partial")

        if df_aoi.empty:
            print("  No events in range — skipping.")
            continue

        win_df       = aggregate_to_windows(df_aoi)
        win_dfs[aoi] = win_df
        y_max        = global_y_max(win_df)
        print(f"  {len(win_df)} event windows  |  y_max = {y_max:.4f} {VOL_UNIT}")

        if MAKE_MULTI_PANEL:
            out = OUTPUT_DIR / f"{aoi}_drainage_volume_by_season_multi{suffix}.png"
            plot_multi_panel(win_df, aoi, y_max, out)

    if MAKE_ROLLING_MEAN and not OVERLAY_ROLLING_MEAN and len(win_dfs) == len(AOIS):
        out = OUTPUT_DIR / f"combined_drainage_volume_rolling_mean{suffix}.png"
        plot_rolling_mean_combined(win_dfs, out)

    if MAKE_SINGLE_PANEL and len(win_dfs) == len(AOIS):
        shared_y_max = max(global_y_max(w) for w in win_dfs.values())
        out = OUTPUT_DIR / f"combined_drainage_volume_by_season_single{suffix}.png"
        plot_combined_single_panel(win_dfs, shared_y_max, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
