# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:54:29 2026

@author: Lukeu

plot_era5_timeseries.py

Plots annual JJA (June-July-August) mean and maximum ERA5-Land climate
time series from the AOI CSV exported by the ERA5 GEE workflow.

The preferred input CSV contains one row per site/date.  The loader also
supports the elevation-bin CSV shape by averaging bins to one daily AOI value.

Outputs:
  - era5_jja_mean_max_{filename}.png  (one per variable, one stacked panel per AOI)
  - era5_jja_mean_max_summary.csv     (read by plot_era5_r2_hysteresis.py)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ============================================================
# 1) FILE PATHS
# ============================================================

INPUT_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\era5"
    r"\ERA5Land_T2m_SWdown_OST_PTM_2013_2025_MaySep.csv"
)

FIG_OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_era5_mean_max")

CSV_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
)

# In Spyder this displays each saved figure in the Plots pane.  Set False for
# fully silent batch runs that only save PNG files.
SHOW_FIGURES = True


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]

AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

VARIABLES = {
    "temp_2m_C": {
        "label": "2 m air temperature",
        "unit": "deg C",
        "colour": "#B2182B",
        "filename": "temperature_2m",
    },
    "srad_Wm2": {
        "label": "Downward shortwave radiation",
        "unit": "W m-2",
        "colour": "#2166AC",
        "filename": "shortwave_down",
    },
}

# JJA is selected using calendar months to handle leap years correctly.
JJA_MONTHS = [6, 7, 8]


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

FIG_WIDTH_IN = 7.087
FIG_HEIGHT_IN = 5.2
DPI_SCREEN = 150
DPI_SAVE = 300
HSPACE = 0.32

FONT_FAMILY = "sans-serif"
BASE_FONT_SIZE = 7
TICK_FONT_SIZE = 6
LABEL_FONT_SIZE = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

LS_MEAN = "-"
LS_MAX = "--"
LINE_WIDTH = 1.2
MARKER_STYLE = "o"
MARKER_SIZE = 3.5
MARKER_EDGE_WIDTH = 0.4
MARKER_EDGE_COLOR = "white"
LINE_ZORDER = 3

SPINE_WIDTH = 0.6
TICK_WIDTH = 0.6
TICK_LENGTH = 3.0
YEAR_LABEL_ROTATION = 45

GRID_LINESTYLE = ":"
GRID_ALPHA = 0.45
GRID_LINEWIDTH = 0.5
Y_HEADROOM = 1.10

PANEL_LABEL_X = -0.11
PANEL_LABEL_Y = 1.03


# ============================================================
# 4) DATA LOADING AND STATISTICS
# ============================================================

def _apply_rcparams() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "axes.titlesize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": SPINE_WIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": DPI_SCREEN,
        "savefig.dpi": DPI_SAVE,
    })


def load_daily_aoi_values(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found:\n  {path}")

    required = {"date", "year", *VARIABLES.keys()}
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )
    if "site" not in df.columns and "bin_id" not in df.columns:
        raise ValueError("Input CSV must include either a 'site' or 'bin_id' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "site" in df.columns:
        df["aoi"] = df["site"].astype(str)
    else:
        df["aoi"] = df["bin_id"].astype(str).str.extract(r"^(OST|PTM)", expand=False)
    df = df[df["aoi"].isin(AOIS)].copy()

    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    daily = df.groupby(["aoi", "date"], as_index=False).agg({
        **{col: "mean" for col in VARIABLES},
        "year": "first",
    })
    daily = daily.sort_values(["aoi", "date"]).reset_index(drop=True)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    return daily


def compute_jja_stats(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    years = sorted(int(y) for y in daily["year"].dropna().unique())
    jja = daily[daily["month"].isin(JJA_MONTHS)].copy()

    for aoi in AOIS:
        aoi_jja = jja[jja["aoi"] == aoi]
        for year in years:
            sub = aoi_jja[aoi_jja["year"] == year]
            row = {
                "aoi": aoi,
                "year": year,
                "n_days": int(sub["date"].nunique()),
            }
            for col in VARIABLES:
                vals = sub[col].dropna()
                row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
                row[f"{col}_max"] = float(vals.max()) if not vals.empty else np.nan
            records.append(row)

    return pd.DataFrame(records)


# ============================================================
# 5) PLOTTING HELPERS
# ============================================================

def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _configure_x_axis(ax: plt.Axes, years: list[int], show_labels: bool) -> None:
    ax.set_xticks(years)
    if show_labels:
        ax.set_xticklabels([str(y) for y in years], rotation=YEAR_LABEL_ROTATION, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlim(years[0] - 0.7, years[-1] + 0.7)


def _configure_y_axis(ax: plt.Axes, stats: pd.DataFrame, mean_col: str, max_col: str) -> None:
    vals = pd.concat([stats[mean_col].dropna(), stats[max_col].dropna()])
    if vals.empty:
        y_min, y_max = 0.0, 1.0
    else:
        data_min = float(vals.min())
        data_max = float(vals.max())
        if data_min >= 0:
            y_min = 0.0
            y_max = data_max * Y_HEADROOM if data_max > 0 else 1.0
        else:
            span = data_max - data_min
            pad = span * 0.10 if span > 0 else 1.0
            y_min = data_min - pad
            y_max = data_max + pad
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
        zorder=0,
    )


def _plot_mean_max_line(
    ax: plt.Axes,
    stats: pd.DataFrame,
    years: list[int],
    mean_col: str,
    max_col: str,
    colour: str,
) -> tuple[Line2D, Line2D]:
    x = np.array(years, dtype=float)
    mean_vals = np.array([
        stats.loc[stats["year"] == y, mean_col].iloc[0]
        if (stats["year"] == y).any() else np.nan
        for y in years
    ])
    max_vals = np.array([
        stats.loc[stats["year"] == y, max_col].iloc[0]
        if (stats["year"] == y).any() else np.nan
        for y in years
    ])

    kw_line = dict(color=colour, linewidth=LINE_WIDTH, zorder=LINE_ZORDER)
    kw_marker = dict(
        color=colour,
        linestyle="none",
        marker=MARKER_STYLE,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        markeredgecolor=MARKER_EDGE_COLOR,
        zorder=LINE_ZORDER + 1,
        clip_on=False,
    )

    for vals, ls in ((mean_vals, LS_MEAN), (max_vals, LS_MAX)):
        ax.plot(x, vals, linestyle=ls, **kw_line)
        valid = ~np.isnan(vals)
        ax.plot(x[valid], vals[valid], **kw_marker)

    kw_proxy = dict(
        color=colour,
        linewidth=LINE_WIDTH,
        marker=MARKER_STYLE,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH,
        markeredgecolor=MARKER_EDGE_COLOR,
    )
    h_mean = Line2D([0], [0], linestyle=LS_MEAN, label="JJA mean", **kw_proxy)
    h_max = Line2D([0], [0], linestyle=LS_MAX, label="JJA maximum", **kw_proxy)
    return h_mean, h_max


def _add_legend(ax: plt.Axes, h_mean: Line2D, h_max: Line2D) -> None:
    ax.legend(
        handles=[h_mean, h_max],
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=2.2,
    )


def make_variable_figure(stats: pd.DataFrame, variable: str, out_path: Path) -> None:
    info = VARIABLES[variable]
    years = sorted(int(y) for y in stats["year"].dropna().unique())
    mean_col = f"{variable}_mean"
    max_col = f"{variable}_max"

    fig, axes = plt.subplots(
        len(AOIS), 1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=False,
    )
    if len(AOIS) == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=HSPACE)

    for idx, (ax, aoi) in enumerate(zip(axes, AOIS)):
        aoi_stats = stats[stats["aoi"] == aoi].copy()
        h_mean, h_max = _plot_mean_max_line(
            ax, aoi_stats, years, mean_col, max_col, info["colour"]
        )
        _configure_y_axis(ax, aoi_stats, mean_col, max_col)
        _configure_x_axis(ax, years, show_labels=True)
        _add_panel_label(ax, f"({chr(97 + idx)})  {AOI_NAMES.get(aoi, aoi)}")
        _add_legend(ax, h_mean, h_max)
        ax.set_ylabel(f"{info['label']}\n({info['unit']})")

    axes[-1].set_xlabel("Year")
    fig.suptitle(
        f"ERA5-Land {info['label']}, JJA annual mean and maximum",
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    if SHOW_FIGURES and mpl.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved -> {out_path}")


# ============================================================
# 6) OUTPUTS AND MAIN
# ============================================================

def save_summary(stats: pd.DataFrame) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_OUTPUT_DIR / "era5_jja_mean_max_summary.csv"
    stats.to_csv(out, index=False)
    print(f"  Saved CSV -> {out}")


def print_summary(stats: pd.DataFrame) -> None:
    for variable, info in VARIABLES.items():
        mean_col = f"{variable}_mean"
        max_col = f"{variable}_max"
        print(f"\n{info['label']} ({info['unit']})")
        print(f"  {'AOI':<5} {'Year':<6} {'n':>4} {'JJA mean':>12} {'JJA max':>12}")
        for _, row in stats.sort_values(["aoi", "year"]).iterrows():
            print(
                f"  {row['aoi']:<5} {int(row['year']):<6} {int(row['n_days']):>4} "
                f"{row[mean_col]:>12.3f} {row[max_col]:>12.3f}"
            )


def main() -> None:
    _apply_rcparams()

    print("Loading ERA5-Land AOI CSV ...")
    daily = load_daily_aoi_values(INPUT_CSV)
    print(f"  Daily AOI rows: {len(daily)}")

    print("Computing annual JJA mean/max statistics ...")
    stats = compute_jja_stats(daily)
    print_summary(stats)

    print("\nSaving ERA5 time-series outputs ...")
    save_summary(stats)
    for variable, info in VARIABLES.items():
        out = FIG_OUTPUT_DIR / f"era5_jja_mean_max_{info['filename']}.png"
        make_variable_figure(stats, variable, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
