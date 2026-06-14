# -*- coding: utf-8 -*-
"""
plot_mean_max_volume.py

Plots annual JJA (June-July-August) mean and maximum lake-volume time series
from both observed and scaled (observed + projected) block data.

For each AOI two figures are produced:
  - Observed figure : uses direct measurements from the raw bin-level volume CSV
  - Scaled figure   : uses scaled data (obs + proj)

Each figure contains two stacked panels:
  (a) Lake volume
  (b) AOI footprint coverage

The x-axis shows calendar years.  Within the volume panel a solid line marks
the annual JJA mean and a dashed line marks the annual JJA maximum.

Only 5-window blocks that fall within the JJA day-of-year window AND whose
AOI footprint coverage meets FOOTPRINT_COVERAGE_THRESHOLD are included in
the annual statistics.

"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline
from scipy.stats import norm


# ============================================================
# 1) FILE PATHS
# ============================================================

# Bin-level observed lake volume CSV
OBS_VOL_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_lake_volume_by_aspect_elev_bins.csv"
)

# Bin-level projected lake volume CSV
PROJ_VOL_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_projected_lake_volume_by_aspect_elev_bins.csv"
)

# Raw block summary - used only for the '{AOI}_footprint_proportion' filter
TOTAL_SUMMARY_RAW_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_total_summary_raw.csv"
)

# Output directory
OBS_OUTPUT_DIR    = Path(r"Q:\ThesisData\data\figures\plot_mean_max_volume")
SCALED_OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_mean_max_volume")

# CSV export directory
CSV_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
)


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]

AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

# Calendar years to include in the time series
YEARS = list(range(2014, 2026))

# JJA day-of-year bounds (DOY 152 = Jun 1, DOY 243 = Aug 31 in non-leap years)
JJA_DOY_START = 152
JJA_DOY_END   = 243

# Minimum AOI footprint coverage required for a block to contribute to JJA
# statistics.  Applied to the '{AOI}_footprint_proportion' column (0 - 1).
FOOTPRINT_COVERAGE_THRESHOLD = 0.10   # 10 % of AOI area must be observed

# Volume unit conversion
VOL_DIVISOR = 1e6       # m³ -> Mm³
VOL_UNIT    = "Mm³"

# Figure output toggles
PLOT_OBSERVED = True
PLOT_SCALED   = True


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

# Figure dimensions - 180 mm wide (two-column) x ~152 mm tall (two panels)
FIG_WIDTH_IN  = 7.087   # 180 mm
FIG_HEIGHT_IN = 6.0     # ~152 mm

DPI_SCREEN = 150
DPI_SAVE   = 300

HSPACE = 0.38

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

# Colour - lake (ponded water)
COLOUR_LAKE     = "#1A9641"   # green

LS_MEAN      = "-"
LS_MAX       = "--"
LINE_WIDTH   = 1.2
MARKER_STYLE = "o"
MARKER_SIZE  = 3.5
MARKER_EDGE_WIDTH = 0.4
MARKER_EDGE_COLOR = "white"
LINE_ZORDER  = 3

SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0
YEAR_LABEL_ROTATION = 45

GRID_LINESTYLE = ":"
GRID_ALPHA     = 0.45
GRID_LINEWIDTH = 0.5

Y_HEADROOM = 1.10

SMOOTH_LINES    = True
SMOOTH_TENSION  = 1.0
SMOOTH_N_POINTS = 80

PANEL_LABEL_X = -0.11
PANEL_LABEL_Y =  1.03

COLOUR_COVERAGE    = "#666666"
COVERAGE_BAR_ALPHA = 0.70


# ============================================================
# 4) DATA LOADING
# ============================================================

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("block_start", "block_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _aggregate_bins_to_aoi(df: pd.DataFrame, aoi: str) -> pd.Series:
    """Sum all '{aoi}_*' bin columns per row.  Rows where every bin is NaN -> NaN."""
    cols = [c for c in df.columns if c.startswith(f"{aoi}_")]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    subset = df[cols]
    result = subset.sum(axis=1, skipna=True)
    result[subset.isna().all(axis=1)] = np.nan
    return result


def build_observed_df(base_df: pd.DataFrame, aois: list) -> pd.DataFrame:
    """
    Load the observed bin-level volume CSV, aggregate to per-AOI totals, and
    merge footprint proportion columns from the raw block summary.
    """
    print("  Loading observed lake volume bins CSV ...")
    obs_vol = _load_csv(OBS_VOL_BINS_CSV)

    fp_cols = [c for c in base_df.columns if "footprint_proportion" in c]
    df = base_df[["block_start", "block_end"] + fp_cols].copy()

    for aoi in aois:
        agg = pd.DataFrame({
            "block_start":          obs_vol["block_start"],
            f"{aoi}_lake_vol_m3":   _aggregate_bins_to_aoi(obs_vol, aoi),
        })
        df = df.merge(agg, on="block_start", how="left")

    df["year"] = df["block_start"].dt.year
    df["doy"]  = df["block_start"].dt.dayofyear
    return df


def build_scaled_df(base_df: pd.DataFrame, aois: list) -> pd.DataFrame:
    """
    Load bin-level observed and projected volume CSVs, compute scaled volume
    (observed + projected) for each AOI, and return a DataFrame ready for
    compute_jja_stats().

    Scaled volume is set to NaN for any block where the projection is absent.
    """
    print("  Loading bin-level volume CSVs ...")
    obs_vol  = _load_csv(OBS_VOL_BINS_CSV)
    proj_vol = _load_csv(PROJ_VOL_BINS_CSV)

    fp_cols = [c for c in base_df.columns if "footprint_proportion" in c]
    df = base_df[["block_start", "block_end"] + fp_cols].copy()

    for aoi in aois:
        obs_agg  = pd.DataFrame({
            "block_start": obs_vol["block_start"],
            "_obs":        _aggregate_bins_to_aoi(obs_vol, aoi),
        })
        proj_agg = pd.DataFrame({
            "block_start": proj_vol["block_start"],
            "_proj":       _aggregate_bins_to_aoi(proj_vol, aoi),
        })
        tbl = obs_agg.merge(proj_agg, on="block_start", how="outer")
        tbl[f"{aoi}_lake_vol_m3"] = np.where(
            tbl["_proj"].notna(),
            tbl["_obs"].fillna(0.0) + tbl["_proj"],
            np.nan,
        )
        df = df.merge(
            tbl[["block_start", f"{aoi}_lake_vol_m3"]],
            on="block_start", how="left",
        )

    df["year"] = df["block_start"].dt.year
    df["doy"]  = df["block_start"].dt.dayofyear
    return df


# ============================================================
# 5) JJA STATISTICS
# ============================================================

def compute_jja_stats(df: pd.DataFrame, aoi: str) -> pd.DataFrame:
    """
    Compute annual JJA mean and maximum lake volume.

    Only blocks whose block_start falls within JJA_DOY_START..JJA_DOY_END and
    whose AOI footprint coverage >= FOOTPRINT_COVERAGE_THRESHOLD are included.

    Returns a DataFrame indexed by year with columns:
        lake_mean, lake_max, coverage_mean, n
    """
    fp_col   = f"{aoi}_footprint_proportion"
    vol_col  = f"{aoi}_lake_vol_m3"

    jja_raw = df[df["doy"].between(JJA_DOY_START, JJA_DOY_END)].copy()

    if fp_col in jja_raw.columns:
        jja = jja_raw[jja_raw[fp_col] >= FOOTPRINT_COVERAGE_THRESHOLD].copy()
    else:
        print(f"  WARNING: column '{fp_col}' not found - no coverage filter applied.")
        jja = jja_raw.copy()

    print(f"  [{aoi}] vol_col='{vol_col}' present: {vol_col in jja.columns}")

    def _safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if s.notna().any() else np.nan

    def _safe_max(s: pd.Series) -> float:
        return float(s.max()) if s.notna().any() else np.nan

    records = []
    for yr in YEARS:
        sub     = jja[jja["year"] == yr]
        sub_raw = jja_raw[jja_raw["year"] == yr]
        n = int(sub[vol_col].notna().sum()) if vol_col in sub.columns else 0
        cov_mean = (
            _safe_mean(sub_raw[fp_col]) if fp_col in jja_raw.columns else np.nan
        )
        records.append({
            "year":          yr,
            "lake_mean":     _safe_mean(sub[vol_col]) if vol_col in sub.columns else np.nan,
            "lake_max":      _safe_max( sub[vol_col]) if vol_col in sub.columns else np.nan,
            "coverage_mean": cov_mean,
            "n":             n,
        })

    return pd.DataFrame(records).set_index("year")


# ============================================================
# 6) CONSOLE SUMMARY
# ============================================================

def print_jja_summary(stats: pd.DataFrame, aoi: str, label: str) -> None:
    print(
        f"\n  {aoi}  [{label}]  -  JJA lake-volume statistics  "
        f"(footprint threshold >= {FOOTPRINT_COVERAGE_THRESHOLD:.0%})"
    )
    header = (
        f"  {'Year':<6} {'n':>4}  {'Coverage':>10}  "
        f"{'Lake mean':>12} {'Lake max':>12}  ({VOL_UNIT})"
    )
    print(header)
    for yr, row in stats.iterrows():
        cov = (
            f"{row['coverage_mean']:.1%}"
            if not np.isnan(row["coverage_mean"]) else "-"
        )
        if row["n"] == 0:
            print(f"  {yr:<6} {'-':>4}  {cov:>10}")
            continue
        def _f(v):
            return f"{v / VOL_DIVISOR:12.3f}" if not np.isnan(v) else f"{'-':>12}"
        print(
            f"  {yr:<6} {int(row['n']):>4}  {cov:>10}  "
            f"{_f(row['lake_mean'])}  {_f(row['lake_max'])}"
        )


# ============================================================
# 7) STATISTICAL TESTS
# ============================================================

def _mann_kendall(x: np.ndarray) -> dict:
    """Two-sided Mann-Kendall monotonic trend test with tie correction."""
    n = len(x)
    s = sum(
        int(np.sign(x[j] - x[i]))
        for i in range(n - 1)
        for j in range(i + 1, n)
    )
    _, counts = np.unique(x, return_counts=True)
    tie_term = sum(int(c) * (int(c) - 1) * (2 * int(c) + 5) for c in counts if c > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    z = (s - int(np.sign(s))) / np.sqrt(var_s) if var_s > 0 else 0.0
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = ("increasing" if s > 0 else "decreasing") if p < 0.05 else "no trend"
    return {"trend": trend, "tau": tau, "s": s, "z": z, "p": p}


def _pettitt(x: np.ndarray, n_perm: int = 9_999, seed: int = 0) -> dict:
    """
    Pettitt's non-parametric change-point test.
    p-value via Monte Carlo permutation (exact at small n).
    """
    n = len(x)

    def _K(arr: np.ndarray):
        abs_U = np.abs(np.array([
            float(np.sign(arr[:t, None] - arr[None, t:]).sum())
            for t in range(1, n)
        ]))
        return float(abs_U.max()), int(abs_U.argmax())

    K, cp_idx = _K(x)
    rng = np.random.default_rng(seed)
    exceed = sum(1 for _ in range(n_perm) if _K(rng.permutation(x))[0] >= K)
    p = float((exceed + 1) / (n_perm + 1))
    return {"K": K, "p": p, "cp_idx": cp_idx}


def _sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_trend_tests(stats: pd.DataFrame, aoi: str, label: str) -> None:
    """Print Mann-Kendall and Pettitt's test results for JJA lake-volume metrics."""
    metrics = [
        ("lake_mean", "Lake mean"),
        ("lake_max",  "Lake max"),
    ]
    SEP = "  " + "-" * 74

    print(f"\n  {aoi}  [{label}]  -  Mann-Kendall trend test")
    print(f"  {'Series':<18} {'n':>3}  {'tau':>7}  {'S':>6}  {'z':>6}  {'p-value':>9}  {'':3}  Trend")
    print(SEP)
    for col, name in metrics:
        sub = stats[col].dropna()
        if len(sub) < 4:
            print(f"  {name:<18}  -  (insufficient data, n={len(sub)})")
            continue
        mk = _mann_kendall(sub.values)
        print(
            f"  {name:<18} {len(sub):>3}  {mk['tau']:>7.3f}  {mk['s']:>6}  "
            f"{mk['z']:>6.2f}  {mk['p']:>9.4f}  {_sig(mk['p']):<3}  {mk['trend']}"
        )

    print(f"\n  {aoi}  [{label}]  -  Pettitt's change-point test")
    print(f"  {'Series':<18} {'n':>3}  {'K':>8}  {'Change point':>16}  {'p-value':>9}  {'':3}")
    print(SEP)
    for col, name in metrics:
        sub = stats[col].dropna()
        if len(sub) < 4:
            print(f"  {name:<18}  -  (insufficient data, n={len(sub)})")
            continue
        years = list(sub.index)
        pt = _pettitt(sub.values)
        cp_year = years[pt["cp_idx"]]
        print(
            f"  {name:<18} {len(sub):>3}  {pt['K']:>8.1f}  "
            f"  after {cp_year}      {pt['p']:>9.4f}  {_sig(pt['p']):<3}"
        )


# ============================================================
# 8) PLOTTING - HELPERS
# ============================================================

def _apply_nature_rcparams() -> None:
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


def _configure_y_axis(ax: plt.Axes, y_max: float) -> None:
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
        zorder=0,
    )


def _configure_x_axis(
    ax: plt.Axes, show_labels: bool, n_map: dict | None = None
) -> None:
    ax.set_xticks(YEARS)
    if show_labels:
        if n_map is not None:
            labels = [f"{y} ({n_map.get(y, 0)})" for y in YEARS]
        else:
            labels = [str(y) for y in YEARS]
        ax.set_xticklabels(labels, rotation=YEAR_LABEL_ROTATION, ha="right")
    else:
        ax.set_xticklabels([])
    ax.set_xlim(YEARS[0] - 0.7, YEARS[-1] + 0.7)


def _hermite_smooth(xv: np.ndarray, yv: np.ndarray) -> tuple:
    n = len(xv)
    dydx = np.zeros(n)
    for i in range(1, n - 1):
        dydx[i] = (yv[i + 1] - yv[i - 1]) / (xv[i + 1] - xv[i - 1])
    dydx[0]  = (yv[1]  - yv[0])  / (xv[1]  - xv[0])
    dydx[-1] = (yv[-1] - yv[-2]) / (xv[-1] - xv[-2])
    dydx *= (1.0 - SMOOTH_TENSION)
    cs = CubicHermiteSpline(xv, yv, dydx)
    x_fine = np.linspace(xv[0], xv[-1], SMOOTH_N_POINTS)
    return x_fine, np.maximum(cs(x_fine), 0.0)


def _smooth_segments(x_all: np.ndarray, y_all: np.ndarray) -> list:
    segments = []
    valid = ~np.isnan(y_all)
    in_seg, seg_start = False, 0
    for i in range(len(valid) + 1):
        currently_valid = i < len(valid) and valid[i]
        if currently_valid and not in_seg:
            seg_start = i
            in_seg = True
        elif not currently_valid and in_seg:
            xv, yv = x_all[seg_start:i], y_all[seg_start:i]
            if len(xv) == 1:
                segments.append((xv, yv))
            else:
                segments.append(_hermite_smooth(xv, yv))
            in_seg = False
    return segments


def _plot_mean_max_line(
    ax: plt.Axes,
    stats: pd.DataFrame,
    mean_col: str,
    max_col: str,
    colour: str,
) -> tuple:
    """
    Draw the JJA mean (solid) and JJA maximum (dashed) volume lines.
    Returns proxy Line2D handles for the legend.
    """
    years_arr = np.array(YEARS, dtype=float)
    mean_vals = np.full(len(YEARS), np.nan)
    max_vals  = np.full(len(YEARS), np.nan)

    for i, yr in enumerate(YEARS):
        if yr in stats.index and stats.loc[yr, "n"] > 0:
            mean_vals[i] = stats.loc[yr, mean_col] / VOL_DIVISOR
            max_vals[i]  = stats.loc[yr, max_col]  / VOL_DIVISOR

    kw_line   = dict(color=colour, linewidth=LINE_WIDTH, zorder=LINE_ZORDER)
    kw_marker = dict(
        color=colour, linestyle="none",
        marker=MARKER_STYLE, markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor=MARKER_EDGE_COLOR,
        zorder=LINE_ZORDER + 1, clip_on=False,
    )

    for vals, ls in ((mean_vals, LS_MEAN), (max_vals, LS_MAX)):
        if SMOOTH_LINES:
            for seg_x, seg_y in _smooth_segments(years_arr, vals):
                ax.plot(seg_x, seg_y, linestyle=ls, **kw_line)
        else:
            ax.plot(years_arr, vals, linestyle=ls, **kw_line)
        valid = ~np.isnan(vals)
        ax.plot(years_arr[valid], vals[valid], **kw_marker)

    kw_proxy = dict(color=colour, linewidth=LINE_WIDTH,
                    marker=MARKER_STYLE, markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor=MARKER_EDGE_COLOR)
    h_mean = Line2D([0], [0], linestyle=LS_MEAN, label="JJA mean",    **kw_proxy)
    h_max  = Line2D([0], [0], linestyle=LS_MAX,  label="JJA maximum", **kw_proxy)
    return h_mean, h_max


def _compute_y_max(stats: pd.DataFrame, mean_col: str, max_col: str) -> float:
    """Return the panel y-axis ceiling with headroom."""
    vals = pd.concat([
        stats[mean_col].dropna(),
        stats[max_col].dropna(),
    ]) / VOL_DIVISOR
    return float(vals.max() * Y_HEADROOM) if not vals.empty and vals.max() > 0 else 1.0


def _add_legend(ax: plt.Axes, h_mean: Line2D, h_max: Line2D) -> None:
    ax.legend(
        handles=[h_mean, h_max],
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=2.2,
    )


def _plot_coverage_bars(ax: plt.Axes, stats: pd.DataFrame) -> None:
    """Draw annual mean JJA AOI coverage as bars."""
    years_arr = np.array(YEARS, dtype=float)
    cov_vals  = np.array([
        stats.loc[yr, "coverage_mean"] if yr in stats.index else np.nan
        for yr in YEARS
    ])
    valid = ~np.isnan(cov_vals)
    ax.bar(
        years_arr[valid], cov_vals[valid],
        width=0.6, color=COLOUR_COVERAGE, alpha=COVERAGE_BAR_ALPHA, zorder=2,
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
        zorder=0,
    )


# ============================================================
# 8) PLOTTING - FIGURE BUILDER
# ============================================================

def make_figure(
    stats: pd.DataFrame,
    aoi: str,
    label: str,
    out_path: Path,
) -> None:
    """
    Build and save a two-panel JJA lake-volume time-series figure for one AOI.

    Parameters
    ----------
    stats    : DataFrame indexed by year (output of compute_jja_stats)
    aoi      : AOI identifier, e.g. "OST" or "PTM"
    label    : "Observed" or "Scaled" (used in title and filename)
    out_path : destination PNG path
    """
    y_lake = _compute_y_max(stats, "lake_mean", "lake_max")

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 0.6]},
    )
    fig.subplots_adjust(hspace=HSPACE)

    # -- (a) Lake volume -------------------------------------------------------
    h_mean, h_max = _plot_mean_max_line(
        ax_a, stats, "lake_mean", "lake_max", COLOUR_LAKE
    )
    ax_a.set_ylabel(f"Volume ({VOL_UNIT})")
    _configure_y_axis(ax_a, y_lake)
    _configure_x_axis(ax_a, show_labels=True)
    _add_panel_label(ax_a, "(a)  Lake")
    _add_legend(ax_a, h_mean, h_max)

    # -- (b) AOI coverage -------------------------------------------------------
    n_map = {yr: int(stats.loc[yr, "n"]) for yr in YEARS if yr in stats.index}
    _plot_coverage_bars(ax_b, stats)
    ax_b.set_ylabel("Coverage")
    ax_b.set_xlabel("Year (n = blocks per year)")
    _configure_x_axis(ax_b, show_labels=True, n_map=n_map)
    _add_panel_label(ax_b, "(b)  AOI coverage")

    # -- Figure title ----------------------------------------------------------
    aoi_name  = AOI_NAMES.get(aoi, aoi)
    threshold = FOOTPRINT_COVERAGE_THRESHOLD
    fig.suptitle(
        f"Lake Volume, {label} JJA, {aoi_name}, "
        f"(footprint coverage >= {threshold:.0%})",
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=0.995,
    )

    # -- Save ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved -> {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 9) CSV EXPORT
# ============================================================

_METRICS = [
    ("lake_mean", "lake_mean"),
    ("lake_max",  "lake_max"),
]


def _summary_rows(stats: pd.DataFrame, aoi: str, label: str) -> list:
    rows = []
    for yr, row in stats.iterrows():
        def _mm3(col):
            v = row[col]
            return round(v / VOL_DIVISOR, 6) if not np.isnan(v) else np.nan
        rows.append({
            "aoi":              aoi,
            "label":            label,
            "year":             yr,
            "n":                int(row["n"]),
            "coverage_mean":    round(row["coverage_mean"], 6) if not np.isnan(row["coverage_mean"]) else np.nan,
            "lake_mean_Mm3":    _mm3("lake_mean"),
            "lake_max_Mm3":     _mm3("lake_max"),
        })
    return rows


def _mk_rows(stats: pd.DataFrame, aoi: str, label: str) -> list:
    rows = []
    for col, name in _METRICS:
        sub = stats[col].dropna()
        if len(sub) < 4:
            continue
        mk = _mann_kendall(sub.values)
        rows.append({
            "aoi":          aoi,
            "label":        label,
            "series":       name,
            "n":            len(sub),
            "tau":          round(mk["tau"], 6),
            "S":            int(mk["s"]),
            "z":            round(mk["z"], 6),
            "p_value":      round(mk["p"], 8),
            "significance": _sig(mk["p"]),
            "trend":        mk["trend"],
        })
    return rows


def _pettitt_rows(stats: pd.DataFrame, aoi: str, label: str) -> list:
    rows = []
    for col, name in _METRICS:
        sub = stats[col].dropna()
        if len(sub) < 4:
            continue
        years = list(sub.index)
        pt = _pettitt(sub.values)
        rows.append({
            "aoi":                     aoi,
            "label":                   label,
            "series":                  name,
            "n":                       len(sub),
            "K":                       round(pt["K"], 4),
            "change_point_after_year": years[pt["cp_idx"]],
            "p_value":                 round(pt["p"], 8),
            "significance":            _sig(pt["p"]),
        })
    return rows


def _save_csvs(summary: list, mk: list, pettitt: list) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, rows in (
        ("volume_jja_summary.csv",  summary),
        ("volume_mann_kendall.csv", mk),
        ("volume_pettitt.csv",      pettitt),
    ):
        path = CSV_OUTPUT_DIR / fname
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved CSV -> {path}")


# ============================================================
# 10) MAIN
# ============================================================

def main() -> None:
    if not PLOT_OBSERVED and not PLOT_SCALED:
        print("Both PLOT_OBSERVED and PLOT_SCALED are False - nothing to do.")
        return

    _apply_nature_rcparams()

    # -- Validate input files --------------------------------------------------
    required_obs = [TOTAL_SUMMARY_RAW_CSV, OBS_VOL_BINS_CSV]
    required_scaled = [TOTAL_SUMMARY_RAW_CSV, OBS_VOL_BINS_CSV, PROJ_VOL_BINS_CSV]
    to_check = required_obs + (required_scaled if PLOT_SCALED else [])
    for p in dict.fromkeys(to_check):
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found:\n  {p}")

    # -- Load base data --------------------------------------------------------
    print("Loading raw block summary ...")
    base_df = _load_csv(TOTAL_SUMMARY_RAW_CSV)
    print(f"  {len(base_df)} total blocks")

    obs_df    = build_observed_df(base_df, AOIS) if PLOT_OBSERVED else None
    scaled_df = build_scaled_df(base_df, AOIS)   if PLOT_SCALED   else None

    # -- Per-AOI processing and plotting ---------------------------------------
    all_summary = []
    all_mk      = []
    all_pettitt = []

    for aoi in AOIS:
        print(f"\n{'=' * 65}")
        print(f"  AOI: {aoi}  ({AOI_NAMES.get(aoi, '')})")
        print(f"{'=' * 65}")

        if PLOT_OBSERVED:
            print("\n[Observed]")
            stats = compute_jja_stats(obs_df, aoi)
            print_jja_summary(stats, aoi, "Observed")
            print_trend_tests(stats, aoi, "Observed")
            all_summary.extend(_summary_rows(stats, aoi, "Observed"))
            all_mk.extend(_mk_rows(stats, aoi, "Observed"))
            all_pettitt.extend(_pettitt_rows(stats, aoi, "Observed"))
            out = OBS_OUTPUT_DIR / f"jja_mean_max_volume_observed_{aoi}.png"
            make_figure(stats, aoi, "Observed", out)

        if PLOT_SCALED:
            print("\n[Scaled]")
            stats = compute_jja_stats(scaled_df, aoi)
            print_jja_summary(stats, aoi, "Scaled")
            print_trend_tests(stats, aoi, "Scaled")
            all_summary.extend(_summary_rows(stats, aoi, "Scaled"))
            all_mk.extend(_mk_rows(stats, aoi, "Scaled"))
            all_pettitt.extend(_pettitt_rows(stats, aoi, "Scaled"))
            out = SCALED_OUTPUT_DIR / f"jja_mean_max_volume_scaled_{aoi}.png"
            make_figure(stats, aoi, "Scaled", out)

    print("\nSaving CSV outputs ...")
    _save_csvs(all_summary, all_mk, all_pettitt)
    print("\nDone.")


if __name__ == "__main__":
    main()
