# -*- coding: utf-8 -*-
"""
plot_mean_max_area.py

Plots annual JJA (June-July-August) mean and maximum melt area time series
from both observed and scaled (observed + projected) block data.

For each AOI two figures are produced:
  - Observed figure : uses direct measurements from the raw block summary CSV
  - Scaled figure   : uses scaled data (obs + proj)

Each figure contains three stacked panels:
  (a) Slush area
  (b) Lake (ponded water) area
  (c) Combined slush + lake area

The x-axis shows calendar years.  Within each panel a solid line marks the
annual JJA mean and a dashed line marks the annual JJA maximum.

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

# Raw block summary - footprint coverage + observed slush / lake columns
TOTAL_SUMMARY_RAW_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_total_summary_raw.csv"
)

# Bin-level observed area CSVs (used for scaled figures)
OBS_SLUSH_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_slush_area_by_aspect_elev_bins.csv"
)
OBS_LAKE_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_lake_area_by_aspect_elev_bins.csv"
)

# Bin-level projected area CSVs (used for scaled figures)
PROJ_SLUSH_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_projected_slush_area_by_aspect_elev_bins.csv"
)
PROJ_LAKE_BINS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_projected_lake_area_by_aspect_elev_bins.csv"
)

# Output directories (one per figure type)
OBS_OUTPUT_DIR    = Path(r"Q:\ThesisData\data\figures\plot_mean_max_area")
SCALED_OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_mean_max_area")

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

# JJA day-of-year bounds (non-leap year)
JJA_DOY_START = 152   # 1 June
JJA_DOY_END   = 243   # 31 August

# Minimum AOI footprint coverage required for a block to contribute to JJA
# statistics.  Applied to the '{AOI}_footprint_proportion' column (0 - 1).
# Blocks below this threshold are silently excluded.
FOOTPRINT_COVERAGE_THRESHOLD = 0.10   # 10 % of AOI area must be observed


# Area unit conversion
AREA_DIVISOR = 1e6       # m2 -> km2
AREA_UNIT    = "km2"

# Figure output toggles
PLOT_OBSERVED = True
PLOT_SCALED   = True


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

# Figure dimensions - 180 mm wide (two-column) x 210 mm tall (three panels)
FIG_WIDTH_IN  = 7.087   # 180 mm
FIG_HEIGHT_IN = 8.268   # 210 mm

DPI_SCREEN = 150   # interactive preview
DPI_SAVE   = 300

# Panel vertical spacing fraction
HSPACE = 0.38

# Font
FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8    # bold panel labels (a), (b), (c)

# Colours - one per melt type; colorblind-safe (ColorBrewer / Wong palette)
COLOUR_SLUSH    = "#2166AC"   # blue   - slush area panels
COLOUR_LAKE     = "#1A9641"   # green  - lake area panels
COLOUR_COMBINED = "#762A83"   # purple - combined area panels

# Line / marker styling
LS_MEAN      = "-"    # solid line  -> JJA mean
LS_MAX       = "--"   # dashed line -> JJA maximum
LINE_WIDTH   = 1.2
MARKER_STYLE = "o"
MARKER_SIZE  = 3.5
MARKER_EDGE_WIDTH = 0.4
MARKER_EDGE_COLOR = "white"
LINE_ZORDER  = 3

# Axes
SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0
YEAR_LABEL_ROTATION = 45

# Grid (horizontal reference lines only)
GRID_LINESTYLE = ":"
GRID_ALPHA     = 0.45
GRID_LINEWIDTH = 0.5

# Y-axis headroom above the data maximum
Y_HEADROOM = 1.10

# Line smoothing - tension-controlled Hermite spline between annual data points
# softens the sharp vertices produced by straight-line segments.
#   SMOOTH_TENSION = 0.0  ->  smoothest curve (Catmull-Rom; pronounced rounding)
#   SMOOTH_TENSION = 0.5  ->  gentle rounding (recommended default)
#   SMOOTH_TENSION = 1.0  ->  straight line segments (no smoothing)
SMOOTH_LINES    = True   # set False to revert to straight segments
SMOOTH_TENSION  = 1.0    # 0.0 (smooth) -> 1.0 (straight)
SMOOTH_N_POINTS = 80     # interpolated points per contiguous segment

# Panel label position relative to axes (axes-fraction coordinates)
PANEL_LABEL_X = -0.11
PANEL_LABEL_Y =  1.03

# 4th panel: AOI coverage
COLOUR_COVERAGE       = "#666666"    # dark gray - coverage bars
FIG_HEIGHT_IN_4P      = 10.0         # figure height for four-panel layout (~254 mm)
COVERAGE_BAR_ALPHA    = 0.70


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


def build_observed_df(base_df: pd.DataFrame) -> pd.DataFrame:
    """Attach year and doy columns; observed area columns already present."""
    df = base_df.copy()
    df["year"] = df["block_start"].dt.year
    df["doy"]  = df["block_start"].dt.dayofyear
    return df


def build_scaled_df(base_df: pd.DataFrame, aois: list) -> pd.DataFrame:
    """
    Load bin-level CSVs, compute scaled area (observed + projected) for each
    AOI, and return a DataFrame ready for compute_jja_stats().

    Scaled area is set to NaN for any block where the projection is absent.
    The '{AOI}_footprint_proportion' columns from base_df are preserved so
    that the same coverage filter can be applied to scaled figures.
    """
    print("  Loading bin-level area CSVs ...")
    obs_slush  = _load_csv(OBS_SLUSH_BINS_CSV)
    obs_lake   = _load_csv(OBS_LAKE_BINS_CSV)
    proj_slush = _load_csv(PROJ_SLUSH_BINS_CSV)
    proj_lake  = _load_csv(PROJ_LAKE_BINS_CSV)

    fp_cols = [c for c in base_df.columns if "footprint_proportion" in c]
    df = base_df[["block_start", "block_end"] + fp_cols].copy()

    for aoi in aois:
        slush_tbl = pd.DataFrame({
            "block_start": obs_slush["block_start"],
            "_obs_s":      _aggregate_bins_to_aoi(obs_slush,  aoi),
            "_proj_s":     _aggregate_bins_to_aoi(proj_slush, aoi),
        })
        lake_tbl = pd.DataFrame({
            "block_start": obs_lake["block_start"],
            "_obs_l":      _aggregate_bins_to_aoi(obs_lake,  aoi),
            "_proj_l":     _aggregate_bins_to_aoi(proj_lake, aoi),
        })

        slush_tbl[f"{aoi}_slush_area_m2"] = np.where(
            slush_tbl["_proj_s"].notna(),
            slush_tbl["_obs_s"].fillna(0.0) + slush_tbl["_proj_s"],
            np.nan,
        )
        lake_tbl[f"{aoi}_lake_area_m2"] = np.where(
            lake_tbl["_proj_l"].notna(),
            lake_tbl["_obs_l"].fillna(0.0) + lake_tbl["_proj_l"],
            np.nan,
        )

        df = df.merge(
            slush_tbl[["block_start", f"{aoi}_slush_area_m2"]],
            on="block_start", how="left",
        )
        df = df.merge(
            lake_tbl[["block_start", f"{aoi}_lake_area_m2"]],
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
    Compute annual JJA mean and maximum for slush, lake, and combined melt
    area.  Only blocks within JJA_DOY_START - JJA_DOY_END and with AOI
    footprint coverage ≥ FOOTPRINT_COVERAGE_THRESHOLD are included.

    Combined area is computed per block (slush + lake) before aggregating so
    that the resulting mean/max reflects simultaneous co-occurrence.

    Returns a DataFrame indexed by year with columns:
        slush_mean, slush_max, lake_mean, lake_max,
        combined_mean, combined_max, n
    """
    fp_col    = f"{aoi}_footprint_proportion"
    slush_col = f"{aoi}_slush_area_m2"
    lake_col  = f"{aoi}_lake_area_m2"

    # Restrict to JJA window (before coverage filter - used for coverage_mean)
    jja_raw = df[df["doy"].between(JJA_DOY_START, JJA_DOY_END)].copy()

    # Apply footprint coverage threshold
    if fp_col in jja_raw.columns:
        jja = jja_raw[jja_raw[fp_col] >= FOOTPRINT_COVERAGE_THRESHOLD].copy()
    else:
        print(f"  WARNING: column '{fp_col}' not found - no coverage filter applied.")
        jja = jja_raw.copy()

    print(f"  [{aoi}] columns: {[c for c in jja.columns if aoi in c]}")
    print(f"  [{aoi}] lake_col='{lake_col}' present: {lake_col in jja.columns}")
    print(f"  [{aoi}] slush==lake: {jja[slush_col].equals(jja[lake_col])}")

    # Per-block combined area (NaN where both source columns are NaN)
    jja["_combined_m2"] = jja[slush_col].fillna(0.0) + jja[lake_col].fillna(0.0)
    both_nan = jja[slush_col].isna() & jja[lake_col].isna()
    jja.loc[both_nan, "_combined_m2"] = np.nan

    def _safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if s.notna().any() else np.nan

    def _safe_max(s: pd.Series) -> float:
        return float(s.max()) if s.notna().any() else np.nan

    records = []
    for yr in YEARS:
        sub     = jja[jja["year"] == yr]
        sub_raw = jja_raw[jja_raw["year"] == yr]
        n       = int(sub.dropna(subset=[slush_col, lake_col], how="all").shape[0])
        cov_mean = (
            _safe_mean(sub_raw[fp_col]) if fp_col in jja_raw.columns else np.nan
        )
        records.append({
            "year":          yr,
            "slush_mean":    _safe_mean(sub[slush_col]),
            "slush_max":     _safe_max(sub[slush_col]),
            "lake_mean":     _safe_mean(sub[lake_col]),
            "lake_max":      _safe_max(sub[lake_col]),
            "combined_mean": _safe_mean(sub["_combined_m2"]),
            "combined_max":  _safe_max(sub["_combined_m2"]),
            "coverage_mean": cov_mean,
            "n":             n,
        })

    return pd.DataFrame(records).set_index("year")


# ============================================================
# 6) CONSOLE SUMMARY
# ============================================================

def print_jja_summary(stats: pd.DataFrame, aoi: str, label: str) -> None:
    print(
        f"\n  {aoi}  [{label}]  -  JJA statistics  "
        f"(footprint threshold ≥ {FOOTPRINT_COVERAGE_THRESHOLD:.0%})"
    )
    cols = ["slush_mean", "slush_max", "lake_mean", "lake_max",
            "combined_mean", "combined_max"]
    header = (
        f"  {'Year':<6} {'n':>4}  {'Coverage':>10}  "
        f"{'Slush mean':>12} {'Slush max':>10}  "
        f"{'Lake mean':>12} {'Lake max':>10}  "
        f"{'Comb. mean':>12} {'Comb. max':>10}  ({AREA_UNIT})"
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
            return f"{v / AREA_DIVISOR:10.3f}" if not np.isnan(v) else f"{'-':>10}"
        print(
            f"  {yr:<6} {int(row['n']):>4}  {cov:>10}  "
            + "  ".join(_f(row[c]) for c in cols)
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
    cp_idx is the 0-based position in x after which the change occurs.
    p-value is exact (Monte Carlo permutation): at small n (e.g. n=12) the
    closed-form approximation 2*exp(-6K²/(n³+n²)) is anti-conservative and
    can underestimate p by a factor of 2-3, so a permutation p-value is used.
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
    """Print Mann-Kendall and Pettitt's test results for all JJA metrics."""
    metrics = [
        ("slush_mean",    "Slush mean"),
        ("slush_max",     "Slush max"),
        ("lake_mean",     "Lake mean"),
        ("lake_max",      "Lake max"),
        ("combined_mean", "Combined mean"),
        ("combined_max",  "Combined max"),
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
    """Set global matplotlib style parameters."""
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
    """
    Fit a tension-controlled cubic Hermite spline through (xv, yv).

    Tangents are computed as Catmull-Rom finite differences and then scaled
    by (1 - SMOOTH_TENSION).  At SMOOTH_TENSION = 0 the curve is a standard
    Catmull-Rom spline (maximum rounding); at SMOOTH_TENSION = 1 all tangents
    are zero and the result degenerates to straight line segments.  Intermediate
    values produce proportionally less pronounced rounding.
    """
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
    """
    Split (x_all, y_all) into contiguous non-NaN runs and return a list of
    (x_smooth, y_smooth) arrays.  Runs of ≥2 points are passed through
    _hermite_smooth(); single isolated points are returned unchanged.
    """
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
    Draw the J-J-A mean (solid) and J-J-A maximum (dashed) lines for one panel.
    When SMOOTH_LINES is True the lines are rendered as cubic splines to soften
    the angular vertices between annual data points.  Actual data markers are
    overlaid at the original year positions.
    Returns proxy (handle_mean, handle_max) Line2D objects for the legend.
    """
    years_arr = np.array(YEARS, dtype=float)
    mean_vals = np.full(len(YEARS), np.nan)
    max_vals  = np.full(len(YEARS), np.nan)

    for i, yr in enumerate(YEARS):
        if yr in stats.index and stats.loc[yr, "n"] > 0:
            mean_vals[i] = stats.loc[yr, mean_col] / AREA_DIVISOR
            max_vals[i]  = stats.loc[yr, max_col]  / AREA_DIVISOR

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

    # Proxy artists for the legend (correct combined line + marker appearance)
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
    ]) / AREA_DIVISOR
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
    """Draw annual mean JJA AOI coverage as bars, annotated with n JJA windows."""
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
    Build and save a three-panel JJA time-series figure for one AOI.

    Parameters
    ----------
    stats    : DataFrame indexed by year (output of compute_jja_stats)
    aoi      : AOI identifier, e.g. "OST" or "PTM"
    label    : "Observed" or "Scaled" (used in title and filename)
    out_path : destination PNG path
    """
    y_slush    = _compute_y_max(stats, "slush_mean",    "slush_max")
    y_lake     = _compute_y_max(stats, "lake_mean",     "lake_max")
    y_combined = _compute_y_max(stats, "combined_mean", "combined_max")

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        4, 1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN_4P),
        sharex=False,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.6]},
    )
    fig.subplots_adjust(hspace=HSPACE)

    # -- (a) Slush ----------------------------------------------------------
    h_mean, h_max = _plot_mean_max_line(
        ax_a, stats, "slush_mean", "slush_max", COLOUR_SLUSH
    )
    ax_a.set_ylabel(f"Area ({AREA_UNIT})")
    _configure_y_axis(ax_a, y_slush)
    _configure_x_axis(ax_a, show_labels=True)
    _add_panel_label(ax_a, "(a)  Slush")

    _add_legend(ax_a, h_mean, h_max)

    # -- (b) Lake -----------------------------------------------------------
    h_mean_b, h_max_b = _plot_mean_max_line(ax_b, stats, "lake_mean", "lake_max", COLOUR_LAKE)
    ax_b.set_ylabel(f"Area ({AREA_UNIT})")
    _configure_y_axis(ax_b, y_lake)
    _configure_x_axis(ax_b, show_labels=True)
    _add_panel_label(ax_b, "(b)  Lake")
    _add_legend(ax_b, h_mean_b, h_max_b)

    # -- (c) Combined -------------------------------------------------------
    h_mean_c, h_max_c = _plot_mean_max_line(
        ax_c, stats, "combined_mean", "combined_max", COLOUR_COMBINED
    )
    ax_c.set_ylabel(f"Area ({AREA_UNIT})")
    _configure_y_axis(ax_c, y_combined)
    _configure_x_axis(ax_c, show_labels=True)
    _add_panel_label(ax_c, "(c)  Slush + lake")
    _add_legend(ax_c, h_mean_c, h_max_c)

    # -- (d) AOI coverage ---------------------------------------------------
    n_map = {yr: int(stats.loc[yr, "n"]) for yr in YEARS if yr in stats.index}
    _plot_coverage_bars(ax_d, stats)
    ax_d.set_ylabel("Coverage")
    ax_d.set_xlabel("Year (n = blocks per year)")
    _configure_x_axis(ax_d, show_labels=True, n_map=n_map)
    _add_panel_label(ax_d, "(d)  AOI coverage")

    # -- Figure title -------------------------------------------------------
    aoi_name  = AOI_NAMES.get(aoi, aoi)
    threshold = FOOTPRINT_COVERAGE_THRESHOLD
    fig.suptitle(
        f"Melt Area, {label} JJA, {aoi_name}, "
        f"(footprint coverage ≥ {threshold:.0%})",
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=0.995,
    )

    # -- Save ---------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved -> {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 9) MAIN
# ============================================================

# ============================================================
# CSV EXPORT
# ============================================================

_METRICS = [
    ("slush_mean",    "slush_mean"),
    ("slush_max",     "slush_max"),
    ("lake_mean",     "lake_mean"),
    ("lake_max",      "lake_max"),
    ("combined_mean", "combined_mean"),
    ("combined_max",  "combined_max"),
]


def _summary_rows(stats: pd.DataFrame, aoi: str, label: str) -> list:
    rows = []
    for yr, row in stats.iterrows():
        def _km2(col):
            v = row[col]
            return round(v / AREA_DIVISOR, 6) if not np.isnan(v) else np.nan
        rows.append({
            "aoi":               aoi,
            "label":             label,
            "year":              yr,
            "n":                 int(row["n"]),
            "coverage_mean":     round(row["coverage_mean"], 6) if not np.isnan(row["coverage_mean"]) else np.nan,
            "slush_mean_km2":    _km2("slush_mean"),
            "slush_max_km2":     _km2("slush_max"),
            "lake_mean_km2":     _km2("lake_mean"),
            "lake_max_km2":      _km2("lake_max"),
            "combined_mean_km2": _km2("combined_mean"),
            "combined_max_km2":  _km2("combined_max"),
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
        ("area_jja_summary.csv",  summary),
        ("area_mann_kendall.csv", mk),
        ("area_pettitt.csv",      pettitt),
    ):
        path = CSV_OUTPUT_DIR / fname
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Saved CSV -> {path}")


def main() -> None:
    if not PLOT_OBSERVED and not PLOT_SCALED:
        print("Both PLOT_OBSERVED and PLOT_SCALED are False - nothing to do.")
        return

    _apply_nature_rcparams()

    # -- Validate input files -----------------------------------------------
    required_obs = [TOTAL_SUMMARY_RAW_CSV]
    required_scaled = [
        TOTAL_SUMMARY_RAW_CSV,
        OBS_SLUSH_BINS_CSV, OBS_LAKE_BINS_CSV,
        PROJ_SLUSH_BINS_CSV, PROJ_LAKE_BINS_CSV,
    ]
    to_check = required_obs + (required_scaled if PLOT_SCALED else [])
    for p in dict.fromkeys(to_check):   # deduplicate, preserve order
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found:\n  {p}")

    # -- Load base data -----------------------------------------------------
    print("Loading raw block summary ...")
    base_df = _load_csv(TOTAL_SUMMARY_RAW_CSV)
    print(f"  {len(base_df)} total blocks")

    obs_df    = build_observed_df(base_df) if PLOT_OBSERVED else None
    scaled_df = build_scaled_df(base_df, AOIS) if PLOT_SCALED else None

    # -- Per-AOI processing and plotting -----------------------------------
    all_summary  = []
    all_mk       = []
    all_pettitt  = []

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
            out = OBS_OUTPUT_DIR / f"jja_mean_max_observed_{aoi}.png"
            make_figure(stats, aoi, "Observed", out)

        if PLOT_SCALED:
            print("\n[Scaled]")
            stats = compute_jja_stats(scaled_df, aoi)
            print_jja_summary(stats, aoi, "Scaled")
            print_trend_tests(stats, aoi, "Scaled")
            all_summary.extend(_summary_rows(stats, aoi, "Scaled"))
            all_mk.extend(_mk_rows(stats, aoi, "Scaled"))
            all_pettitt.extend(_pettitt_rows(stats, aoi, "Scaled"))
            out = SCALED_OUTPUT_DIR / f"jja_mean_max_scaled_{aoi}.png"
            make_figure(stats, aoi, "Scaled", out)

    print("\nSaving CSV outputs ...")
    _save_csvs(all_summary, all_mk, all_pettitt)
    print("\nDone.")


if __name__ == "__main__":
    main()
