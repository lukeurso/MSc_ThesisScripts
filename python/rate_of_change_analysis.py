# -*- coding: utf-8 -*-
"""
rate_of_change_analysis.py

Quantifies the rate of melt area recovery after cold/low-melt years
to test whether bounce-back is accelerating over time (supporting
the preconditioning hypothesis).

Inputs:
  - area_jja_summary.csv            (from plot_mean_max_area.py)
  - era5_jja_mean_max_summary.csv   (from plot_era5_mean_max.py)
  - area_pettitt.csv                (from plot_mean_max_area.py)
  - area_mann_kendall.csv           (from plot_mean_max_area.py)

Outputs:
  - era5_rate_of_change_cycles.csv
  - era5_rate_of_change_cycle_candidates.csv
  - era5_rate_of_change_pooled_met_models.csv
  - era5_rate_of_change_slopes.png
"""

from __future__ import annotations

from pathlib import Path

try:
    from IPython import get_ipython as _get_ipython
    _ip = _get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import linregress


# ============================================================
# 1) FILE PATHS
# ============================================================

AREA_SUMMARY_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
    r"\area_jja_summary.csv"
)
ERA5_SUMMARY_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
    r"\era5_jja_mean_max_summary.csv"
)
AREA_PETTITT_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
    r"\area_pettitt.csv"
)
AREA_MANN_KENDALL_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
    r"\area_mann_kendall.csv"
)
CSV_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
)
FIG_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\figures\plot_rate_of_change"
)


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]

AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

# Label to select from area_jja_summary.csv (matches plot_mean_max_area.py).
AREA_LABEL = "Observed"

# Primary area column used for all slope fitting.
AREA_METRIC = "combined_mean_km2"

# ERA5 predictor columns for met-normalised (climate-corrected) slopes.
# These are deliberately simple: one mean-temperature model, one
# max-temperature model, and one two-predictor model using the strongest
# temperature summary plus mean shortwave radiation.
ERA5_MODELS = {
    "tmean": {
        "label": "Tmean",
        "predictors": ["temp_2m_C_mean"],
    },
    "tmax": {
        "label": "Tmax",
        "predictors": ["temp_2m_C_max"],
    },
    "tmax_srad": {
        "label": "Tmax + SWmean",
        "predictors": ["temp_2m_C_max", "srad_Wm2_mean"],
    },
}

# ---- Cycle definitions: (trough_year, peak_year) ----------------------
# Each tuple defines one trough-to-peak recovery segment.
# All years between trough and peak (inclusive) are used for OLS fitting.
# Edit freely — OST and PTM can have different cycles.
CYCLES: dict[str, list[tuple[int, int]]] = {
    "OST": [(2017, 2019), (2020, 2023)],
    "PTM": [(2017, 2019), (2020, 2023)],
}

# ---- Bootstrap settings -----------------------------------------------
N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI   = 0.95   # two-sided confidence interval level


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

FIG_WIDTH_IN  = 7.087   # 180 mm (two-column)
FIG_HEIGHT_IN = 5.8     # two rows (one per AOI), two panels each

DPI_SCREEN = 150
DPI_SAVE   = 300
HSPACE     = 0.50
WSPACE     = 0.38

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

SPINE_WIDTH  = 0.6
TICK_WIDTH   = 0.6
TICK_LENGTH  = 3.0

GRID_LINESTYLE = ":"
GRID_ALPHA     = 0.45
GRID_LINEWIDTH = 0.5

LINE_WIDTH        = 1.2
MARKER_SIZE       = 3.5
MARKER_EDGE_WIDTH = 0.4
MARKER_EDGE_COLOR = "white"

PANEL_LABEL_X = -0.14
PANEL_LABEL_Y =  1.03

# One colour per cycle slot (index 0 = cycle 1, index 1 = cycle 2, …).
CYCLE_COLOURS = ["#1F78B4", "#E31A1C"]   # blue, red

BAR_ALPHA_RAW  = 0.85   # solid fill — raw slope
BAR_ALPHA_NORM = 0.45   # lighter hatched fill — met-normalised slope
BAR_WIDTH      = 0.18
BAR_OFFSETS    = {
    "raw":        -1.5 * BAR_WIDTH,
    "tmean":      -0.5 * BAR_WIDTH,
    "tmax":        0.5 * BAR_WIDTH,
    "tmax_srad":   1.5 * BAR_WIDTH,
}
MODEL_HATCHES = {
    "tmean": "///",
    "tmax": "\\\\\\",
    "tmax_srad": "xxx",
}


# ============================================================
# 4) STYLE HELPERS
# ============================================================

def _apply_rcparams() -> None:
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


def _add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, text,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )


def _sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ============================================================
# 5) DATA LOADING
# ============================================================

def load_area(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Area summary CSV not found:\n  {path}")
    df = pd.read_csv(path)
    required = {"aoi", "label", "year", AREA_METRIC}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"area_jja_summary.csv is missing columns: {missing}")
    df = df[(df["label"] == AREA_LABEL) & (df["aoi"].isin(AOIS))].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df[AREA_METRIC] = pd.to_numeric(df[AREA_METRIC], errors="coerce")
    return df.sort_values(["aoi", "year"]).reset_index(drop=True)


def load_era5(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ERA5 summary CSV not found:\n  {path}")
    df = pd.read_csv(path)
    required = {"aoi", "year"}
    for spec in ERA5_MODELS.values():
        required.update(spec["predictors"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"era5_jja_mean_max_summary.csv is missing columns: {missing}")
    df = df[df["aoi"].isin(AOIS)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for predictor in sorted(required - {"aoi", "year"}):
        df[predictor] = pd.to_numeric(df[predictor], errors="coerce")
    return df.sort_values(["aoi", "year"]).reset_index(drop=True)


def _series_name_from_metric(metric: str) -> str:
    """Map summary columns such as combined_mean_km2 to test-table series names."""
    if metric.endswith("_km2"):
        metric = metric[:-4]
    if metric.endswith("_Mm3"):
        metric = metric[:-4]
    return metric


def load_pettitt(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  Pettitt CSV not found; cycle metadata will be blank -> {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"aoi", "label", "series", "change_point_after_year", "p_value", "significance"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"area_pettitt.csv is missing columns: {missing}")
    df = df[
        (df["label"] == AREA_LABEL)
        & (df["aoi"].isin(AOIS))
        & (df["series"] == _series_name_from_metric(AREA_METRIC))
    ].copy()
    df["change_point_after_year"] = pd.to_numeric(
        df["change_point_after_year"], errors="coerce"
    )
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df.reset_index(drop=True)


def load_mann_kendall(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  Mann-Kendall CSV not found; trend metadata will be blank -> {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"aoi", "label", "series", "tau", "p_value", "significance", "trend"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"area_mann_kendall.csv is missing columns: {missing}")
    df = df[
        (df["label"] == AREA_LABEL)
        & (df["aoi"].isin(AOIS))
        & (df["series"] == _series_name_from_metric(AREA_METRIC))
    ].copy()
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df.reset_index(drop=True)


def _test_row(df: pd.DataFrame, aoi: str) -> pd.Series | None:
    if df.empty:
        return None
    sub = df[df["aoi"] == aoi]
    if sub.empty:
        return None
    return sub.iloc[0]


def _cycle_relation_to_cp(y0: int, y1: int, cp_year: float) -> str:
    if np.isnan(cp_year):
        return "unknown"
    cp = int(cp_year)
    if y1 <= cp:
        return "pre_or_through_cp"
    if y0 > cp:
        return "post_cp"
    return "crosses_cp"


# ============================================================
# 6) OLS + BOOTSTRAP
# ============================================================

def _fit_ols(years: np.ndarray, values: np.ndarray) -> dict:
    """OLS via scipy.stats.linregress. Returns NaN-filled dict when n < 3."""
    mask = ~np.isnan(values)
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    if len(x) < 3:
        return dict(
            slope=np.nan, intercept=np.nan, r_squared=np.nan,
            p_value=np.nan, std_err=np.nan, n=int(len(x)),
        )
    res = linregress(x, y)
    return dict(
        slope=float(res.slope),
        intercept=float(res.intercept),
        r_squared=float(res.rvalue ** 2),
        p_value=float(res.pvalue),
        std_err=float(res.stderr),
        n=int(len(x)),
    )


def _bootstrap_slope_ci(
    years: np.ndarray,
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float]:
    """
    Resample (year, value) pairs with replacement and refit OLS slope.
    Returns (lower, upper) percentile CI. Falls back to (nan, nan) if n < 3.
    """
    mask = ~np.isnan(values)
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_slopes = []
    attempts = 0
    max_attempts = n_boot * 10   # guard against infinite loop on pathological data
    while len(boot_slopes) < n_boot and attempts < max_attempts:
        idx = rng.integers(0, n, size=n)
        x_s, y_s = x[idx], y[idx]
        if np.ptp(x_s) == 0:   # all x identical — degenerate resample, skip
            attempts += 1
            continue
        boot_slopes.append(linregress(x_s, y_s).slope)
        attempts += 1
    if len(boot_slopes) < 2:
        return np.nan, np.nan
    boot_slopes = np.array(boot_slopes)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot_slopes, alpha)), float(np.quantile(boot_slopes, 1.0 - alpha))


# ============================================================
# 7) MET-NORMALISED RESIDUALS
# ============================================================

def _fit_predictor_model(
    df: pd.DataFrame,
    response_col: str,
    predictor_cols: list[str],
) -> tuple[pd.Series, dict]:
    """
    Fit OLS response ~ predictors with an intercept using numpy least squares.
    Returns residuals indexed like df plus compact model diagnostics.
    """
    model_df = df[[response_col, *predictor_cols]].dropna().copy()
    p = len(predictor_cols)
    if len(model_df) <= p + 1:
        return pd.Series(dtype=float), {
            "n": len(model_df), "r_squared": np.nan, "adj_r_squared": np.nan,
        }

    x_raw = model_df[predictor_cols].to_numpy(dtype=float)
    if np.linalg.matrix_rank(x_raw) < p:
        return pd.Series(dtype=float), {
            "n": len(model_df), "r_squared": np.nan, "adj_r_squared": np.nan,
        }

    X = np.column_stack([np.ones(len(model_df)), x_raw])
    y = model_df[response_col].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    residuals = y - fitted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = (
        1.0 - (1.0 - r2) * (len(model_df) - 1) / (len(model_df) - p - 1)
        if not np.isnan(r2) and len(model_df) > p + 1 else np.nan
    )
    return (
        pd.Series(residuals, index=model_df.index, name="residual"),
        {"n": len(model_df), "r_squared": r2, "adj_r_squared": adj_r2},
    )


def compute_residuals(
    area_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    aoi: str,
    predictor_cols: list[str],
) -> tuple[pd.Series, dict]:
    """
    Fit OLS of AREA_METRIC ~ ERA5 predictors over all available years for aoi.
    Residuals represent melt area variability not explained by ERA5 forcing.
    """
    a = area_df[area_df["aoi"] == aoi].set_index("year")[[AREA_METRIC]]
    e = era5_df[era5_df["aoi"] == aoi].set_index("year")[predictor_cols]
    merged = pd.concat([a, e], axis=1)
    residuals, diag = _fit_predictor_model(merged, AREA_METRIC, predictor_cols)
    residuals.index.name = "year"
    return residuals, diag


# ============================================================
# 8) PER-CYCLE ANALYSIS
# ============================================================

def analyse_cycles(
    area_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    pettitt_df: pd.DataFrame,
    mk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each AOI and each defined cycle fit raw OLS and met-normalised OLS.
    Returns a tidy DataFrame, one row per (aoi, cycle).
    """
    rows = []
    for aoi in AOIS:
        aoi_area = area_df[area_df["aoi"] == aoi].set_index("year")[AREA_METRIC]
        pettitt = _test_row(pettitt_df, aoi)
        mk = _test_row(mk_df, aoi)
        cp_year = (
            float(pettitt["change_point_after_year"])
            if pettitt is not None and not pd.isna(pettitt["change_point_after_year"])
            else np.nan
        )
        residual_sets = {
            model_id: compute_residuals(area_df, era5_df, aoi, spec["predictors"])
            for model_id, spec in ERA5_MODELS.items()
        }

        for cycle_num, (y0, y1) in enumerate(CYCLES.get(aoi, []), start=1):
            all_years = np.arange(y0, y1 + 1, dtype=float)
            raw_vals  = np.array([
                float(aoi_area[yr]) if yr in aoi_area.index else np.nan
                for yr in range(y0, y1 + 1)
            ])

            ols          = _fit_ols(all_years, raw_vals)
            ci_lo, ci_hi = _bootstrap_slope_ci(all_years, raw_vals)

            trough_val = float(aoi_area[y0]) if y0 in aoi_area.index else np.nan
            peak_val   = float(aoi_area[y1]) if y1 in aoi_area.index else np.nan
            total_rec  = (
                peak_val - trough_val
                if not (np.isnan(trough_val) or np.isnan(peak_val))
                else np.nan
            )
            frac_rate  = (
                ols["slope"] / trough_val
                if not (np.isnan(ols["slope"]) or np.isnan(trough_val) or trough_val == 0)
                else np.nan
            )

            def _r(v, dp=6):
                return round(float(v), dp) if not np.isnan(v) else np.nan

            row = {
                "aoi":                   aoi,
                "cycle_id":              cycle_num,
                "year_start":            y0,
                "year_end":              y1,
                "n_years":               y1 - y0 + 1,
                "raw_n_fit":             ols["n"],
                "trough_area_km2":       _r(trough_val),
                "peak_area_km2":         _r(peak_val),
                "total_recovery_km2":    _r(total_rec),
                # raw OLS
                "slope_km2_per_yr":      _r(ols["slope"]),
                "intercept":             _r(ols["intercept"]),
                "r_squared":             _r(ols["r_squared"]),
                "p_value":               _r(ols["p_value"], dp=8),
                "std_err":               _r(ols["std_err"]),
                "boot_ci_lo":            _r(ci_lo),
                "boot_ci_hi":            _r(ci_hi),
                "frac_rate_per_yr":      _r(frac_rate),
                # metadata
                "area_metric":           AREA_METRIC,
                "area_label":            AREA_LABEL,
                # distributional-change and monotonic-trend context
                "pettitt_cp_after_year":  int(cp_year) if not np.isnan(cp_year) else np.nan,
                "pettitt_p_value":        _r(float(pettitt["p_value"]), dp=8) if pettitt is not None else np.nan,
                "pettitt_significance":   pettitt["significance"] if pettitt is not None else "",
                "cycle_relation_to_cp":   _cycle_relation_to_cp(y0, y1, cp_year),
                "mk_tau":                 _r(float(mk["tau"])) if mk is not None else np.nan,
                "mk_p_value":             _r(float(mk["p_value"]), dp=8) if mk is not None else np.nan,
                "mk_significance":        mk["significance"] if mk is not None else "",
                "mk_trend":               mk["trend"] if mk is not None else "",
                "n_boot":                N_BOOTSTRAP,
                "boot_ci_level":         BOOTSTRAP_CI,
            }

            for model_id, spec in ERA5_MODELS.items():
                residuals, diag = residual_sets[model_id]
                resid_vals = np.array([
                    float(residuals[yr]) if yr in residuals.index else np.nan
                    for yr in range(y0, y1 + 1)
                ])
                norm_ols = _fit_ols(all_years, resid_vals)
                norm_ci_lo, norm_ci_hi = _bootstrap_slope_ci(all_years, resid_vals)
                prefix = f"norm_{model_id}"
                row[f"{prefix}_label"] = spec["label"]
                row[f"{prefix}_predictors"] = "+".join(spec["predictors"])
                row[f"{prefix}_model_n"] = diag["n"]
                row[f"{prefix}_model_r_squared"] = _r(diag["r_squared"])
                row[f"{prefix}_model_adj_r_squared"] = _r(diag["adj_r_squared"])
                row[f"{prefix}_n_fit"] = norm_ols["n"]
                row[f"{prefix}_slope_km2_per_yr"] = _r(norm_ols["slope"])
                row[f"{prefix}_r_squared"] = _r(norm_ols["r_squared"])
                row[f"{prefix}_p_value"] = _r(norm_ols["p_value"], dp=8)
                row[f"{prefix}_boot_ci_lo"] = _r(norm_ci_lo)
                row[f"{prefix}_boot_ci_hi"] = _r(norm_ci_hi)

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 9) CONSOLE SUMMARY
# ============================================================

def _cis_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    if any(np.isnan(v) for v in (lo1, hi1, lo2, hi2)):
        return True   # treat unknown as overlap (conservative)
    return lo1 <= hi2 and lo2 <= hi1


def print_summary(results: pd.DataFrame) -> None:
    SEP = "  " + "-" * 76
    ci_pct = int(BOOTSTRAP_CI * 100)

    for aoi in AOIS:
        sub = results[results["aoi"] == aoi].reset_index(drop=True)
        print(f"\n{'=' * 80}")
        print(f"  {aoi}  ({AOI_NAMES.get(aoi, '')})")
        print(f"{'=' * 80}")
        if not sub.empty and "pettitt_cp_after_year" in sub:
            first = sub.iloc[0]
            print(
                f"  Pettitt context: CP after {int(first['pettitt_cp_after_year'])} "
                f"(p={first['pettitt_p_value']:.4f}, {first['pettitt_significance']}); "
                f"Mann-Kendall tau={first['mk_tau']:.3f} "
                f"(p={first['mk_p_value']:.4f}, {first['mk_trend']})"
            )

        # --- raw slopes ---
        print(f"\n  Raw slopes  (AREA_METRIC={AREA_METRIC}, AREA_LABEL={AREA_LABEL})")
        print(
            f"  {'Cycle':<9} {'Years':<11} {'n':>3}  "
            f"{'Slope (km²/yr)':>15}  {f'Boot {ci_pct}% CI':>22}  {'R²':>6}  {'p':>8}  "
            f"{'Frac rate (%/yr)':>17}"
        )
        print(SEP)
        for _, row in sub.iterrows():
            ci_str = f"[{row['boot_ci_lo']:+.3f}, {row['boot_ci_hi']:+.3f}]"
            fr_str = (
                f"{row['frac_rate_per_yr'] * 100:+.1f}"
                if not np.isnan(row["frac_rate_per_yr"]) else "-"
            )
            print(
                f"  Cycle {int(row['cycle_id']):<3}  "
                f"{int(row['year_start'])}–{int(row['year_end'])}   "
                f"{int(row['raw_n_fit']):>3}  "
                f"{row['slope_km2_per_yr']:>15.4f}  "
                f"{ci_str:>22}  "
                f"{row['r_squared']:>6.3f}  "
                f"{row['p_value']:>8.4f}  "
                f"{fr_str:>17}"
            )

        if len(sub) >= 2:
            print(SEP)
            for i in range(len(sub)):
                for j in range(i + 1, len(sub)):
                    r1, r2  = sub.iloc[i], sub.iloc[j]
                    delta   = r2["slope_km2_per_yr"] - r1["slope_km2_per_yr"]
                    overlap = _cis_overlap(
                        r1["boot_ci_lo"], r1["boot_ci_hi"],
                        r2["boot_ci_lo"], r2["boot_ci_hi"],
                    )
                    verdict = "CIs overlap — not significant" if overlap else "CIs non-overlapping  (*)"
                    print(
                        f"  Δslope (cycle {int(r2['cycle_id'])} − cycle {int(r1['cycle_id'])}): "
                        f"{delta:+.4f} km² yr⁻¹  →  {verdict}"
                    )

        # --- met-normalised slopes ---
        for model_id, spec in ERA5_MODELS.items():
            prefix = f"norm_{model_id}"
            print(f"\n  Met-normalised slopes  ({spec['label']}: {' + '.join(spec['predictors'])})")
            print(
                f"  {'Cycle':<9} {'Years':<11} {'n':>3}  {'Norm slope':>12}  "
                f"{f'Boot {ci_pct}% CI':>22}  {'R²':>6}  {'p':>8}"
            )
            print(SEP)
            for _, row in sub.iterrows():
                ci_str = f"[{row[f'{prefix}_boot_ci_lo']:+.3f}, {row[f'{prefix}_boot_ci_hi']:+.3f}]"
                print(
                    f"  Cycle {int(row['cycle_id']):<3}  "
                    f"{int(row['year_start'])}–{int(row['year_end'])}   "
                    f"{int(row[f'{prefix}_n_fit']):>3}  "
                    f"{row[f'{prefix}_slope_km2_per_yr']:>12.4f}  "
                    f"{ci_str:>22}  "
                    f"{row[f'{prefix}_r_squared']:>6.3f}  "
                    f"{row[f'{prefix}_p_value']:>8.4f}"
                )

            if len(sub) >= 2:
                print(SEP)
                for i in range(len(sub)):
                    for j in range(i + 1, len(sub)):
                        r1, r2  = sub.iloc[i], sub.iloc[j]
                        delta   = r2[f"{prefix}_slope_km2_per_yr"] - r1[f"{prefix}_slope_km2_per_yr"]
                        overlap = _cis_overlap(
                            r1[f"{prefix}_boot_ci_lo"], r1[f"{prefix}_boot_ci_hi"],
                            r2[f"{prefix}_boot_ci_lo"], r2[f"{prefix}_boot_ci_hi"],
                        )
                        verdict = "CIs overlap — not significant" if overlap else "CIs non-overlapping  (*)"
                        print(
                            f"  Δ{spec['label']} residual slope "
                            f"(cycle {int(r2['cycle_id'])} − cycle {int(r1['cycle_id'])}): "
                            f"{delta:+.4f} km² yr⁻¹  →  {verdict}"
                        )


# ============================================================
# 10) FIGURE
# ============================================================

# Panel label grid: row 0 = (a),(b); row 1 = (c),(d)
_PANEL_LABELS = [["(a)", "(b)"], ["(c)", "(d)"]]


def _get_aoi_series(area_df: pd.DataFrame, aoi: str) -> pd.Series:
    return area_df[area_df["aoi"] == aoi].set_index("year")[AREA_METRIC]


def _draw_trajectories(
    ax: plt.Axes,
    aoi_res: pd.DataFrame,
    aoi_area: pd.Series,
    row_idx: int,
    aoi: str,
) -> None:
    """Left panel: recovery trajectories aligned to trough (Δyears from trough)."""
    for _, cycle_row in aoi_res.iterrows():
        y0     = int(cycle_row["year_start"])
        y1     = int(cycle_row["year_end"])
        colour = CYCLE_COLOURS[(int(cycle_row["cycle_id"]) - 1) % len(CYCLE_COLOURS)]
        label  = f"Cycle {int(cycle_row['cycle_id'])}: {y0}–{y1}"

        d_years  = np.arange(0, y1 - y0 + 1, dtype=float)
        raw_vals = np.array([
            float(aoi_area[yr]) if yr in aoi_area.index else np.nan
            for yr in range(y0, y1 + 1)
        ])

        valid = ~np.isnan(raw_vals)
        ax.plot(
            d_years[valid], raw_vals[valid],
            color=colour, linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, markeredgecolor=MARKER_EDGE_COLOR,
            label=label, zorder=3,
        )

        # Dashed OLS fit line evaluated at Δyear positions
        if not np.isnan(cycle_row["slope_km2_per_yr"]) and valid.sum() >= 2:
            x_fit = np.array([d_years[valid][0], d_years[valid][-1]])
            # year = y0 + delta_year  →  area = intercept + slope * year
            y_fit = cycle_row["intercept"] + cycle_row["slope_km2_per_yr"] * (x_fit + y0)
            ax.plot(x_fit, y_fit, color=colour, linewidth=0.7, linestyle="--", zorder=2)

    ax.set_xlabel("Years from trough")
    ax.set_ylabel("Combined melt area (km²)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH, zorder=0)
    ax.legend(fontsize=LEGEND_FONT_SIZE, frameon=True, framealpha=0.9,
              edgecolor="#cccccc", loc="upper left")
    _add_panel_label(ax, f"{_PANEL_LABELS[row_idx][0]}  {AOI_NAMES.get(aoi, aoi)}")


def _draw_slope_bars(
    ax: plt.Axes,
    aoi_res: pd.DataFrame,
    row_idx: int,
) -> None:
    """Right panel: grouped bar chart — raw slope plus met-normalised residual slopes."""
    n_cycles = len(aoi_res)
    x_pos    = np.arange(n_cycles, dtype=float)

    for pos, (_, cycle_row) in enumerate(aoi_res.iterrows()):
        colour    = CYCLE_COLOURS[(int(cycle_row["cycle_id"]) - 1) % len(CYCLE_COLOURS)]
        raw_slope = cycle_row["slope_km2_per_yr"]

        # asymmetric bootstrap CI error bars
        raw_yerr = (
            [[raw_slope - cycle_row["boot_ci_lo"]], [cycle_row["boot_ci_hi"] - raw_slope]]
            if not np.isnan(raw_slope) else [[0], [0]]
        )

        # raw bar
        ax.bar(
            x_pos[pos] + BAR_OFFSETS["raw"], raw_slope,
            width=BAR_WIDTH, color=colour, alpha=BAR_ALPHA_RAW,
            edgecolor="white", linewidth=0.4, zorder=3,
        )
        ax.errorbar(
            x_pos[pos] + BAR_OFFSETS["raw"], raw_slope,
            yerr=raw_yerr, fmt="none",
            color="#333333", linewidth=0.8, capsize=2.5, zorder=4,
        )

        for model_id in ERA5_MODELS:
            prefix = f"norm_{model_id}"
            nrm_slope = cycle_row[f"{prefix}_slope_km2_per_yr"]
            nrm_yerr = (
                [[nrm_slope - cycle_row[f"{prefix}_boot_ci_lo"]],
                 [cycle_row[f"{prefix}_boot_ci_hi"] - nrm_slope]]
                if not np.isnan(nrm_slope) else [[0], [0]]
            )
            ax.bar(
                x_pos[pos] + BAR_OFFSETS[model_id], nrm_slope,
                width=BAR_WIDTH, color=colour, alpha=BAR_ALPHA_NORM,
                edgecolor="white", linewidth=0.4,
                hatch=MODEL_HATCHES[model_id], zorder=3,
            )
            ax.errorbar(
                x_pos[pos] + BAR_OFFSETS[model_id], nrm_slope,
                yerr=nrm_yerr, fmt="none",
                color="#333333", linewidth=0.8, capsize=2.5, zorder=4,
            )

    # Δslope annotation between first two cycles
    if n_cycles >= 2:
        r1 = aoi_res.iloc[0]
        r2 = aoi_res.iloc[1]
        delta   = r2["slope_km2_per_yr"] - r1["slope_km2_per_yr"]
        overlap = _cis_overlap(
            r1["boot_ci_lo"], r1["boot_ci_hi"],
            r2["boot_ci_lo"], r2["boot_ci_hi"],
        )
        sig = "ns" if overlap else "*"
        ax.annotate(
            f"Raw Δ = {delta:+.2f} km²/yr\n{sig}",
            xy=(0.5, 1.01), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=TICK_FONT_SIZE,
        )

    ax.axhline(0, color="#333333", linewidth=0.6, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"Cycle {int(r['cycle_id'])}" for _, r in aoi_res.iterrows()],
        fontsize=TICK_FONT_SIZE,
    )
    ax.set_ylabel("Slope (km² yr⁻¹)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA,
            linewidth=GRID_LINEWIDTH, zorder=0)
    _add_panel_label(ax, _PANEL_LABELS[row_idx][1])


def make_figure(
    results: pd.DataFrame,
    area_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    out_path: Path,
) -> None:
    n_aoi = len(AOIS)
    fig, axes = plt.subplots(
        n_aoi, 2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    if n_aoi == 1:
        axes = axes[np.newaxis, :]
    fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    for row_idx, aoi in enumerate(AOIS):
        aoi_res  = results[results["aoi"] == aoi].reset_index(drop=True)
        aoi_area = _get_aoi_series(area_df, aoi)

        _draw_trajectories(axes[row_idx, 0], aoi_res, aoi_area, row_idx, aoi)
        _draw_slope_bars(axes[row_idx, 1], aoi_res, row_idx)

    # Shared bar-type legend anchored below the right column
    legend_handles = [
        mpatches.Patch(facecolor="#888888", alpha=BAR_ALPHA_RAW,
                       label="Raw slope"),
    ]
    for model_id, spec in ERA5_MODELS.items():
        legend_handles.append(
            mpatches.Patch(
                facecolor="#888888", alpha=BAR_ALPHA_NORM,
                hatch=MODEL_HATCHES[model_id],
                label=f"Residual ({spec['label']})",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.72, -0.03),
    )

    fig.suptitle(
        f"Melt area recovery rate by cycle  ({AREA_LABEL}, {AREA_METRIC})",
        fontsize=BASE_FONT_SIZE + 1,
        fontweight="bold",
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved figure -> {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 11) CSV EXPORT
# ============================================================

def find_cycle_candidates(
    area_df: pd.DataFrame,
    pettitt_df: pd.DataFrame,
    mk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    List simple trough-to-subsequent-peak candidates with at least three years.
    The selected CYCLES remain hand-defined; this table documents alternatives.
    """
    rows = []
    for aoi in AOIS:
        s = _get_aoi_series(area_df, aoi).sort_index()
        years = list(s.index.astype(int))
        vals = s.to_dict()
        pettitt = _test_row(pettitt_df, aoi)
        mk = _test_row(mk_df, aoi)
        cp_year = (
            float(pettitt["change_point_after_year"])
            if pettitt is not None and not pd.isna(pettitt["change_point_after_year"])
            else np.nan
        )

        def _append_candidate(start: int, end: int, source: str) -> None:
            if start not in vals or end not in vals or end <= start:
                return
            if end - start + 1 < 3 or vals[end] <= vals[start]:
                return
            window_years = np.arange(start, end + 1, dtype=float)
            window_vals = np.array([
                float(vals[y]) if y in vals else np.nan
                for y in range(start, end + 1)
            ])
            ols = _fit_ols(window_years, window_vals)
            ci_lo, ci_hi = _bootstrap_slope_ci(window_years, window_vals)
            rows.append({
                "aoi": aoi,
                "year_start": start,
                "year_end": end,
                "n_years": end - start + 1,
                "start_area_km2": round(float(vals[start]), 6),
                "end_area_km2": round(float(vals[end]), 6),
                "total_recovery_km2": round(float(vals[end] - vals[start]), 6),
                "slope_km2_per_yr": round(float(ols["slope"]), 6) if not np.isnan(ols["slope"]) else np.nan,
                "boot_ci_lo": round(float(ci_lo), 6) if not np.isnan(ci_lo) else np.nan,
                "boot_ci_hi": round(float(ci_hi), 6) if not np.isnan(ci_hi) else np.nan,
                "is_selected_cycle": (start, end) in CYCLES.get(aoi, []),
                "candidate_source": source,
                "pettitt_cp_after_year": int(cp_year) if not np.isnan(cp_year) else np.nan,
                "pettitt_p_value": round(float(pettitt["p_value"]), 8) if pettitt is not None else np.nan,
                "pettitt_significance": pettitt["significance"] if pettitt is not None else "",
                "cycle_relation_to_cp": _cycle_relation_to_cp(start, end, cp_year),
                "mk_tau": round(float(mk["tau"]), 6) if mk is not None else np.nan,
                "mk_p_value": round(float(mk["p_value"]), 8) if mk is not None else np.nan,
                "mk_significance": mk["significance"] if mk is not None else "",
                "mk_trend": mk["trend"] if mk is not None else "",
            })

        for start, end in CYCLES.get(aoi, []):
            _append_candidate(start, end, "selected")

        for start in years:
            future_years = [y for y in years if y > start]
            if len(future_years) < 2:
                continue
            end = max(future_years, key=lambda y: vals[y])
            if (start, end) in CYCLES.get(aoi, []):
                continue
            _append_candidate(start, end, "start_to_later_max")
    return pd.DataFrame(rows).sort_values(
        ["aoi", "is_selected_cycle", "year_start"],
        ascending=[True, False, True],
    )


def pooled_met_models(area_df: pd.DataFrame, era5_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit pooled met models with an AOI fixed effect. This uses n=24 while
    avoiding a naive comparison of OST and PTM absolute areas as one glacier.
    """
    merged = area_df[["aoi", "year", AREA_METRIC]].merge(
        era5_df[["aoi", "year", *sorted({p for spec in ERA5_MODELS.values() for p in spec["predictors"]})]],
        on=["aoi", "year"],
        how="inner",
    )
    merged["aoi_ptm"] = (merged["aoi"] == "PTM").astype(float)
    rows = []
    for model_id, spec in ERA5_MODELS.items():
        predictors = ["aoi_ptm", *spec["predictors"]]
        residuals, diag = _fit_predictor_model(merged, AREA_METRIC, predictors)
        rows.append({
            "model_id": model_id,
            "model_label": spec["label"],
            "predictors": "+".join(predictors),
            "n": diag["n"],
            "r_squared": round(float(diag["r_squared"]), 6) if not np.isnan(diag["r_squared"]) else np.nan,
            "adj_r_squared": round(float(diag["adj_r_squared"]), 6) if not np.isnan(diag["adj_r_squared"]) else np.nan,
            "residual_sd_km2": round(float(residuals.std(ddof=1)), 6) if len(residuals) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def save_csv(results: pd.DataFrame) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_OUTPUT_DIR / "era5_rate_of_change_cycles.csv"
    results.to_csv(out, index=False)
    print(f"  Saved CSV -> {out}")


def save_supporting_csvs(
    area_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    pettitt_df: pd.DataFrame,
    mk_df: pd.DataFrame,
) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = find_cycle_candidates(area_df, pettitt_df, mk_df)
    out_candidates = CSV_OUTPUT_DIR / "era5_rate_of_change_cycle_candidates.csv"
    candidates.to_csv(out_candidates, index=False)
    print(f"  Saved cycle candidates -> {out_candidates}")

    pooled = pooled_met_models(area_df, era5_df)
    out_pooled = CSV_OUTPUT_DIR / "era5_rate_of_change_pooled_met_models.csv"
    pooled.to_csv(out_pooled, index=False)
    print(f"  Saved pooled met models -> {out_pooled}")

    if not candidates.empty:
        print("\n  Selected/candidate recovery windows:")
        print(
            candidates[
                ["aoi", "year_start", "year_end", "n_years",
                 "total_recovery_km2", "slope_km2_per_yr",
                 "pettitt_cp_after_year", "cycle_relation_to_cp",
                 "is_selected_cycle"]
            ].to_string(index=False)
        )


# ============================================================
# 12) MAIN
# ============================================================

def main() -> None:
    _apply_rcparams()

    print("Loading area summary ...")
    area_df = load_area(AREA_SUMMARY_CSV)
    print(f"  {len(area_df)} rows  ({AREA_LABEL})")

    print("Loading ERA5 summary ...")
    era5_df = load_era5(ERA5_SUMMARY_CSV)
    print(f"  {len(era5_df)} rows")

    print("Loading Pettitt / Mann-Kendall context ...")
    pettitt_df = load_pettitt(AREA_PETTITT_CSV)
    mk_df = load_mann_kendall(AREA_MANN_KENDALL_CSV)
    print(f"  Pettitt rows: {len(pettitt_df)}; Mann-Kendall rows: {len(mk_df)}")

    print(f"\nRunning cycle analysis  ({N_BOOTSTRAP} bootstrap resamples) ...")
    results = analyse_cycles(area_df, era5_df, pettitt_df, mk_df)
    print_summary(results)

    print("\nSaving CSV ...")
    save_csv(results)
    save_supporting_csvs(area_df, era5_df, pettitt_df, mk_df)

    print("\nBuilding figure ...")
    out_fig = FIG_OUTPUT_DIR / "era5_rate_of_change_slopes.png"
    make_figure(results, area_df, era5_df, out_fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
