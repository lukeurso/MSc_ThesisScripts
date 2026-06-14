# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:50:29 2026

@author: Lukeu

plot_era5_r2_hysteresis.py

Computes R^2 (Pearson) and hysteresis diagnostics between ERA5-Land JJA
statistics and observed melt response variables (area, elevation, volume).

Inputs (all produced by upstream scripts):
  era5_jja_mean_max_summary.csv  <- plot_era5_timeseries.py
  area_jja_summary.csv           <- plot_mean_max_area.py
  elev_jja_summary.csv           <- plot_mean_max_elev.py
  volume_jja_summary.csv         <- plot_mean_max_volume.py

Outputs:
  era5_r2_scatter_{type}_{AOI}.png  (one per response type per AOI)
  era5_r2_scatter_pooled_{type}.png  (pooled AOI scatter with AOI fixed effect)
  era5_r2_summary.csv
  era5_r2_pooled_summary.csv
  era5_hysteresis_analog_pairs.csv
  era5_hysteresis_lag_corr.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# 1) FILE PATHS
# ============================================================

CSV_INPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
)

ERA5_SUMMARY_CSV = CSV_INPUT_DIR / "era5_jja_mean_max_summary.csv"

FIG_OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_era5_mean_max")

CSV_OUTPUT_DIR = CSV_INPUT_DIR

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

# ERA5 predictor columns in era5_jja_mean_max_summary.csv.
ERA5_PREDICTORS = {
    "temp_2m_C_mean": "T2m JJA mean (deg C)",
    "temp_2m_C_max":  "T2m JJA max (deg C)",
    "srad_Wm2_mean":  "SW down JJA mean (W m^-2)",
    "srad_Wm2_max":   "SW down JJA max (W m^-2)",
}

# ERA5 predictor rows shown in all scatter figures (must be keys of ERA5_PREDICTORS).
R2_FIG_ROWS = ["temp_2m_C_mean", "srad_Wm2_mean"]

# Response variable sources.
#
# csv            : summary CSV produced by the corresponding timeseries script
# label          : value of the 'label' column to filter on; None = no filter
# metrics        : all predictor x response pairs used in R^2 / hysteresis analysis
# fig_cols       : response columns shown as grid columns in the scatter figure
# fig_col_titles : human-readable title for each fig_col (column header)
# unit           : y-axis label for the scatter figure panels
RESPONSE_SOURCES = {
    "area": {
        "csv":   CSV_INPUT_DIR / "area_jja_summary.csv",
        "label": "Observed",
        "metrics": {
            "slush_mean_km2":    "Slush mean (km^2)",
            "slush_max_km2":     "Slush max (km^2)",
            "lake_mean_km2":     "Lake mean (km^2)",
            "lake_max_km2":      "Lake max (km^2)",
            "combined_mean_km2": "Combined mean (km^2)",
            "combined_max_km2":  "Combined max (km^2)",
        },
        "fig_cols": ["slush_mean_km2", "lake_mean_km2", "combined_mean_km2"],
        "fig_col_titles": {
            "slush_mean_km2":    "Slush",
            "lake_mean_km2":     "Lake",
            "combined_mean_km2": "Slush + lake",
        },
        "unit": "Area (km^2)",
    },
    "elev": {
        "csv":   CSV_INPUT_DIR / "elev_jja_summary.csv",
        "label": None,
        "metrics": {
            "slush_mean_m":    "Slush mean (m a.s.l.)",
            "slush_max_m":     "Slush max (m a.s.l.)",
            "lake_mean_m":     "Lake mean (m a.s.l.)",
            "lake_max_m":      "Lake max (m a.s.l.)",
            "combined_mean_m": "Combined mean (m a.s.l.)",
            "combined_max_m":  "Combined max (m a.s.l.)",
        },
        "fig_cols": ["slush_mean_m", "lake_mean_m", "combined_mean_m"],
        "fig_col_titles": {
            "slush_mean_m":    "Slush",
            "lake_mean_m":     "Lake",
            "combined_mean_m": "Slush + lake",
        },
        "unit": "Elevation (m a.s.l.)",
    },
    "volume": {
        "csv":   CSV_INPUT_DIR / "volume_jja_summary.csv",
        "label": "Observed",
        "metrics": {
            "lake_mean_Mm3": "Lake mean (Mm^3)",
            "lake_max_Mm3":  "Lake max (Mm^3)",
        },
        "fig_cols": ["lake_mean_Mm3", "lake_max_Mm3"],
        "fig_col_titles": {
            "lake_mean_Mm3": "Lake mean",
            "lake_max_Mm3":  "Lake max",
        },
        "unit": "Volume (Mm^3)",
    },
}

# Hysteresis detection thresholds (in z-score units).
HYSTERESIS_MET_SD = 0.5   # |Delta met z| < threshold  -> "similar" climate years
HYSTERESIS_RES_SD = 1.0   # |Delta resid z| > threshold -> "divergent" melt response


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

FIG_WIDTH_IN      = 7.087
R2_FIG_HEIGHT_IN  = 5.0
DPI_SCREEN        = 150
DPI_SAVE          = 300

FONT_FAMILY       = "sans-serif"
BASE_FONT_SIZE    = 7
TICK_FONT_SIZE    = 6
LABEL_FONT_SIZE   = 7
LEGEND_FONT_SIZE  = 6

SPINE_WIDTH       = 0.6
TICK_WIDTH        = 0.6
TICK_LENGTH       = 3.0

R2_SCATTER_COLOUR = "#444444"
R2_FIT_COLOUR     = "#B2182B"
POOLED_AOI_STYLE = {
    "OST": {"colour": "#1F78B4", "marker": "o"},
    "PTM": {"colour": "#E31A1C", "marker": "s"},
}


# ============================================================
# 4) DATA LOADING
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


def load_era5_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 summary CSV not found. Run plot_era5_timeseries.py first.\n  {path}"
        )
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def load_response_source(source: dict) -> pd.DataFrame | None:
    path = source["csv"]
    if not path.exists():
        print(f"  WARNING: Response CSV not found - skipping.\n  Expected: {path}")
        return None
    df = pd.read_csv(path)
    label = source.get("label")
    if label is not None:
        if "label" not in df.columns:
            print(f"  WARNING: No 'label' column in {path.name} - using all rows.")
        else:
            df = df[df["label"] == label].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    for col in source["metrics"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ============================================================
# 5) MERGE AND ANALYSIS
# ============================================================

def _merge_era5_response(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    aoi: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Inner-join ERA5 JJA stats with response stats for one AOI on year."""
    era5_aoi = era5_stats[era5_stats["aoi"] == aoi].set_index("year")
    resp_cols = [c for c in metric_cols if c in response_df.columns]
    resp_aoi = (
        response_df[response_df["aoi"] == aoi]
        .dropna(subset=["year"])
        .set_index("year")[resp_cols]
    )
    return era5_aoi.join(resp_aoi, how="inner").sort_index()


def _sig(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def compute_r_squared(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    aoi: str,
    source_key: str,
    metrics: dict[str, str],
) -> pd.DataFrame:
    """
    Compute Pearson r, R^2, and p-value for every ERA5 predictor x response
    metric pair.  Returns a tidy DataFrame, one row per pair.
    """
    merged = _merge_era5_response(era5_stats, response_df, aoi, list(metrics.keys()))
    rows = []
    for pred_col, pred_label in ERA5_PREDICTORS.items():
        if pred_col not in merged.columns:
            continue
        for resp_col, resp_label in metrics.items():
            if resp_col not in merged.columns:
                continue
            sub = merged[[pred_col, resp_col]].dropna()
            if len(sub) < 4:
                continue
            r, p = pearsonr(sub[pred_col].values, sub[resp_col].values)
            rows.append({
                "response_type":   source_key,
                "aoi":             aoi,
                "predictor":       pred_col,
                "predictor_label": pred_label,
                "response":        resp_col,
                "response_label":  resp_label,
                "n":               len(sub),
                "r":               round(float(r), 4),
                "r_squared":       round(float(r) ** 2, 4),
                "p_value":         round(float(p), 6),
                "sig":             _sig(float(p)),
            })
    return pd.DataFrame(rows)


def _fit_linear_model(
    df: pd.DataFrame,
    response_col: str,
    predictor_cols: list[str],
) -> dict:
    """Fit response ~ predictors with an intercept; return compact R^2 diagnostics."""
    sub = df[[response_col, *predictor_cols]].dropna().copy()
    p = len(predictor_cols)
    if len(sub) <= p + 1:
        return {"n": len(sub), "r_squared": np.nan, "adj_r_squared": np.nan}

    x_raw = sub[predictor_cols].to_numpy(dtype=float)
    if np.linalg.matrix_rank(x_raw) < p:
        return {"n": len(sub), "r_squared": np.nan, "adj_r_squared": np.nan}

    X = np.column_stack([np.ones(len(sub)), x_raw])
    y = sub[response_col].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    residuals = y - fitted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = (
        1.0 - (1.0 - r2) * (len(sub) - 1) / (len(sub) - p - 1)
        if not np.isnan(r2) and len(sub) > p + 1 else np.nan
    )
    return {"n": len(sub), "r_squared": r2, "adj_r_squared": adj_r2}


def _fit_linear_model_full(
    df: pd.DataFrame,
    response_col: str,
    predictor_cols: list[str],
) -> dict:
    """Fit response ~ predictors with an intercept and keep coefficients."""
    sub = df[[response_col, *predictor_cols]].dropna().copy()
    p = len(predictor_cols)
    if len(sub) <= p + 1:
        return {
            "n": len(sub), "r_squared": np.nan, "adj_r_squared": np.nan,
            "intercept": np.nan, "coefficients": {},
        }

    x_raw = sub[predictor_cols].to_numpy(dtype=float)
    if np.linalg.matrix_rank(x_raw) < p:
        return {
            "n": len(sub), "r_squared": np.nan, "adj_r_squared": np.nan,
            "intercept": np.nan, "coefficients": {},
        }

    X = np.column_stack([np.ones(len(sub)), x_raw])
    y = sub[response_col].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    residuals = y - fitted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj_r2 = (
        1.0 - (1.0 - r2) * (len(sub) - 1) / (len(sub) - p - 1)
        if not np.isnan(r2) and len(sub) > p + 1 else np.nan
    )
    return {
        "n": len(sub),
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "intercept": float(coef[0]),
        "coefficients": {
            col: float(value) for col, value in zip(predictor_cols, coef[1:])
        },
    }


def _pool_era5_response(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Inner-join ERA5 and response data for both AOIs, preserving AOI labels."""
    merged_parts = []
    for aoi in AOIS:
        merged = _merge_era5_response(era5_stats, response_df, aoi, metric_cols)
        if merged.empty:
            continue
        merged["aoi"] = aoi
        merged_parts.append(merged.reset_index())
    if not merged_parts:
        return pd.DataFrame()
    pooled = pd.concat(merged_parts, ignore_index=True)
    pooled["aoi_ptm"] = (pooled["aoi"] == "PTM").astype(float)
    return pooled


def compute_pooled_r_squared(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    source_key: str,
    metrics: dict[str, str],
) -> pd.DataFrame:
    """
    Compute pooled AOI R^2 diagnostics. The AOI-fixed-effect model is the
    preferred pooled result because it allows OST and PTM different baselines.
    """
    pooled = _pool_era5_response(era5_stats, response_df, list(metrics.keys()))
    if pooled.empty:
        return pd.DataFrame()

    rows = []
    for pred_col, pred_label in ERA5_PREDICTORS.items():
        if pred_col not in pooled.columns:
            continue
        for resp_col, resp_label in metrics.items():
            if resp_col not in pooled.columns:
                continue

            raw = _fit_linear_model(pooled, resp_col, [pred_col])
            fixed = _fit_linear_model(pooled, resp_col, ["aoi_ptm", pred_col])

            for model_type, diag, predictors in (
                ("pooled_raw", raw, pred_col),
                ("pooled_aoi_fixed_effect", fixed, f"aoi_ptm+{pred_col}"),
            ):
                rows.append({
                    "response_type":   source_key,
                    "model_type":      model_type,
                    "predictor":       pred_col,
                    "predictor_label": pred_label,
                    "response":        resp_col,
                    "response_label":  resp_label,
                    "predictors":      predictors,
                    "n":               diag["n"],
                    "r_squared":       round(float(diag["r_squared"]), 4) if not np.isnan(diag["r_squared"]) else np.nan,
                    "adj_r_squared":   round(float(diag["adj_r_squared"]), 4) if not np.isnan(diag["adj_r_squared"]) else np.nan,
                })
    return pd.DataFrame(rows)


def test_hysteresis(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    aoi: str,
    source_key: str,
    metrics: dict[str, str],
) -> pd.DataFrame:
    """
    Identify analog year pairs: years where ERA5 forcing is similar
    (|Delta met z-score| < HYSTERESIS_MET_SD) but response residuals diverge
    (|Delta residual z-score| > HYSTERESIS_RES_SD).

    For each predictor x response pair an OLS line is fitted; residuals
    represent response variability not explained by the climate variable.
    Large residuals in similar-climate year pairs indicate state-dependent
    (hysteretic) behaviour.
    """
    merged = _merge_era5_response(era5_stats, response_df, aoi, list(metrics.keys()))
    rows = []
    for pred_col in ERA5_PREDICTORS:
        if pred_col not in merged.columns:
            continue
        for resp_col in metrics:
            if resp_col not in merged.columns:
                continue
            sub = merged[[pred_col, resp_col]].dropna()
            if len(sub) < 5:
                continue

            x = sub[pred_col].values.astype(float)
            y = sub[resp_col].values.astype(float)
            years = list(sub.index)

            coeffs = np.polyfit(x, y, 1)
            resid  = y - np.polyval(coeffs, x)

            x_z   = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
            res_z = (resid - resid.mean()) / (resid.std(ddof=1) + 1e-12)

            n = len(years)
            for i in range(n):
                for j in range(i + 1, n):
                    met_diff = abs(x_z[i] - x_z[j])
                    res_diff = abs(res_z[i] - res_z[j])
                    if met_diff < HYSTERESIS_MET_SD and res_diff > HYSTERESIS_RES_SD:
                        rows.append({
                            "response_type": source_key,
                            "aoi":           aoi,
                            "predictor":     pred_col,
                            "response":      resp_col,
                            "year_a":        years[i],
                            "year_b":        years[j],
                            "met_val_a":     round(float(x[i]), 3),
                            "met_val_b":     round(float(x[j]), 3),
                            "met_z_diff":    round(float(met_diff), 3),
                            "resp_val_a":    round(float(y[i]), 4),
                            "resp_val_b":    round(float(y[j]), 4),
                            "resid_z_diff":  round(float(res_diff), 3),
                        })
    return pd.DataFrame(rows)


def compute_lag_correlation(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    aoi: str,
    source_key: str,
    metrics: dict[str, str],
) -> pd.DataFrame:
    """
    Test state dependence: Spearman rho between OLS residuals (current year)
    and prior-year response.  A significant positive correlation indicates
    that above-average response in year t-1 leads to above-average response
    in year t even after accounting for ERA5 forcing - a signature of
    multi-year hysteresis (e.g. albedo feedback, firn densification).
    """
    merged = _merge_era5_response(era5_stats, response_df, aoi, list(metrics.keys()))
    rows = []
    for pred_col in ERA5_PREDICTORS:
        if pred_col not in merged.columns:
            continue
        for resp_col in metrics:
            if resp_col not in merged.columns:
                continue
            sub = merged[[pred_col, resp_col]].dropna()
            if len(sub) < 5:
                continue

            x = sub[pred_col].values.astype(float)
            y = sub[resp_col].values.astype(float)
            coeffs = np.polyfit(x, y, 1)
            resid  = y - np.polyval(coeffs, x)

            # Look up prior-year response by calendar year (not by row position)
            # so that gaps in the record don't shift the wrong value.
            prior_years = [yr - 1 for yr in sub.index]
            lag_resp = pd.Series(
                [merged[resp_col].get(py, np.nan) for py in prior_years],
                index=sub.index,
            )
            valid = lag_resp.notna().values
            if valid.sum() < 4:
                continue

            rho, p = spearmanr(resid[valid], lag_resp.values[valid])
            rows.append({
                "response_type": source_key,
                "aoi":           aoi,
                "predictor":     pred_col,
                "response":      resp_col,
                "n":             int(valid.sum()),
                "rho":           round(float(rho), 4),
                "p_value":       round(float(p), 6),
                "sig":           _sig(float(p)),
            })
    return pd.DataFrame(rows)


# ============================================================
# 6) CONSOLE SUMMARIES
# ============================================================

def print_r_squared(r2_df: pd.DataFrame, aoi: str, source_key: str) -> None:
    SEP = "  " + "-" * 86
    print(f"\n  {aoi} [{source_key}]  -  R^2: ERA5 predictors vs melt response")
    print(
        f"  {'Predictor':<22} {'Response':<28} {'n':>3}  "
        f"{'r':>7}  {'R^2':>7}  {'p-value':>9}"
    )
    print(SEP)
    for _, row in r2_df.iterrows():
        print(
            f"  {row['predictor']:<22} {row['response']:<28} {int(row['n']):>3}  "
            f"{row['r']:>7.3f}  {row['r_squared']:>7.3f}  "
            f"{row['p_value']:>9.4f}  {row['sig']}"
        )


def print_pooled_r_squared(pooled_df: pd.DataFrame, source_key: str) -> None:
    SEP = "  " + "-" * 96
    print(f"\n  Pooled AOIs [{source_key}]  -  R^2 with and without AOI fixed effect")
    if pooled_df.empty:
        print("  (no results)")
        return
    print(
        f"  {'Model':<24} {'Predictor':<22} {'Response':<28} {'n':>3}  "
        f"{'R^2':>7}  {'adj R^2':>7}"
    )
    print(SEP)
    for _, row in pooled_df.iterrows():
        print(
            f"  {row['model_type']:<24} {row['predictor']:<22} "
            f"{row['response']:<28} {int(row['n']):>3}  "
            f"{row['r_squared']:>7.3f}  {row['adj_r_squared']:>7.3f}"
        )


def print_hysteresis(
    analog_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    aoi: str,
    source_key: str,
) -> None:
    SEP = "  " + "-" * 86

    print(
        f"\n  {aoi} [{source_key}]  -  Hysteresis: lagged state-dependence "
        f"(Spearman rho, OLS residuals vs prior-year response)"
    )
    if lag_df.empty:
        print("  (no results)")
    else:
        print(
            f"  {'Predictor':<22} {'Response':<28} {'n':>3}  "
            f"{'rho':>7}  {'p-value':>9}"
        )
        print(SEP)
        for _, row in lag_df.iterrows():
            print(
                f"  {row['predictor']:<22} {row['response']:<28} "
                f"{int(row['n']):>3}  {row['rho']:>7.3f}  "
                f"{row['p_value']:>9.4f}  {row['sig']}"
            )

    print(
        f"\n  {aoi} [{source_key}]  -  Hysteresis: analog year pairs "
        f"(|Delta met z| < {HYSTERESIS_MET_SD}, |Delta resid z| > {HYSTERESIS_RES_SD})"
    )
    if analog_df.empty:
        print("  (no analog pairs detected)")
    else:
        for _, row in analog_df.iterrows():
            print(
                f"  {row['predictor']} -> {row['response']}: "
                f"years {row['year_a']} vs {row['year_b']}  "
                f"met=({row['met_val_a']:.2f}, {row['met_val_b']:.2f})  "
                f"resp=({row['resp_val_a']:.3f}, {row['resp_val_b']:.3f})  "
                f"|Delta met z|={row['met_z_diff']:.2f}  "
                f"|Delta resid z|={row['resid_z_diff']:.2f}"
            )


# ============================================================
# 7) R^2 SCATTER FIGURE
# ============================================================

def make_r2_figure(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    aoi: str,
    source_key: str,
    source: dict,
    out_path: Path,
) -> None:
    """
    Grid scatter figure: rows = R2_FIG_ROWS (ERA5 predictors),
    cols = source["fig_cols"] (response metrics).
    Each panel shows the data points, OLS fit line, and R^2/p annotation.
    """
    merged = _merge_era5_response(
        era5_stats, response_df, aoi, list(source["metrics"].keys())
    )

    fig_cols = [c for c in source["fig_cols"] if c in merged.columns]
    if not fig_cols:
        print(
            f"  No figure columns available for {source_key} / {aoi} - skipping figure."
        )
        return

    n_rows = len(R2_FIG_ROWS)
    n_cols = len(fig_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH_IN, R2_FIG_HEIGHT_IN),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.42)

    x_labels     = {k: v for k, v in ERA5_PREDICTORS.items() if k in R2_FIG_ROWS}
    col_titles   = source["fig_col_titles"]

    for ri, pred_col in enumerate(R2_FIG_ROWS):
        for ci, resp_col in enumerate(fig_cols):
            ax = axes[ri][ci]

            if pred_col not in merged.columns or resp_col not in merged.columns:
                ax.set_visible(False)
                continue

            sub = merged[[pred_col, resp_col]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            x     = sub[pred_col].values.astype(float)
            y     = sub[resp_col].values.astype(float)
            years = sub.index.tolist()

            ax.scatter(
                x, y,
                s=18, color=R2_SCATTER_COLOUR,
                linewidths=0.3, edgecolors="white",
                zorder=3,
            )

            for xi, yi, yr in zip(x, y, years):
                ax.annotate(
                    str(yr),
                    xy=(xi, yi),
                    xytext=(2, 3),
                    textcoords="offset points",
                    fontsize=4.5,
                    color="#666666",
                )

            if len(sub) >= 3:
                coeffs = np.polyfit(x, y, 1)
                x_fit  = np.linspace(x.min(), x.max(), 80)
                ax.plot(
                    x_fit, np.polyval(coeffs, x_fit),
                    color=R2_FIT_COLOUR, linewidth=0.9, zorder=2,
                )
                r, p = pearsonr(x, y)
                ax.text(
                    0.97, 0.05,
                    f"R^2 = {r**2:.2f}\np = {p:.3f} {_sig(p)}",
                    transform=ax.transAxes,
                    fontsize=5,
                    ha="right", va="bottom",
                    color=R2_FIT_COLOUR,
                )

            ax.set_xlabel(x_labels.get(pred_col, pred_col), fontsize=5.5)
            if ci == 0:
                ax.set_ylabel(source["unit"], fontsize=5.5)
            if ri == 0:
                ax.set_title(col_titles.get(resp_col, resp_col), fontsize=6, pad=3)

            ax.tick_params(labelsize=5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(SPINE_WIDTH)
            ax.spines["bottom"].set_linewidth(SPINE_WIDTH)

    fig.suptitle(
        f"ERA5 vs {source_key.title()} - {AOI_NAMES.get(aoi, aoi)}",
        fontsize=BASE_FONT_SIZE,
        fontweight="bold",
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    if SHOW_FIGURES and mpl.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved -> {out_path}")


def make_pooled_r2_figure(
    era5_stats: pd.DataFrame,
    response_df: pd.DataFrame,
    source_key: str,
    source: dict,
    out_path: Path,
) -> None:
    """
    Grid scatter figure pooled across AOIs. Points are coloured by AOI and
    each panel shows the common-slope AOI fixed-effect fit as parallel lines.
    """
    pooled = _pool_era5_response(
        era5_stats, response_df, list(source["metrics"].keys())
    )
    if pooled.empty:
        print(f"  No pooled data available for {source_key} - skipping figure.")
        return

    fig_cols = [c for c in source["fig_cols"] if c in pooled.columns]
    if not fig_cols:
        print(f"  No pooled figure columns available for {source_key} - skipping figure.")
        return

    n_rows = len(R2_FIG_ROWS)
    n_cols = len(fig_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH_IN, R2_FIG_HEIGHT_IN),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.42)

    x_labels = {k: v for k, v in ERA5_PREDICTORS.items() if k in R2_FIG_ROWS}
    col_titles = source["fig_col_titles"]

    for ri, pred_col in enumerate(R2_FIG_ROWS):
        for ci, resp_col in enumerate(fig_cols):
            ax = axes[ri][ci]

            if pred_col not in pooled.columns or resp_col not in pooled.columns:
                ax.set_visible(False)
                continue

            sub = pooled[["year", "aoi", "aoi_ptm", pred_col, resp_col]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            for aoi in AOIS:
                aoi_sub = sub[sub["aoi"] == aoi]
                if aoi_sub.empty:
                    continue
                style = POOLED_AOI_STYLE.get(
                    aoi, {"colour": R2_SCATTER_COLOUR, "marker": "o"}
                )
                ax.scatter(
                    aoi_sub[pred_col].values.astype(float),
                    aoi_sub[resp_col].values.astype(float),
                    s=18,
                    color=style["colour"],
                    marker=style["marker"],
                    linewidths=0.3,
                    edgecolors="white",
                    label=aoi,
                    zorder=3,
                )
                for _, point in aoi_sub.iterrows():
                    ax.annotate(
                        str(int(point["year"])),
                        xy=(point[pred_col], point[resp_col]),
                        xytext=(2, 3),
                        textcoords="offset points",
                        fontsize=4.5,
                        color="#666666",
                    )

            diag = _fit_linear_model_full(sub, resp_col, ["aoi_ptm", pred_col])
            if not np.isnan(diag["r_squared"]):
                x_min = float(sub[pred_col].min())
                x_max = float(sub[pred_col].max())
                x_fit = np.linspace(x_min, x_max, 80)
                pred_coef = diag["coefficients"].get(pred_col, np.nan)
                aoi_coef = diag["coefficients"].get("aoi_ptm", 0.0)
                for aoi in AOIS:
                    style = POOLED_AOI_STYLE.get(
                        aoi, {"colour": R2_FIT_COLOUR, "marker": "o"}
                    )
                    aoi_offset = aoi_coef if aoi == "PTM" else 0.0
                    y_fit = diag["intercept"] + aoi_offset + pred_coef * x_fit
                    ax.plot(
                        x_fit, y_fit,
                        color=style["colour"],
                        linewidth=0.9,
                        alpha=0.9,
                        zorder=2,
                    )
                ax.text(
                    0.97, 0.05,
                    f"fixed-effect R^2 = {diag['r_squared']:.2f}\n"
                    f"adj R^2 = {diag['adj_r_squared']:.2f}",
                    transform=ax.transAxes,
                    fontsize=5,
                    ha="right", va="bottom",
                    color=R2_FIT_COLOUR,
                )

            ax.set_xlabel(x_labels.get(pred_col, pred_col), fontsize=5.5)
            if ci == 0:
                ax.set_ylabel(source["unit"], fontsize=5.5)
            if ri == 0:
                ax.set_title(col_titles.get(resp_col, resp_col), fontsize=6, pad=3)

            ax.tick_params(labelsize=5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(SPINE_WIDTH)
            ax.spines["bottom"].set_linewidth(SPINE_WIDTH)

    legend_handles = [
        Line2D(
            [], [], linestyle="none",
            marker=POOLED_AOI_STYLE[aoi]["marker"],
            markersize=4,
            markerfacecolor=POOLED_AOI_STYLE[aoi]["colour"],
            markeredgecolor="white",
            label=f"{aoi} points",
        )
        for aoi in AOIS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"ERA5 vs {source_key.title()} - pooled AOIs with AOI fixed effect",
        fontsize=BASE_FONT_SIZE,
        fontweight="bold",
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    if SHOW_FIGURES and mpl.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
    print(f"  Saved pooled -> {out_path}")


# ============================================================
# 8) CSV EXPORT
# ============================================================

def _save_r2_csvs(
    r2_df: pd.DataFrame,
    pooled_r2_df: pd.DataFrame,
    analog_df: pd.DataFrame,
    lag_df: pd.DataFrame,
) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, df in (
        ("era5_r2_summary.csv",              r2_df),
        ("era5_r2_pooled_summary.csv",       pooled_r2_df),
        ("era5_hysteresis_analog_pairs.csv", analog_df),
        ("era5_hysteresis_lag_corr.csv",     lag_df),
    ):
        path = CSV_OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        print(f"  Saved CSV -> {path}")


# ============================================================
# 9) MAIN
# ============================================================

def main() -> None:
    _apply_rcparams()

    print("Loading ERA5 JJA summary ...")
    era5_stats = load_era5_stats(ERA5_SUMMARY_CSV)
    print(f"  ERA5 rows: {len(era5_stats)}")

    all_r2     = []
    all_pooled = []
    all_analog = []
    all_lag    = []

    for source_key, source in RESPONSE_SOURCES.items():
        print(f"\n{'=' * 65}")
        print(f"  Response type: {source_key}")
        print(f"{'=' * 65}")

        response_df = load_response_source(source)
        if response_df is None:
            continue
        print(f"  Rows loaded: {len(response_df)}")
        metrics = source["metrics"]

        pooled_df = compute_pooled_r_squared(era5_stats, response_df, source_key, metrics)
        print_pooled_r_squared(pooled_df, source_key)
        all_pooled.append(pooled_df)

        pooled_fig = FIG_OUTPUT_DIR / f"era5_r2_scatter_pooled_{source_key}.png"
        make_pooled_r2_figure(
            era5_stats, response_df, source_key, source, pooled_fig
        )

        for aoi in AOIS:
            print(f"\n  --- AOI: {aoi}  ({AOI_NAMES.get(aoi, '')}) ---")

            r2_df = compute_r_squared(era5_stats, response_df, aoi, source_key, metrics)
            print_r_squared(r2_df, aoi, source_key)
            all_r2.append(r2_df)

            r2_fig = FIG_OUTPUT_DIR / f"era5_r2_scatter_{source_key}_{aoi}.png"
            make_r2_figure(era5_stats, response_df, aoi, source_key, source, r2_fig)

            analog_df = test_hysteresis(era5_stats, response_df, aoi, source_key, metrics)
            lag_df    = compute_lag_correlation(era5_stats, response_df, aoi, source_key, metrics)
            print_hysteresis(analog_df, lag_df, aoi, source_key)
            all_analog.append(analog_df)
            all_lag.append(lag_df)

    print("\nSaving R^2 and hysteresis CSVs ...")
    _save_r2_csvs(
        pd.concat(all_r2,     ignore_index=True) if all_r2     else pd.DataFrame(),
        pd.concat(all_pooled, ignore_index=True) if all_pooled else pd.DataFrame(),
        pd.concat(all_analog, ignore_index=True) if all_analog else pd.DataFrame(),
        pd.concat(all_lag,    ignore_index=True) if all_lag    else pd.DataFrame(),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
