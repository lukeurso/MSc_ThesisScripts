# -*- coding: utf-8 -*-
"""
rate_of_change_per_aoi_correction.py

The original rate_of_change_analysis.py uses a *pooled* OST+PTM regression
(adj-R² = 0.53 for T_max + SW_mean) as the climate baseline.  Per-AOI
regressions are far stronger:

    PTM T_max → combined_max area:  R² = 0.85 (***)
    OST T_max → combined_max area:  R² = 0.25 (ns)

Pooling washes out the inter-glacier asymmetry.  This script reproduces the
recovery-rate analysis with per-AOI met-residual regression instead.

Method (matches original cycle definitions: 2017→2019 and 2020→2023 at both AOIs):
  1) For each AOI, fit OLS:    combined_mean_km2 ~ predictor(s)
     (predictor variants: T_mean,  T_max,  T_max + SW_mean)
  2) Compute residuals = observed - fitted.
  3) Within each cycle, fit OLS: residual ~ year and report slope (km²/yr),
     R², and p with bootstrap CI.

This is the same workflow as the pooled version, just with per-AOI fits.

Inputs:
  area_jja_summary.csv          (combined_mean_km2)
  era5_jja_mean_max_summary.csv (predictors)

Outputs:
  csvs/rate_of_change_per_aoi_cycles.csv     (per-AOI cycle slopes)
  csvs/rate_of_change_per_aoi_models.csv     (per-AOI met models)
  figures/rate_of_change_per_aoi_slopes.png  (replaces pooled figure)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ============================================================
# 1) Paths
# ============================================================
CSV_DIR = Path(r"C:\msc_thesis_local\working_material\csv_files\csv_outputs\time_series_tests")
OUT_DIR_CSV = Path(r"C:\msc_thesis_local\working_material\new_work_default\results_gap_fills\csvs")
OUT_DIR_FIG = Path(r"C:\msc_thesis_local\working_material\new_work_default\results_gap_fills\figures")
OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

area = pd.read_csv(CSV_DIR / "area_jja_summary.csv")
era5 = pd.read_csv(CSV_DIR / "era5_jja_mean_max_summary.csv")

area_obs = area[area["label"] == "Observed"].copy()
RESPONSE = "combined_mean_km2"

# Match year ranges
df = (area_obs[["aoi", "year", RESPONSE]]
      .merge(era5, on=["aoi", "year"], how="inner"))


# ============================================================
# 2) Cycles (matching original)
# ============================================================
CYCLES = [
    ("OST", 1, 2017, 2019),
    ("OST", 2, 2020, 2023),
    ("PTM", 1, 2017, 2019),
    ("PTM", 2, 2020, 2023),
]

PRED_VARIANTS = [
    ("Tmean",          ["temp_2m_C_mean"]),
    ("Tmax",           ["temp_2m_C_max"]),
    ("Tmax+SWmean",    ["temp_2m_C_max", "srad_Wm2_mean"]),
]


def fit_ols(X, y):
    """Multivariate OLS via numpy. Returns (coefficients, residuals, R²)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    resid = y - yhat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, resid, r2


def cycle_slope(years, values, n_boot=2000, ci_level=0.95):
    """OLS slope of values ~ years, with bootstrap CI."""
    res = linregress(years, values)
    slope = res.slope
    pval = res.pvalue
    rsq = res.rvalue ** 2
    rng = np.random.default_rng(seed=42)
    boots = []
    n = len(years)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(years[idx])) < 2:
            continue
        b = linregress(years[idx], values[idx]).slope
        boots.append(b)
    boots = np.asarray(boots)
    ci_lo = np.quantile(boots, (1 - ci_level) / 2)
    ci_hi = np.quantile(boots, 1 - (1 - ci_level) / 2)
    return slope, pval, rsq, ci_lo, ci_hi


# ============================================================
# 3) Fit per-AOI models, compute residuals, then cycle slopes
# ============================================================
model_rows = []
cycle_rows = []

for aoi in ["OST", "PTM"]:
    sub = df[df["aoi"] == aoi].sort_values("year").reset_index(drop=True)
    y_full = sub[RESPONSE].values
    yr_full = sub["year"].values

    # Fit each predictor variant
    for var_label, predictors in PRED_VARIANTS:
        X = sub[predictors].values
        beta, resid, r2 = fit_ols(X, y_full)
        # Adjusted R²
        n = len(y_full)
        k = len(predictors)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan
        model_rows.append({
            "aoi": aoi,
            "model": var_label,
            "predictors": "+".join(predictors),
            "n": n,
            "r_squared": r2,
            "adj_r_squared": adj_r2,
            "residual_sd_km2": resid.std(ddof=1),
        })

        # For each cycle, slope of residual ~ year
        for c_aoi, c_id, y_start, y_end in CYCLES:
            if c_aoi != aoi:
                continue
            mask = (yr_full >= y_start) & (yr_full <= y_end)
            yrs_c = yr_full[mask]
            res_c = resid[mask]
            if len(yrs_c) < 2:
                continue
            slope, p, rsq, ci_lo, ci_hi = cycle_slope(yrs_c, res_c)
            cycle_rows.append({
                "aoi": aoi,
                "cycle_id": c_id,
                "year_start": y_start,
                "year_end": y_end,
                "n_years": len(yrs_c),
                "predictor_set": var_label,
                "slope_km2_per_yr_residual": slope,
                "r_squared": rsq,
                "p_value": p,
                "boot_ci_lo": ci_lo,
                "boot_ci_hi": ci_hi,
            })

    # Also raw (no-correction) cycle slopes for reference
    for c_aoi, c_id, y_start, y_end in CYCLES:
        if c_aoi != aoi:
            continue
        mask = (yr_full >= y_start) & (yr_full <= y_end)
        yrs_c = yr_full[mask]
        vals_c = y_full[mask]
        slope, p, rsq, ci_lo, ci_hi = cycle_slope(yrs_c, vals_c)
        cycle_rows.append({
            "aoi": aoi,
            "cycle_id": c_id,
            "year_start": y_start,
            "year_end": y_end,
            "n_years": len(yrs_c),
            "predictor_set": "Raw",
            "slope_km2_per_yr_residual": slope,
            "r_squared": rsq,
            "p_value": p,
            "boot_ci_lo": ci_lo,
            "boot_ci_hi": ci_hi,
        })

models = pd.DataFrame(model_rows)
cycles = pd.DataFrame(cycle_rows)
models.to_csv(OUT_DIR_CSV / "rate_of_change_per_aoi_models.csv", index=False)
cycles.to_csv(OUT_DIR_CSV / "rate_of_change_per_aoi_cycles.csv", index=False)
print(f"Wrote rate_of_change_per_aoi_models.csv ({len(models)} rows)")
print(f"Wrote rate_of_change_per_aoi_cycles.csv ({len(cycles)} rows)")

print("\n  -- Per-AOI met models --")
print(models.to_string(index=False, float_format="%.3f"))

print("\n  -- Per-AOI cycle slopes (raw vs climate-corrected) --")
print(cycles.pivot_table(index=["aoi", "cycle_id", "year_start", "year_end"],
                         columns="predictor_set",
                         values="slope_km2_per_yr_residual",
                         aggfunc="first").round(2).to_string())


# ============================================================
# 4) Compare to pooled (existing) values
# ============================================================
pooled = pd.read_csv(CSV_DIR / "era5_rate_of_change_pooled_met_models.csv")
print("\n  -- Pooled (existing) for reference --")
print(pooled.to_string(index=False, float_format="%.3f"))


# ============================================================
# 5) Plot: per-AOI cycle slopes under each correction model
# ============================================================
plt.style.use("default")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
})

models_order = ["Raw", "Tmean", "Tmax", "Tmax+SWmean"]
hatches = {"Raw": "", "Tmean": "//", "Tmax": "xx", "Tmax+SWmean": ".."}
colours = {"OST": "#3a86b8", "PTM": "#c1272d"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)

for ax_i, aoi in enumerate(["OST", "PTM"]):
    ax = axes[ax_i]
    sub = cycles[cycles["aoi"] == aoi].copy()
    cycles_in_aoi = sorted(sub["cycle_id"].unique())
    n_cycles = len(cycles_in_aoi)
    n_models = len(models_order)
    bar_w = 0.18
    for j, cid in enumerate(cycles_in_aoi):
        for k, model in enumerate(models_order):
            row = sub[(sub["cycle_id"] == cid) & (sub["predictor_set"] == model)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            x_pos = j + (k - (n_models - 1) / 2) * bar_w
            slope = row["slope_km2_per_yr_residual"]
            ci_lo = row["boot_ci_lo"]
            ci_hi = row["boot_ci_hi"]
            err_lo = max(slope - ci_lo, 0)
            err_hi = max(ci_hi - slope, 0)
            ax.bar(x_pos, slope, width=bar_w, color=colours[aoi],
                   edgecolor="black", linewidth=0.5, hatch=hatches[model],
                   label=model if j == 0 else None, alpha=0.85)
            ax.errorbar(x_pos, slope, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_title(("(a)  " if aoi == "OST" else "(b)  ") + aoi)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Slope (km² yr⁻¹)")
    ax.set_xticks(range(n_cycles))
    ax.set_xticklabels([
        f"Cycle {cid}\n{int(sub[sub['cycle_id']==cid]['year_start'].iloc[0])}–"
        f"{int(sub[sub['cycle_id']==cid]['year_end'].iloc[0])}"
        for cid in cycles_in_aoi
    ])
    if ax_i == 0:
        ax.legend(loc="upper left", frameon=False, fontsize=8, title="Correction")

fig.suptitle(
    "Melt-area recovery slope per cycle — per-AOI climate correction\n"
    "Compared to the pooled-AOI correction in the original analysis",
    y=1.02, fontsize=12, weight="bold",
)
fig.tight_layout()
out_png = OUT_DIR_FIG / "rate_of_change_per_aoi_slopes.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")
print("Done.")
