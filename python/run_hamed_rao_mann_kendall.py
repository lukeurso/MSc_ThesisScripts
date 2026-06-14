# -*- coding: utf-8 -*-
"""
run_hamed_rao_mann_kendall.py

Re-runs Mann-Kendall on every observed series (area, elevation, volume × OST/PTM
× mean/max combinations) using the Hamed-Rao autocorrelation correction
(`pymannkendall.hamed_rao_modification_test`).

This is motivated by the documented multi-year memory at PTM (lag-1 ρ ≈ 0.68
for area lake_max), which can inflate standard MK significance.

Outputs:
  csvs/hamed_rao_mk_results.csv               (all series, both methods side-by-side)
  csvs/hamed_rao_mk_significance_changes.csv  (only series whose significance changed)
  figures/hamed_rao_p_value_comparison.png    (visual side-by-side of p-values)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymannkendall as mk


# ============================================================
# 1) Paths
# ============================================================
CSV_DIR = Path(r"C:\msc_thesis_local\working_material\csv_files\csv_outputs\time_series_tests")
OUT_DIR_CSV = Path(r"C:\msc_thesis_local\working_material\new_work_default\results_gap_fills\csvs")
OUT_DIR_FIG = Path(r"C:\msc_thesis_local\working_material\new_work_default\results_gap_fills\figures")
OUT_DIR_CSV.mkdir(parents=True, exist_ok=True)
OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)

area = pd.read_csv(CSV_DIR / "area_jja_summary.csv")
elev = pd.read_csv(CSV_DIR / "elev_jja_summary.csv")
vol  = pd.read_csv(CSV_DIR / "volume_jja_summary.csv")


# ============================================================
# 2) Series specification — observed only, both AOIs, all stats
# ============================================================
SPECS = []

# Area: per AOI, observed, all 6 series
for aoi in ["OST", "PTM"]:
    sub = area[(area["aoi"] == aoi) & (area["label"] == "Observed")].sort_values("year")
    for series, col in [
        ("slush_mean", "slush_mean_km2"),
        ("slush_max", "slush_max_km2"),
        ("lake_mean", "lake_mean_km2"),
        ("lake_max", "lake_max_km2"),
        ("combined_mean", "combined_mean_km2"),
        ("combined_max", "combined_max_km2"),
    ]:
        SPECS.append(("area", aoi, series, sub["year"].values, sub[col].values))

# Elevation: per AOI (no label column)
for aoi in ["OST", "PTM"]:
    sub = elev[elev["aoi"] == aoi].sort_values("year")
    for series, col in [
        ("slush_mean", "slush_mean_m"),
        ("slush_max", "slush_max_m"),
        ("lake_mean", "lake_mean_m"),
        ("lake_max", "lake_max_m"),
        ("combined_mean", "combined_mean_m"),
        ("combined_max", "combined_max_m"),
    ]:
        SPECS.append(("elev", aoi, series, sub["year"].values, sub[col].values))

# Volume: per AOI, observed, lake mean and max
for aoi in ["OST", "PTM"]:
    sub = vol[(vol["aoi"] == aoi) & (vol["label"] == "Observed")].sort_values("year")
    for series, col in [
        ("lake_mean", "lake_mean_Mm3"),
        ("lake_max", "lake_max_Mm3"),
    ]:
        SPECS.append(("volume", aoi, series, sub["year"].values, sub[col].values))


# ============================================================
# 3) Run both standard and Hamed-Rao MK on each series
# ============================================================
def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "ns"


rows = []
for variable, aoi, series, years, y in SPECS:
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        # pymannkendall handles NaN by raising; skip
        print(f"  ! Skipping {variable} {aoi} {series} (NaN present)")
        continue
    res_std = mk.original_test(y)
    res_hr  = mk.hamed_rao_modification_test(y)
    rows.append({
        "variable": variable,
        "aoi": aoi,
        "series": series,
        "n": len(y),
        # Standard MK
        "std_tau": res_std.Tau,
        "std_S": res_std.s,
        "std_z": res_std.z,
        "std_var_s": res_std.var_s,
        "std_p": res_std.p,
        "std_sig": sig_label(res_std.p),
        "std_trend": res_std.trend,
        # Hamed-Rao corrected
        "hr_tau": res_hr.Tau,
        "hr_S": res_hr.s,
        "hr_z": res_hr.z,
        "hr_var_s": res_hr.var_s,
        "hr_p": res_hr.p,
        "hr_sig": sig_label(res_hr.p),
        "hr_trend": res_hr.trend,
        # Variance ratio: hr_var_s / std_var_s tells you the autocorrelation correction factor
        "var_inflation": res_hr.var_s / res_std.var_s if res_std.var_s > 0 else np.nan,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR_CSV / "hamed_rao_mk_results.csv", index=False)
print(f"Wrote {OUT_DIR_CSV / 'hamed_rao_mk_results.csv'} ({len(df)} rows)")

# Print summary
print("\n  -- Hamed-Rao vs Standard MK -- ")
print(df[["variable", "aoi", "series", "std_tau", "std_p", "std_sig",
          "hr_p", "hr_sig", "var_inflation"]]
        .to_string(index=False, float_format="%.3f"))

# Identify significance changes
diff = df[df["std_sig"] != df["hr_sig"]]
diff.to_csv(OUT_DIR_CSV / "hamed_rao_mk_significance_changes.csv", index=False)
print(f"\nSignificance changes: {len(diff)}")
if len(diff) > 0:
    print(diff[["variable", "aoi", "series", "std_p", "std_sig", "hr_p", "hr_sig", "var_inflation"]]
          .to_string(index=False, float_format="%.3f"))


# ============================================================
# 4) Visualisation: side-by-side p-value comparison
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

# One row per series — 28 series total
df_sorted = df.sort_values(["variable", "aoi", "series"]).reset_index(drop=True)
labels = df_sorted.apply(lambda r: f"{r['variable']:<6s}  {r['aoi']}  {r['series']}", axis=1).tolist()

fig, ax = plt.subplots(figsize=(11, 9))
y_positions = np.arange(len(df_sorted))
ax.scatter(df_sorted["std_p"], y_positions, s=58, c="#3a86b8", label="Standard MK",
           edgecolor="white", linewidth=0.7, zorder=3)
ax.scatter(df_sorted["hr_p"], y_positions, s=58, c="#c1272d", marker="D", label="Hamed-Rao corrected",
           edgecolor="white", linewidth=0.7, zorder=3)
# Connecting lines
for i in range(len(df_sorted)):
    ax.plot([df_sorted["std_p"].iloc[i], df_sorted["hr_p"].iloc[i]],
            [i, i], color="#cccccc", lw=0.6, zorder=1)
ax.axvline(0.05, color="#666666", ls="--", lw=0.8, alpha=0.7)
ax.axvline(0.10, color="#aaaaaa", ls=":",  lw=0.8, alpha=0.7)
ax.text(0.05, len(df_sorted) - 0.3, "p = 0.05", color="#666666", fontsize=8,
        ha="left", va="top")
ax.text(0.10, len(df_sorted) - 0.3, "p = 0.10", color="#aaaaaa", fontsize=8,
        ha="left", va="top")

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8.5, family="monospace")
ax.set_xlabel("p-value")
ax.set_title("Mann-Kendall p-values: Standard vs Hamed-Rao autocorrelation-corrected\n"
             "Observed JJA series, all AOIs and response variables")
ax.set_xlim(-0.02, 1.02)
ax.invert_yaxis()
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
out_png = OUT_DIR_FIG / "hamed_rao_p_value_comparison.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out_png}")

# Variance-inflation distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["var_inflation"].dropna(), bins=20, color="#3a86b8", edgecolor="white")
ax.axvline(1.0, color="#666666", ls="--", lw=0.9)
ax.set_xlabel("Variance inflation factor  (Hamed-Rao Var(S) / standard Var(S))")
ax.set_ylabel("Number of series")
ax.set_title("Distribution of Hamed-Rao variance-inflation factor across all series\n"
             "(>1 means autocorrelation reduces effective sample size; "
             "<1 means anti-correlation increases it)")
fig.tight_layout()
out_png = OUT_DIR_FIG / "hamed_rao_var_inflation_distribution.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_png}")

print("\nDone.")
