# compute_anomalies_all.py
#
# Replicates the baseline-selection + anomaly computation from
# plot_jja_area_anomalies.py and extends it to:
#   - all six area metrics (slush/lake/combined  x  mean/max), Observed + Scaled
#   - all six elevation metrics, Observed only (no scaled elevation product exists)
#   - both volume metrics (lake_mean_Mm3, lake_max_Mm3), Observed + Scaled
#
# Baseline windows are selected ONCE per AOI per processing on the area
# combined_max_km2 series (matching plot_jja_area_anomalies.py); the same
# window is then applied to every metric so the resulting anomalies are
# mutually comparable across families.
#
# Inputs:
#   anomaly_csvs/area_jja_summary.csv     (copies of the time_series_tests CSVs)
#   anomaly_csvs/elev_jja_summary.csv
#   anomaly_csvs/volume_jja_summary.csv
#
# To re-run after a refresh of the upstream JJA summaries, copy the latest
# versions of those three files from
#   Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests\
# into ROOT/anomaly_csvs/ and run this script.
#
# Outputs (under ROOT/csvs/):
#   anomalies_area_all_metrics.csv
#   anomalies_elev_all_metrics.csv
#   anomalies_volume_all_metrics.csv
#   anomalies_combined_all.csv     (concatenation of the three above)
#   baselines_used.csv             (per-family baseline means table)
#   obs_vs_scaled_diffs.csv        (paired Observed vs Scaled differences)
#   extreme_year_ranks_top3.csv    (top 3 |anomaly_pct| per metric/processing/AOI)
#   area_jja_anomalies_slush_lake_decomp.csv (filter of area to slush+lake only)

from pathlib import Path
import numpy as np
import pandas as pd

# ROOT auto-detects whether the script is being run on Luke's Windows
# workstation or inside the cowork Linux sandbox.
WIN_ROOT = Path(r"Q:\ThesisData\data\tasks\anomoly_task_folder")
LIN_ROOT = Path("/sessions/magical-compassionate-thompson/mnt/anomoly_task_folder")
ROOT = WIN_ROOT if WIN_ROOT.exists() else LIN_ROOT

IN = ROOT / "anomaly_csvs"
OUT = ROOT / "csvs"
OUT.mkdir(parents=True, exist_ok=True)

AOIS = ["OST", "PTM"]

# baseline algorithm parameters - identical to plot_jja_area_anomalies.py
BASELINE_WINDOW_YEARS = 3
MIN_N_BLOCKS = 3
MIN_COVERAGE_MEAN = 0.20
HIGH_QUANTILE = 0.75
CV_TIE_TOLERANCE = 0.02
SELECTION_METRIC = "combined_max_km2"


def is_contiguous(years):
    v = [int(y) for y in years]
    return v == list(range(v[0], v[0] + len(v)))


def candidate_windows(df_aoi, metric):
    g = df_aoi.sort_values("year").reset_index(drop=True)
    threshold = float(g[metric].quantile(HIGH_QUANTILE))
    rows = []
    for i in range(0, len(g) - BASELINE_WINDOW_YEARS + 1):
        win = g.iloc[i:i + BASELINE_WINDOW_YEARS]
        years = [int(y) for y in win["year"]]
        vals = win[metric]
        contiguous = is_contiguous(win["year"])
        enough_n = bool((win["n"] >= MIN_N_BLOCKS).all())
        enough_cov = bool((win["coverage_mean"] >= MIN_COVERAGE_MEAN).all())
        has_vals = bool(vals.notna().all())
        has_high = bool((vals >= threshold).any()) if has_vals else True
        eligible = contiguous and enough_n and enough_cov and has_vals and not has_high
        m = float(vals.mean()) if has_vals else np.nan
        s = float(vals.std(ddof=0)) if has_vals else np.nan
        cv = (s / m) if has_vals and m != 0 else np.nan
        rows.append(dict(window_start_year=years[0], window_end_year=years[-1],
                          baseline_years="-".join(str(y) for y in years),
                          mean=m, std=s, cv=cv,
                          coverage_mean=float(win["coverage_mean"].mean()),
                          min_n=int(win["n"].min()),
                          eligible=eligible))
    return pd.DataFrame(rows)


def select_baseline_window(df_aoi, metric):
    cand = candidate_windows(df_aoi, metric)
    elig = cand[cand["eligible"]].copy()
    if elig.empty:
        raise ValueError("No eligible baseline window")
    cv_min = float(elig["cv"].min())
    near = elig[elig["cv"] <= cv_min + CV_TIE_TOLERANCE].copy()
    near = near.sort_values(["mean", "cv", "coverage_mean", "window_start_year"],
                             ascending=[True, True, False, True])
    return near.iloc[0]["baseline_years"]


def compute_metric_anomalies(df, baseline_years_per_aoi, metric, label, family):
    out = []
    for aoi in AOIS:
        sub = df[df["aoi"] == aoi].copy()
        years = [int(y) for y in baseline_years_per_aoi[aoi].split("-")]
        base = sub[sub["year"].isin(years)]
        if base.empty or base[metric].isna().any():
            continue
        base_mean = float(base[metric].mean())
        for _, row in sub.iterrows():
            v = float(row[metric]) if not pd.isna(row[metric]) else np.nan
            anom = v - base_mean if not np.isnan(v) else np.nan
            anom_pct = 100.0 * (v / base_mean - 1.0) if (base_mean != 0 and not np.isnan(v)) else np.nan
            fold = (v / base_mean) if (base_mean != 0 and not np.isnan(v)) else np.nan
            out.append(dict(
                aoi=aoi, label=label, family=family, metric=metric,
                year=int(row["year"]), value=v,
                baseline_years=baseline_years_per_aoi[aoi],
                baseline_value=base_mean,
                anomaly=anom, anomaly_pct=anom_pct, fold_change=fold,
                n=int(row["n"]), coverage_mean=float(row["coverage_mean"]),
                is_baseline_year=int(row["year"]) in years))
    return out


# ----- 1) load -----
area = pd.read_csv(IN / "area_jja_summary.csv")
elev = pd.read_csv(IN / "elev_jja_summary.csv")
vol = pd.read_csv(IN / "volume_jja_summary.csv")


# ----- 2) baseline windows (combined_max_km2 algorithm-selected per AOI per processing) -----
baselines = {}
for label in ["Observed", "Scaled"]:
    sub = area[area["label"] == label]
    baselines[label] = {}
    for aoi in AOIS:
        baselines[label][aoi] = select_baseline_window(sub[sub["aoi"] == aoi], SELECTION_METRIC)
elev_baseline = baselines["Observed"]   # elev has no Scaled product

print("Baseline windows selected (combined_max_km2 low-CV, no high year):")
for label, bd in baselines.items():
    for aoi, by in bd.items():
        print(f"  {label}  {aoi}: {by}")
print(f"  elev: uses {elev_baseline} (no Scaled elev product)")


# ----- 3) baseline reference table (per family) -----
base_rows = []
for label, bdict in baselines.items():
    for aoi, by in bdict.items():
        years = [int(y) for y in by.split("-")]
        b = area[(area["label"] == label) & (area["aoi"] == aoi) & (area["year"].isin(years))]
        base_rows.append(dict(
            family="area", label=label, aoi=aoi, baseline_years=by,
            combined_max_baseline=float(b["combined_max_km2"].mean()),
            combined_mean_baseline=float(b["combined_mean_km2"].mean()),
            slush_max_baseline=float(b["slush_max_km2"].mean()),
            slush_mean_baseline=float(b["slush_mean_km2"].mean()),
            lake_max_baseline=float(b["lake_max_km2"].mean()),
            lake_mean_baseline=float(b["lake_mean_km2"].mean())))
for aoi, by in elev_baseline.items():
    years = [int(y) for y in by.split("-")]
    e = elev[(elev["aoi"] == aoi) & (elev["year"].isin(years))]
    base_rows.append(dict(
        family="elev", label="Observed", aoi=aoi, baseline_years=by,
        combined_max_baseline=float(e["combined_max_m"].mean()),
        combined_mean_baseline=float(e["combined_mean_m"].mean()),
        slush_max_baseline=float(e["slush_max_m"].mean()),
        slush_mean_baseline=float(e["slush_mean_m"].mean()),
        lake_max_baseline=float(e["lake_max_m"].mean()),
        lake_mean_baseline=float(e["lake_mean_m"].mean())))
for label, bdict in baselines.items():
    for aoi, by in bdict.items():
        years = [int(y) for y in by.split("-")]
        v = vol[(vol["label"] == label) & (vol["aoi"] == aoi) & (vol["year"].isin(years))]
        base_rows.append(dict(
            family="volume", label=label, aoi=aoi, baseline_years=by,
            combined_max_baseline=np.nan, combined_mean_baseline=np.nan,
            slush_max_baseline=np.nan, slush_mean_baseline=np.nan,
            lake_max_baseline=float(v["lake_max_Mm3"].mean()),
            lake_mean_baseline=float(v["lake_mean_Mm3"].mean())))
pd.DataFrame(base_rows).to_csv(OUT / "baselines_used.csv", index=False)


# ----- 4) anomalies -----
area_metrics = ["combined_max_km2", "combined_mean_km2",
                "slush_max_km2", "slush_mean_km2",
                "lake_max_km2", "lake_mean_km2"]
elev_metrics = ["combined_max_m", "combined_mean_m",
                "slush_max_m", "slush_mean_m",
                "lake_max_m", "lake_mean_m"]
vol_metrics = ["lake_max_Mm3", "lake_mean_Mm3"]

area_rows = []
for label in ["Observed", "Scaled"]:
    sub = area[area["label"] == label]
    for m in area_metrics:
        area_rows.extend(compute_metric_anomalies(sub, baselines[label], m, label, "area"))
pd.DataFrame(area_rows).to_csv(OUT / "anomalies_area_all_metrics.csv", index=False)

elev_rows = []
for m in elev_metrics:
    elev_rows.extend(compute_metric_anomalies(elev, elev_baseline, m, "Observed", "elev"))
pd.DataFrame(elev_rows).to_csv(OUT / "anomalies_elev_all_metrics.csv", index=False)

vol_rows = []
for label in ["Observed", "Scaled"]:
    sub = vol[vol["label"] == label]
    for m in vol_metrics:
        vol_rows.extend(compute_metric_anomalies(sub, baselines[label], m, label, "volume"))
pd.DataFrame(vol_rows).to_csv(OUT / "anomalies_volume_all_metrics.csv", index=False)

all_anom = pd.concat([pd.DataFrame(area_rows), pd.DataFrame(elev_rows),
                       pd.DataFrame(vol_rows)], ignore_index=True)
all_anom.to_csv(OUT / "anomalies_combined_all.csv", index=False)


# ----- 5) Observed vs Scaled diffs (area + volume only; elev has no Scaled) -----
diffs = []
for AOI in AOIS:
    pairs = [("area", "combined_max_km2"), ("area", "combined_mean_km2"),
             ("area", "slush_max_km2"), ("area", "slush_mean_km2"),
             ("area", "lake_max_km2"), ("area", "lake_mean_km2"),
             ("volume", "lake_max_Mm3"), ("volume", "lake_mean_Mm3")]
    for family, metric in pairs:
        obs = all_anom[(all_anom["family"] == family) & (all_anom["metric"] == metric) &
                        (all_anom["aoi"] == AOI) & (all_anom["label"] == "Observed")][["year", "anomaly_pct"]].rename(columns={"anomaly_pct": "obs_pct"})
        scl = all_anom[(all_anom["family"] == family) & (all_anom["metric"] == metric) &
                        (all_anom["aoi"] == AOI) & (all_anom["label"] == "Scaled")][["year", "anomaly_pct"]].rename(columns={"anomaly_pct": "scl_pct"})
        m = pd.merge(obs, scl, on="year")
        m["diff_pct_pts"] = m["obs_pct"] - m["scl_pct"]
        m["family"] = family
        m["metric"] = metric
        m["aoi"] = AOI
        diffs.append(m)
pd.concat(diffs, ignore_index=True).to_csv(OUT / "obs_vs_scaled_diffs.csv", index=False)


# ----- 6) extreme-year top 3 ranks -----
ranks = []
for (family, label, metric, AOI), group in all_anom.groupby(["family", "label", "metric", "aoi"]):
    nb = group[~group["is_baseline_year"]].copy()
    if nb.empty:
        continue
    nb = nb.assign(abs_pct=nb["anomaly_pct"].abs())
    top = nb.nlargest(3, "abs_pct")
    for _, r in top.iterrows():
        ranks.append(dict(family=family, label=label, metric=metric, aoi=AOI,
                          year=int(r["year"]), value=r["value"],
                          anomaly_pct=r["anomaly_pct"], fold=r["fold_change"]))
pd.DataFrame(ranks).to_csv(OUT / "extreme_year_ranks_top3.csv", index=False)


# ----- 7) slush + lake decomposition CSV (subset of area_all_metrics) -----
sub = pd.DataFrame(area_rows)
new_metrics = ["slush_max_km2", "slush_mean_km2", "lake_max_km2", "lake_mean_km2"]
sub[sub["metric"].isin(new_metrics)].to_csv(OUT / "area_jja_anomalies_slush_lake_decomp.csv", index=False)


# ----- 8) verification against user-supplied combined_max anomaly CSVs -----
def verify(label, aoi, year, expected_pct):
    row = all_anom[(all_anom["family"] == "area") & (all_anom["metric"] == "combined_max_km2") &
                    (all_anom["label"] == label) & (all_anom["aoi"] == aoi) & (all_anom["year"] == year)]
    if row.empty:
        print(f"  [WARN] no row for {label} {aoi} {year}")
        return
    actual = float(row["anomaly_pct"].iloc[0])
    ok = abs(actual - expected_pct) < 0.01
    print(f"  {label} {aoi} {year} combined_max: expected {expected_pct:+.4f}%, got {actual:+.4f}%  {'OK' if ok else 'MISMATCH'}")

print("\n=== Verification against user-supplied area_jja_anomalies_*.csv ===")
verify("Observed", "OST", 2014, -28.5427)
verify("Observed", "OST", 2023, +384.7717)
verify("Observed", "PTM", 2014, +152.7160)
verify("Observed", "PTM", 2023, +1227.0525)
verify("Scaled",   "OST", 2024, -64.5205)
verify("Scaled",   "PTM", 2023, +907.1042)

print(f"\nDone. Wrote outputs to {OUT}")
