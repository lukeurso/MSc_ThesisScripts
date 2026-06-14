# -*- coding: utf-8 -*-
"""
plot_jja_area_anomalies.py

Selects an objective annual JJA baseline for each glacier and computes
annual melt-area anomalies relative to that baseline.

Input:
  area_jja_summary.csv from plot_mean_max_area.py

Outputs:
  area_jja_baseline_candidates.csv
  area_jja_baseline_summary.csv
  area_jja_anomalies.csv
  jja_area_anomaly_pct_{AOI}.png
"""

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


# ============================================================
# 1) FILE PATHS
# ============================================================

SUMMARY_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
    r"\area_jja_summary.csv"
)

CSV_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\time_series_tests"
)

FIG_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\figures\plot_jja_area_anomalies"
)


# ============================================================
# 2) ANALYSIS SETTINGS
# ============================================================

AOIS = ["OST", "PTM"]

AOI_NAMES = {
    "OST": "C.H. Ostenfeld Glacier",
    "PTM": "Petermann Glacier",
}

DATA_LABEL = "scaled"

BASELINE_WINDOW_YEARS = 3
MIN_N_BLOCKS          = 3
MIN_COVERAGE_MEAN     = 0.20
HIGH_QUANTILE         = 0.75
CV_TIE_TOLERANCE      = 0.02

BASELINE_SELECTION_METRIC = "combined_max_km2"

ANOMALY_METRICS = [
    ("combined_max_km2",  "Combined maximum"),
    ("combined_mean_km2", "Combined mean"),
]


# ============================================================
# 3) FIGURE / STYLE PARAMETERS
# ============================================================

FIG_WIDTH_IN  = 7.087   # 180 mm
FIG_HEIGHT_IN = 5.4

DPI_SCREEN = 150
DPI_SAVE   = 300

HSPACE = 0.38

FONT_FAMILY      = "sans-serif"
BASE_FONT_SIZE   = 7
TICK_FONT_SIZE   = 6
LABEL_FONT_SIZE  = 7
LEGEND_FONT_SIZE = 6
PANEL_LABEL_SIZE = 8

SPINE_WIDTH         = 0.6
TICK_WIDTH          = 0.6
TICK_LENGTH         = 3.0
YEAR_LABEL_ROTATION = 45

PANEL_LABEL_X = -0.11
PANEL_LABEL_Y =  1.03

COLOUR_POSITIVE = "#B2182B"
COLOUR_NEGATIVE = "#2166AC"
COLOUR_BASELINE = "#BDBDBD"
COLOUR_ZERO     = "#333333"

BAR_ALPHA           = 0.85
BASELINE_SPAN_ALPHA = 0.18
GRID_LINESTYLE      = ":"
GRID_ALPHA          = 0.45
GRID_LINEWIDTH      = 0.5

LABEL_ANOMALY_THRESHOLD_PCT = 50.0


# ============================================================
# 4) DATA LOADING
# ============================================================

def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found:\n  {path}")

    df = pd.read_csv(path)
    required = {
        "aoi", "label", "year", "n", "coverage_mean",
        "combined_max_km2", "combined_mean_km2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input summary CSV is missing required columns: "
            + ", ".join(missing)
        )

    df = df[(df["label"] == DATA_LABEL) & (df["aoi"].isin(AOIS))].copy()
    if df.empty:
        raise ValueError(
            f"No rows found for label={DATA_LABEL!r} and AOIs={AOIS}."
        )

    numeric_cols = [
        "year", "n", "coverage_mean",
        "combined_max_km2", "combined_mean_km2",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = df["year"].astype("Int64")
    return df.sort_values(["aoi", "year"]).reset_index(drop=True)


# ============================================================
# 5) BASELINE SELECTION
# ============================================================

def _is_contiguous(years: pd.Series) -> bool:
    vals = [int(y) for y in years]
    return vals == list(range(vals[0], vals[0] + len(vals)))


def _candidate_windows(aoi_df: pd.DataFrame, aoi: str) -> pd.DataFrame:
    records = []
    g = aoi_df.sort_values("year").reset_index(drop=True)
    threshold = float(g[BASELINE_SELECTION_METRIC].quantile(HIGH_QUANTILE))

    for start_idx in range(0, len(g) - BASELINE_WINDOW_YEARS + 1):
        win = g.iloc[start_idx:start_idx + BASELINE_WINDOW_YEARS].copy()
        years = [int(y) for y in win["year"]]
        values = win[BASELINE_SELECTION_METRIC]

        contiguous      = _is_contiguous(win["year"])
        enough_n        = bool((win["n"] >= MIN_N_BLOCKS).all())
        enough_coverage = bool((win["coverage_mean"] >= MIN_COVERAGE_MEAN).all())
        has_values      = bool(values.notna().all())
        has_high_year   = bool((values >= threshold).any()) if has_values else True
        eligible = (
            contiguous
            and enough_n
            and enough_coverage
            and has_values
            and not has_high_year
        )

        mean_val = float(values.mean()) if has_values else np.nan
        std_val  = float(values.std(ddof=0)) if has_values else np.nan
        cv_val   = (
            std_val / mean_val
            if has_values
            and mean_val is not None
            and not np.isnan(mean_val)
            and mean_val != 0
            else np.nan
        )

        records.append({
            "aoi":                       aoi,
            "window_start_year":         years[0],
            "window_end_year":           years[-1],
            "baseline_years":            "-".join(str(y) for y in years),
            "selection_metric":          BASELINE_SELECTION_METRIC,
            "selection_metric_mean_km2": mean_val,
            "selection_metric_std_km2":  std_val,
            "selection_metric_cv":       cv_val,
            "coverage_mean":             float(win["coverage_mean"].mean()),
            "min_n":                     int(win["n"].min()),
            "high_quantile_threshold_km2": threshold,
            "contiguous":      contiguous,
            "enough_n":        enough_n,
            "enough_coverage": enough_coverage,
            "has_values":      has_values,
            "has_high_year":   has_high_year,
            "eligible":        eligible,
            "selected":        False,
        })

    return pd.DataFrame(records)


def select_baseline(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates[candidates["eligible"]].copy()
    if eligible.empty:
        raise ValueError(
            "No eligible baseline window found. Consider lowering "
            "MIN_COVERAGE_MEAN or reviewing annual JJA coverage."
        )

    min_cv    = float(eligible["selection_metric_cv"].min())
    near_best = eligible[
        eligible["selection_metric_cv"] <= min_cv + CV_TIE_TOLERANCE
    ].copy()
    near_best = near_best.sort_values(
        [
            "selection_metric_mean_km2",
            "selection_metric_cv",
            "coverage_mean",
            "window_start_year",
        ],
        ascending=[True, True, False, True],
    )
    return near_best.iloc[0]


def compute_baselines(df: pd.DataFrame) -> tuple:
    candidate_tables = []
    summary_rows     = []

    for aoi in AOIS:
        aoi_df     = df[df["aoi"] == aoi].copy()
        candidates = _candidate_windows(aoi_df, aoi)
        selected   = select_baseline(candidates)
        selected_key = (
            int(selected["window_start_year"]),
            int(selected["window_end_year"]),
        )
        candidates.loc[
            (candidates["window_start_year"] == selected_key[0])
            & (candidates["window_end_year"] == selected_key[1]),
            "selected",
        ] = True
        candidate_tables.append(candidates)

        baseline_years_str = str(selected["baseline_years"])
        if baseline_years_str in ("nan", "<NA>", "None", ""):
            raise ValueError(
                f"[{aoi}] Selected baseline has an invalid 'baseline_years' value: "
                f"{baseline_years_str!r}"
            )
        baseline_years = [int(y) for y in baseline_years_str.split("-")]
        base = aoi_df[aoi_df["year"].isin(baseline_years)].copy()
        row = {
            "aoi":                  aoi,
            "label":                DATA_LABEL,
            "baseline_method":      "auto_stable_3yr_combined_max",
            "baseline_years":       "-".join(str(y) for y in baseline_years),
            "window_start_year":    baseline_years[0],
            "window_end_year":      baseline_years[-1],
            "selection_metric":     BASELINE_SELECTION_METRIC,
            "selection_metric_cv":  float(selected["selection_metric_cv"]),
            "coverage_mean":        float(base["coverage_mean"].mean()),
            "min_n":                int(base["n"].min()),
        }
        for metric, _name in ANOMALY_METRICS:
            vals = base[metric].dropna()
            row[f"{metric}_baseline_mean"]    = float(vals.mean())
            row[f"{metric}_baseline_median"]  = float(vals.median())
            row[f"{metric}_baseline_std"]     = float(vals.std(ddof=0))
            row[f"{metric}_baseline_min"]     = float(vals.min())
            row[f"{metric}_baseline_max"]     = float(vals.max())
            row[f"{metric}_baseline_n_years"] = int(vals.shape[0])
        summary_rows.append(row)

    return pd.concat(candidate_tables, ignore_index=True), pd.DataFrame(summary_rows)


# ============================================================
# 6) ANOMALIES
# ============================================================

def compute_anomalies(df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, base_row in baselines.iterrows():
        aoi    = base_row["aoi"]
        aoi_df = df[df["aoi"] == aoi].copy()

        for metric, metric_label in ANOMALY_METRICS:
            baseline_value = float(base_row[f"{metric}_baseline_mean"])
            for _, row in aoi_df.iterrows():
                value = float(row[metric]) if not pd.isna(row[metric]) else np.nan
                anomaly = value - baseline_value if not np.isnan(value) else np.nan
                anomaly_pct = (
                    100.0 * (value / baseline_value - 1.0)
                    if baseline_value != 0 and not np.isnan(value)
                    else np.nan
                )
                fold_change = (
                    value / baseline_value
                    if baseline_value != 0 and not np.isnan(value)
                    else np.nan
                )
                rows.append({
                    "aoi":                aoi,
                    "label":              DATA_LABEL,
                    "metric":             metric,
                    "metric_label":       metric_label,
                    "year":               int(row["year"]),
                    "value_km2":          value,
                    "baseline_method":    base_row["baseline_method"],
                    "baseline_years":     base_row["baseline_years"],
                    "baseline_value_km2": baseline_value,
                    "anomaly_km2":        anomaly,
                    "anomaly_pct":        anomaly_pct,
                    "fold_change":        fold_change,
                    "n":                  int(row["n"]),
                    "coverage_mean":      float(row["coverage_mean"]),
                    "is_baseline_year": int(row["year"]) in [
                        int(y) for y in str(base_row["baseline_years"]).split("-")
                    ],
                    "is_positive_high_anomaly": (
                        bool(anomaly_pct >= LABEL_ANOMALY_THRESHOLD_PCT)
                        if not np.isnan(anomaly_pct) else False
                    ),
                })

    return pd.DataFrame(rows)


# ============================================================
# 7) PLOTTING - HELPERS
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


def _add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        PANEL_LABEL_X, PANEL_LABEL_Y, text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        clip_on=False,
    )


def _annotate_high_anomalies(ax: plt.Axes, sub: pd.DataFrame) -> None:
    valid = sub.dropna(subset=["anomaly_pct"])
    if valid.empty:
        return

    label_years = set(
        valid.loc[
            valid["anomaly_pct"] >= LABEL_ANOMALY_THRESHOLD_PCT,
            "year",
        ].astype(int)
    )
    positive = valid[valid["anomaly_pct"] > 0]
    if not positive.empty:
        label_years.add(int(positive.loc[positive["anomaly_pct"].idxmax(), "year"]))

    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.025
    for _, row in valid[valid["year"].isin(label_years)].iterrows():
        y = float(row["anomaly_pct"])
        ax.text(
            int(row["year"]),
            y + offset,
            f"{y:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=TICK_FONT_SIZE,
            color=COLOUR_ZERO,
            clip_on=False,
        )


# ============================================================
# 8) PLOTTING - FIGURE BUILDER
# ============================================================

def plot_aoi_anomalies(anomalies: pd.DataFrame, aoi: str, out_path: Path) -> None:
    aoi_df = anomalies[anomalies["aoi"] == aoi].copy()
    years  = sorted(aoi_df["year"].unique())
    baseline_years = sorted(
        aoi_df.loc[aoi_df["is_baseline_year"], "year"].unique()
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=True,
    )
    fig.subplots_adjust(hspace=HSPACE)

    panel_info = [
        ("combined_max_km2",  "Max Area Anomalies",  "(a)"),
        ("combined_mean_km2", "Mean Area Anomalies", "(b)"),
    ]

    for ax, (metric, ylabel, panel) in zip(axes, panel_info):
        sub = aoi_df[aoi_df["metric"] == metric].sort_values("year").copy()
        colors = [
            COLOUR_POSITIVE if v >= 0 else COLOUR_NEGATIVE
            for v in sub["anomaly_pct"].fillna(0)
        ]

        if baseline_years:
            ax.axvspan(
                min(baseline_years) - 0.5,
                max(baseline_years) + 0.5,
                color=COLOUR_BASELINE,
                alpha=BASELINE_SPAN_ALPHA,
                linewidth=0,
                zorder=0,
            )
        ax.axhline(0, color=COLOUR_ZERO, linewidth=0.8, zorder=2)
        ax.bar(
            sub["year"],
            sub["anomaly_pct"],
            width=0.72,
            color=colors,
            alpha=BAR_ALPHA,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
        ax.grid(
            axis="y",
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
            zorder=0,
        )
        _add_panel_label(ax, panel)

        vals = sub["anomaly_pct"].dropna()
        if not vals.empty:
            y_abs = max(abs(float(vals.min())), abs(float(vals.max())))
            y_abs = max(y_abs * 1.22, 25.0)
            ax.set_ylim(-y_abs, y_abs)

        _annotate_high_anomalies(ax, sub)

        ax.legend(
            handles=[
                mpatches.Patch(
                    color=COLOUR_POSITIVE, alpha=BAR_ALPHA, label="Above baseline"
                ),
                mpatches.Patch(
                    color=COLOUR_NEGATIVE, alpha=BAR_ALPHA, label="Below baseline"
                ),
                mpatches.Patch(
                    color=COLOUR_BASELINE,
                    alpha=BASELINE_SPAN_ALPHA + 0.2,
                    label="Baseline period",
                ),
            ],
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
            handlelength=2.2,
        )

    axes[-1].set_xticks(years)
    axes[-1].set_xticklabels(
        [str(y) for y in years],
        rotation=YEAR_LABEL_ROTATION,
        ha="right",
    )
    axes[-1].set_xlabel("Year")

    baseline_label = "-".join(str(int(y)) for y in baseline_years)
    fig.suptitle(
        f"{AOI_NAMES.get(aoi, aoi)}, {DATA_LABEL} JJA melt-area anomalies "
        f"(baseline {baseline_label})",
        fontsize=BASE_FONT_SIZE,
        fontweight="bold",
        y=0.95,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# 9) CSV EXPORT
# ============================================================

def print_baseline_summary(baselines: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    print("\nSelected baselines")
    print("-" * 72)
    for _, row in baselines.iterrows():
        aoi   = row["aoi"]
        years = row["baseline_years"]
        cv    = row["selection_metric_cv"]
        print(f"{aoi}: {years}  (combined_max CV={cv:.3f})")

        aoi_anom = anomalies[
            (anomalies["aoi"] == aoi)
            & (anomalies["metric"] == "combined_max_km2")
        ].copy()
        top = aoi_anom.sort_values("anomaly_pct", ascending=False).head(3)
        for _, anom in top.iterrows():
            print(
                f"  {int(anom['year'])}: "
                f"{anom['value_km2']:.1f} km2, "
                f"{anom['anomaly_pct']:+.0f}%, "
                f"{anom['fold_change']:.2f}x baseline"
            )


def save_csvs(
    candidates: pd.DataFrame,
    baselines: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (f"area_jja_baseline_candidates_{DATA_LABEL}.csv", candidates),
        (f"area_jja_baseline_summary_{DATA_LABEL}.csv",    baselines),
        (f"area_jja_anomalies_{DATA_LABEL}.csv",           anomalies),
    ]
    for name, table in outputs:
        path = CSV_OUTPUT_DIR / name
        table.to_csv(path, index=False)
        print(f"  Saved CSV -> {path}")


# ============================================================
# 10) MAIN
# ============================================================

def main() -> None:
    _apply_nature_rcparams()
    summary = load_summary(SUMMARY_CSV)
    candidates, baselines = compute_baselines(summary)
    anomalies = compute_anomalies(summary, baselines)

    print_baseline_summary(baselines, anomalies)
    save_csvs(candidates, baselines, anomalies)

    for aoi in AOIS:
        out_path = FIG_OUTPUT_DIR / f"jja_area_anomaly_pct_{aoi}_{DATA_LABEL}.png"
        plot_aoi_anomalies(anomalies, aoi, out_path)
        print(f"  Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
