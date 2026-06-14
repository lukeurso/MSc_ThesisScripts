# -*- coding: utf-8 -*-
"""
create_per_lake_delta_volume_csv_data.py

Reads per_lake_volume.csv and per_lake_mean_depth.csv (produced by
create_per_lake_volume_csv_data.py) and computes derived metrics for every
lake, written to one CSV per AOI.

Output CSVs
-----------
  per_lake_delta_volume_PTM.csv   (P*** lakes)
  per_lake_delta_volume_OST.csv   (C*** lakes)

Columns per lake (grouped together in order)
--------------------------------------------
  <id>_vol          observed volume [m³] for that window (NaN if no valid obs)
  <id>_mean_dpt_m   mean depth [m] used in the volume calculation
  <id>_delta_prop   proportional change from previous valid obs within same year
                      = (curr − prev) / prev
  <id>_delta_m3     absolute change [m³] from previous valid obs within same year
                      = curr − prev
  <id>_n_windows    windows elapsed since the previous valid observation
                      (crosses year boundaries; NaN for the first valid obs ever)

Year-reset rule (delta_prop and delta_m3 only)
----------------------------------------------
  No comparison is made across calendar-year boundaries.
  The first valid observation of a lake in each year yields NaN for both delta
  columns; the n_windows column is unaffected and counts all windows.

Special case
------------
  If the previous valid value within the same year is exactly 0.0,
  NEAR_ZERO (0.001) is used as the denominator for delta_prop to avoid
  division by zero while preserving a large positive fill signal.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1) USER SETTINGS
# ============================================================

INPUT_VOL_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\per_lake_volume.csv"
)

INPUT_DEPTH_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\per_lake_mean_depth.csv"
)

OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs"
)

DEBUG = True

NEAR_ZERO = 0.001  # substitute denominator when prev valid volume == 0.0

# Map AOI name → lake-ID prefix character used in per_lake_volume.csv
AOI_PREFIXES = {
    "PTM": "P",
    "OST": "C",
}


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


# ============================================================
# 3) PER-LAKE METRIC COMPUTATION
# ============================================================

def compute_lake_metrics(
    values: np.ndarray,
    years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all four metric arrays for a single lake's volume series.

    Parameters
    ----------
    values : float64 array  — volume [m³], NaN where observation is absent.
    years  : int array      — calendar year for each row.

    Returns
    -------
    vol         : copy of values (pass-through for convenience)
    delta_prop  : proportional change within year, NaN outside/across years
    delta_m3    : absolute change [m³] within year, NaN outside/across years
    n_windows   : windows since previous valid obs (year-agnostic), NaN for first
    """
    n = len(values)
    delta_prop = np.full(n, np.nan, dtype=np.float64)
    delta_m3   = np.full(n, np.nan, dtype=np.float64)
    n_windows  = np.full(n, np.nan, dtype=np.float64)

    prev_val       = None
    prev_year      = None
    prev_valid_idx = None

    for i in range(n):
        curr      = values[i]
        curr_year = int(years[i]) if not np.isnan(years[i]) else None

        if np.isnan(curr):
            continue  # blank source cell — do not update trackers

        # Windows-since (no year reset)
        if prev_valid_idx is not None:
            n_windows[i] = i - prev_valid_idx

        # Deltas (year-reset applies)
        if prev_val is not None and prev_year == curr_year:
            denom        = prev_val if prev_val != 0.0 else NEAR_ZERO
            delta_prop[i] = (curr - prev_val) / denom
            delta_m3[i]   = curr - prev_val
        # else: first valid obs this year → deltas stay NaN

        prev_val       = curr
        prev_year      = curr_year
        prev_valid_idx = i

    return values.copy(), delta_prop, delta_m3, n_windows


# ============================================================
# 4) MAIN
# ============================================================

def main():
    for path in (INPUT_VOL_CSV, INPUT_DEPTH_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

    debug_print(f"Reading: {INPUT_VOL_CSV}")
    vol_df   = pd.read_csv(INPUT_VOL_CSV)
    debug_print(f"Reading: {INPUT_DEPTH_CSV}")
    depth_df = pd.read_csv(INPUT_DEPTH_CSV)

    if "win_start" not in vol_df.columns:
        raise KeyError("Expected column 'win_start' not found in volume CSV.")

    years     = pd.to_datetime(vol_df["win_start"], errors="coerce").dt.year.to_numpy()
    meta_cols = [c for c in ("win_start", "win_end") if c in vol_df.columns]
    lake_cols = [c for c in vol_df.columns if c not in set(meta_cols)]

    debug_print(f"Windows : {len(vol_df)}")
    debug_print(f"Lakes   : {len(lake_cols)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for aoi, prefix_char in AOI_PREFIXES.items():
        aoi_lakes = [c for c in lake_cols if c.startswith(prefix_char)]
        if not aoi_lakes:
            debug_print(f"[{aoi}] No lakes found — skipping.")
            continue

        debug_print(f"\n[{aoi}] Building metrics for {len(aoi_lakes)} lake(s) …")

        col_data: dict[str, np.ndarray] = {}
        for lid in aoi_lakes:
            vol, dp, dm3, nwin = compute_lake_metrics(
                vol_df[lid].to_numpy(dtype=np.float64), years
            )
            mean_dpt = (
                depth_df[lid].to_numpy(dtype=np.float64)
                if lid in depth_df.columns
                else np.full(len(vol_df), np.nan)
            )
            col_data[f"{lid}_vol"]        = np.round(vol,      2)
            col_data[f"{lid}_mean_dpt_m"] = np.round(mean_dpt, 4)
            col_data[f"{lid}_delta_prop"] = np.round(dp,       6)
            col_data[f"{lid}_delta_m3"]   = np.round(dm3,      2)
            col_data[f"{lid}_n_windows"]  = nwin

        out_df = pd.concat(
            [vol_df[meta_cols].reset_index(drop=True), pd.DataFrame(col_data)],
            axis=1,
        )

        out_path = OUTPUT_DIR / f"per_lake_delta_volume_{aoi}.csv"
        out_df.to_csv(out_path, index=False, na_rep="")
        print(f"\nWritten : {out_path}")
        print(f"Shape   : {out_df.shape[0]} rows × {out_df.shape[1]} columns")

        if DEBUG:
            print(f"\n── QA [{aoi}] ─────────────────────────────────────────────")

            def _stat_line(label, s):
                if s.empty:
                    print(f"  {label}: no valid cells")
                else:
                    print(
                        f"  {label}: "
                        f"min={s.min():.3g}  max={s.max():.3g}  "
                        f"mean={s.mean():.3g}  n={len(s)}"
                    )

            _stat_line("vol [m³]      ", out_df[[c for c in out_df if c.endswith("_vol")]].stack().dropna())
            _stat_line("mean_dpt_m    ", out_df[[c for c in out_df if c.endswith("_mean_dpt_m")]].stack().dropna())
            _stat_line("delta_prop    ", out_df[[c for c in out_df if c.endswith("_delta_prop")]].stack().dropna())
            _stat_line("delta_m3 [m³] ", out_df[[c for c in out_df if c.endswith("_delta_m3")]].stack().dropna())
            _stat_line("n_windows     ", out_df[[c for c in out_df if c.endswith("_n_windows")]].stack().dropna())

    print("\nDone.")


if __name__ == "__main__":
    main()
