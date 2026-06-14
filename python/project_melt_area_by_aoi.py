"""
project_melt_area_by_AOI.py

For each melt-season two-day window, estimates the melt area (m²) that would
be expected in the **unobserved** portion of each AOI, by applying the observed
melt density (melt area per unit footprint area) across the entire observed
footprint to the unobserved area of the AOI.

Method
------
For a given window and AOI:

    footprint_area_AOI = sum of footprint areas across all aspect-elevation
                         bins within the AOI
    observed_melt_AOI  = sum of observed melt areas across all bins

    observed_density   = observed_melt_AOI / footprint_area_AOI
    unobserved_area    = max(0, total_AOI_area − footprint_area_AOI)
    projected_area     = observed_density × unobserved_area

Bin aggregation:
  NaN bins are treated as 0 when summing, provided at least one bin is
  non-NaN (i.e. partial footprint coverage across the AOI is valid).
  If ALL bins are NaN the AOI total is NaN.

Conditions on the aggregated AOI values:
  • footprint_area_AOI is NaN or 0 → projected = NaN (density cannot be estimated)
  • footprint_area_AOI > 0         → apply formula above
    – observed_melt_AOI NaN treated as 0 (missing melt file when footprint
      exists → assume no melt)

Inputs
------
  footprint_area_by_aspect_elev_bins.csv  — footprint area (m²) per bin/window
  slush_area_by_aspect_elev_bins.csv      — observed slush area (m²) per bin/window
  lake_area_by_aspect_elev_bins.csv       — observed lake area  (m²) per bin/window
  PTM_AOI_1500m.shp                       — AOI polygon (for total AOI area)
  OST_AOI_1500m.shp

Outputs
-------
  projected_slush_area_by_AOI.csv  — estimated slush area in unobserved zone (m²)
  projected_lake_area_by_AOI.csv   — estimated lake area  in unobserved zone (m²)
  total_slush_area_by_AOI.csv      — total slush area (observed + projected)  (m²)
  total_lake_area_by_AOI.csv       — total lake area  (observed + projected)  (m²)

Each output has one column per AOI:  OST, PTM

Blank cells : footprint_area_AOI is NaN or 0 for that window (density unknown).
Zero cells  : footprint covers this AOI but no melt observed → projected = 0,
              OR the entire AOI is already observed (unobserved_area = 0).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1) USER SETTINGS
# ============================================================

PTM_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp")
OST_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp")

FOOTPRINT_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\footprint_area_by_aspect_elev_bins.csv")
SLUSH_AREA_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\slush_area_by_aspect_elev_bins.csv")
LAKE_AREA_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\lake_area_by_aspect_elev_bins.csv")

PROJ_SLUSH_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\projected_slush_area_by_AOI.csv")
PROJ_LAKE_CSV   = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\projected_lake_area_by_AOI.csv")
TOTAL_SLUSH_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\total_slush_area_by_AOI.csv")
TOTAL_LAKE_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\total_lake_area_by_AOI.csv")

# Projected CRS used for all area calculations (must match the source CSVs)
PROJECTED_CRS = "EPSG:3995"

# AOI names and their shapefiles — order determines column order in outputs
AOI_SHAPEFILES = {
    "OST": OST_AOI_SHP,
    "PTM": PTM_AOI_SHP,
}

DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce win_start / win_end to ISO YYYY-MM-DD strings."""
    out = df.copy()
    for col in ("win_start", "win_end"):
        if col not in out.columns:
            raise KeyError(f"CSV is missing required column: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def load_aoi_area(shp_path: Path, aoi: str) -> float:
    """
    Read an AOI shapefile, reproject to PROJECTED_CRS, and return the total
    area (m²) of all features unioned together.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] AOI shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"[{aoi}] AOI shapefile contains no features: {shp_path}")

    if gdf.crs is None:
        warnings.warn(f"[{aoi}] AOI shapefile has no CRS; area calculations may be unreliable.")
    elif str(gdf.crs) != PROJECTED_CRS:
        debug_print(f"[{aoi}] Reprojecting AOI from {gdf.crs} to {PROJECTED_CRS}")
        gdf = gdf.to_crs(PROJECTED_CRS)

    total_area = float(gdf.geometry.union_all().area)
    debug_print(f"[{aoi}] AOI total area: {total_area:,.0f} m²")
    return total_area


def aggregate_bins_to_aoi(df: pd.DataFrame, aoi: str) -> pd.Series:
    """
    Sum all columns starting with '<aoi>_' across the DataFrame.

    Per-row behaviour:
      • All bins NaN  → NaN  (no footprint / melt data for this window)
      • At least one bin non-NaN → sum of non-NaN bins (NaN bins → 0)
    """
    prefix = f"{aoi}_"
    aoi_cols = [c for c in df.columns if c.startswith(prefix)]

    if not aoi_cols:
        debug_print(f"  [{aoi}] No bin columns found with prefix '{prefix}' — returning NaN series")
        return pd.Series(np.nan, index=df.index)

    subset = df[aoi_cols]
    result = subset.sum(axis=1, skipna=True)
    all_nan = subset.isna().all(axis=1)
    result[all_nan] = np.nan
    return result


def identify_meta_cols(df: pd.DataFrame, aoi_names: list) -> list:
    """
    Return column names that are not bin data columns.
    Bin columns start with '<AOI>_'; everything else is metadata.
    """
    return [
        c for c in df.columns
        if not any(c.startswith(f"{aoi}_") for aoi in aoi_names)
    ]


def project_melt_aoi(
    footprint_df: pd.DataFrame,
    melt_df: pd.DataFrame,
    aoi_names: list,
    total_aoi_areas: dict,
    meta_cols: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the projected-melt DataFrame with one column per AOI.

    For each row (window) and each AOI:
      footprint_area = aggregated footprint across all bins
      observed_melt  = aggregated melt   across all bins (NaN → 0 if fp > 0)
      density        = observed_melt / footprint_area
      unobserved     = max(0, total_AOI_area - footprint_area)
      projected      = density * unobserved
    """
    out_df = footprint_df[meta_cols].copy()

    for aoi in aoi_names:
        total_area = total_aoi_areas[aoi]
        fp_aoi   = aggregate_bins_to_aoi(footprint_df, aoi)
        melt_aoi = aggregate_bins_to_aoi(melt_df,      aoi)

        projected = np.full(len(out_df), np.nan)

        for i in range(len(out_df)):
            fp = fp_aoi.iloc[i]

            if pd.isna(fp) or fp == 0.0:
                # Cannot estimate density — leave blank
                projected[i] = np.nan
                continue

            melt = melt_aoi.iloc[i]
            if pd.isna(melt):
                melt = 0.0

            density      = melt / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)

        out_df[aoi] = projected

    n_filled = out_df[aoi_names].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label}: {n_filled}/{len(out_df)} window(s) "
        "with at least one AOI projected"
    )
    return out_df


def combine_areas_aoi(
    observed_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    aoi_names: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the combined (observed + projected) area DataFrame.

    For each row (window) and each AOI:
      • projected is NaN (no footprint → density unknown):
            combined = NaN  (total cannot be estimated)
      • projected is not NaN:
            combined = observed_AOI (NaN → 0) + projected

    This matches the semantics of project_melt_aoi: a missing melt file when
    a footprint exists is assumed to mean zero melt.
    """
    out_df = projected_df.copy()

    for aoi in aoi_names:
        obs_aoi  = aggregate_bins_to_aoi(observed_df, aoi)
        proj_col = projected_df[aoi]

        combined = proj_col.copy()
        mask = proj_col.notna()
        combined[mask] = obs_aoi[mask].fillna(0.0) + proj_col[mask]
        combined = combined.round(2)
        out_df[aoi] = combined

    n_filled = out_df[aoi_names].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label}: {n_filled}/{len(out_df)} window(s) "
        "with at least one AOI combined"
    )
    return out_df


# ============================================================
# 3) MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Load input CSVs
    # --------------------------------------------------------
    for csv_path in (FOOTPRINT_CSV, SLUSH_AREA_CSV, LAKE_AREA_CSV):
        if not csv_path.exists():
            raise FileNotFoundError(f"Required input CSV not found: {csv_path}")

    debug_print("Loading input CSVs …")
    footprint_df = normalize_date_columns(pd.read_csv(FOOTPRINT_CSV))
    slush_df     = normalize_date_columns(pd.read_csv(SLUSH_AREA_CSV))
    lake_df      = normalize_date_columns(pd.read_csv(LAKE_AREA_CSV))
    debug_print(f"  Footprint rows: {len(footprint_df)}")
    debug_print(f"  Slush area rows: {len(slush_df)}")
    debug_print(f"  Lake area rows:  {len(lake_df)}")

    if len(footprint_df) != len(slush_df) or len(footprint_df) != len(lake_df):
        warnings.warn(
            "Input CSVs have different row counts. "
            "Rows are matched by position — verify all CSVs share the same template."
        )

    aoi_names = list(AOI_SHAPEFILES.keys())
    meta_cols = identify_meta_cols(footprint_df, aoi_names)
    debug_print(f"  Metadata columns: {meta_cols}")

    # --------------------------------------------------------
    # Load AOI shapefiles and compute total AOI areas
    # --------------------------------------------------------
    debug_print("\nLoading AOI shapefiles …")
    total_aoi_areas = {}
    for aoi, shp_path in AOI_SHAPEFILES.items():
        total_aoi_areas[aoi] = load_aoi_area(shp_path, aoi)

    # --------------------------------------------------------
    # Compute projected areas
    # --------------------------------------------------------
    debug_print("\nProjecting melt areas into unobserved AOI zones …")
    proj_slush_df = project_melt_aoi(
        footprint_df, slush_df, aoi_names, total_aoi_areas, meta_cols, "slush"
    )
    proj_lake_df = project_melt_aoi(
        footprint_df, lake_df, aoi_names, total_aoi_areas, meta_cols, "lake"
    )

    # --------------------------------------------------------
    # Compute combined (observed + projected) areas
    # --------------------------------------------------------
    debug_print("\nCombining observed and projected melt areas …")
    total_slush_df = combine_areas_aoi(slush_df, proj_slush_df, aoi_names, "slush")
    total_lake_df  = combine_areas_aoi(lake_df,  proj_lake_df,  aoi_names, "lake")

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    for csv_path in (PROJ_SLUSH_CSV, PROJ_LAKE_CSV, TOTAL_SLUSH_CSV, TOTAL_LAKE_CSV):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    proj_slush_df.to_csv(PROJ_SLUSH_CSV,   index=False)
    proj_lake_df.to_csv(PROJ_LAKE_CSV,     index=False)
    total_slush_df.to_csv(TOTAL_SLUSH_CSV, index=False)
    total_lake_df.to_csv(TOTAL_LAKE_CSV,   index=False)

    debug_print(f"\nProjected slush area written to:      {PROJ_SLUSH_CSV}")
    debug_print(f"Projected lake area written to:       {PROJ_LAKE_CSV}")
    debug_print(f"Total combined slush area written to: {TOTAL_SLUSH_CSV}")
    debug_print(f"Total combined lake area written to:  {TOTAL_LAKE_CSV}")

    # --------------------------------------------------------
    # QA summary
    # --------------------------------------------------------
    debug_print("\n── QA Summary ──────────────────────────────────────────")
    for aoi in aoi_names:
        for layer_label, df in (
            ("projected slush",      proj_slush_df),
            ("projected lake",       proj_lake_df),
            ("total combined slush", total_slush_df),
            ("total combined lake",  total_lake_df),
        ):
            n_blank   = df[aoi].isna().sum()
            n_covered = (df[aoi] > 0).sum()
            n_zero    = (df[aoi].notna() & ~(df[aoi] > 0)).sum()
            debug_print(
                f"  {aoi} {layer_label}: "
                f"{n_covered} window(s) with projected area > 0  |  "
                f"{n_zero} window(s) with data but zero projected area  |  "
                f"{n_blank} window(s) with no data (blank)"
            )


if __name__ == "__main__":
    main()
