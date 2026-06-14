"""
project_melt_area_by_aspect_elev_bins.py

For each melt-season two-day window, estimates the melt area (m²) that would
be expected in the **unobserved** portion of each aspect-split elevation bin,
by applying the observed melt density (melt area per unit footprint area) to
the area of the bin that lies outside the footprint.

Method
------
For a given window and bin:

    observed_density  = observed_melt_area  / footprint_area
    unobserved_area   = max(0,  total_bin_area − footprint_area)
    projected_area    = observed_density × unobserved_area

Conditions:
  • footprint_area is NaN  → projected = NaN  (no footprint data for this window/AOI)
  • footprint_area == 0    → projected = NaN  (footprint does not cover this bin;
                                               density cannot be estimated)
  • footprint_area > 0     → apply formula above
    – observed_melt_area is NaN here is treated as 0 (melt file may be missing
      for this window even though a footprint exists)

Inputs
------
  footprint_area_by_aspect_elev_bins.csv   — footprint area (m²) per bin/window
  slush_area_by_aspect_elev_bins.csv       — observed slush area (m²) per bin/window
  lake_area_by_aspect_elev_bins.csv        — observed lake area (m²) per bin/window
  PTM_aspect_elevation_bins.shp            — bin geometries (for total bin areas)
  OST_aspect_elevation_bins.shp

Outputs
-------
  projected_slush_area_by_aspect_elev_bins.csv  — estimated slush area in unobserved zone (m²)
  projected_lake_area_by_aspect_elev_bins.csv   — estimated lake area  in unobserved zone (m²)
  total_slush_area_by_aspect_elev_bins.csv      — total slush area (observed + projected) per bin (m²)
  total_lake_area_by_aspect_elev_bins.csv       — total lake area  (observed + projected) per bin (m²)

Column naming matches the other bin CSVs:
  OST_0_100, …, OST_900_1000
  OST_1000_1100_E / _W, …, OST_1400_1500_E / _W
  PTM_1000_1100_E / _W, …  (same pattern)

Blank cells : footprint_area is NaN or 0 for that bin/window (density unknown).
Zero cells  : footprint covers this bin but no melt observed → projected area = 0,
              OR the entire bin is observed (unobserved_area = 0).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1) USER SETTINGS
# ============================================================

PTM_BINS_SHP = Path(r"Q:\ThesisData\data\study_areas\elevation_bins\PTM_aspect_elevation_bins.shp")
OST_BINS_SHP = Path(r"Q:\ThesisData\data\study_areas\elevation_bins\OST_aspect_elevation_bins.shp")

FOOTPRINT_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\footprint_area_by_aspect_elev_bins.csv")
SLUSH_AREA_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\slush_area_by_aspect_elev_bins.csv")
LAKE_AREA_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\lake_area_by_aspect_elev_bins.csv")

PROJ_SLUSH_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\projected_slush_area_by_aspect_elev_bins.csv")
PROJ_LAKE_CSV   = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\projected_lake_area_by_aspect_elev_bins.csv")

TOTAL_SLUSH_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\total_slush_area_by_aspect_elev_bins.csv")
TOTAL_LAKE_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\total_lake_area_by_aspect_elev_bins.csv")

# Projected CRS used for all area calculations (must match the source CSVs)
PROJECTED_CRS = "EPSG:3995"

DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


def col_name(aoi: str, start_elv, end_elv, sector: str) -> str:
    """
    Return the output column name for an aspect-split elevation bin.

    sector='main'  → 'OST_0_100'
    sector='east'  → 'OST_1000_1100_E'
    sector='west'  → 'OST_1000_1100_W'
    """
    base = f"{aoi}_{int(start_elv)}_{int(end_elv)}"
    if sector == "east":
        return base + "_E"
    if sector == "west":
        return base + "_W"
    return base


def build_bin_columns(bins_gdf: gpd.GeoDataFrame, aoi: str) -> list:
    cols = []
    for _, row in bins_gdf.iterrows():
        cols.append(col_name(aoi, row["start_elv"], row["end_elv"], row["sector"]))
    return cols


def all_bin_columns(ost_bins: gpd.GeoDataFrame, ptm_bins: gpd.GeoDataFrame) -> list:
    return build_bin_columns(ost_bins, "OST") + build_bin_columns(ptm_bins, "PTM")


def load_bins(shp_path: Path, aoi: str) -> gpd.GeoDataFrame:
    """
    Read an aspect-split elevation-bin shapefile, reproject to PROJECTED_CRS,
    and return a GeoDataFrame sorted by (start_elv, sector).
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] Bin shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"[{aoi}] Bin shapefile contains no features: {shp_path}")

    for col in ("start_elv", "end_elv", "sector"):
        if col not in gdf.columns:
            raise KeyError(
                f"[{aoi}] Bin shapefile is missing column '{col}'. "
                "Run split_elevation_bins_by_aspect.py to generate the required files."
            )

    if gdf.crs is None:
        warnings.warn(f"[{aoi}] Bin shapefile has no CRS; area calculations may be unreliable.")
    elif str(gdf.crs) != PROJECTED_CRS:
        debug_print(f"[{aoi}] Reprojecting bins from {gdf.crs} to {PROJECTED_CRS}")
        gdf = gdf.to_crs(PROJECTED_CRS)

    gdf = gdf.sort_values(["start_elv", "sector"]).reset_index(drop=True)
    debug_print(
        f"[{aoi}] Loaded {len(gdf)} bin feature(s): "
        f"{(gdf['sector'] == 'main').sum()} main, "
        f"{(gdf['sector'] == 'east').sum()} east, "
        f"{(gdf['sector'] == 'west').sum()} west"
    )
    return gdf


def build_total_bin_areas(bins_gdf: gpd.GeoDataFrame, aoi: str) -> dict:
    """
    Return a dict mapping col_name -> total bin area in m² (from the shapefile geometry).
    Bins with null/empty/zero geometry are excluded.
    """
    areas = {}
    for _, row in bins_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty or geom.area == 0.0:
            continue
        cname = col_name(aoi, row["start_elv"], row["end_elv"], row["sector"])
        areas[cname] = float(geom.area)
    return areas


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce win_start / win_end to ISO YYYY-MM-DD strings."""
    out = df.copy()
    for col in ("win_start", "win_end"):
        if col not in out.columns:
            raise KeyError(f"CSV is missing required column: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def project_melt(
    footprint_df: pd.DataFrame,
    melt_df: pd.DataFrame,
    bin_cols: list,
    total_bin_areas: dict,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the projected-melt DataFrame.

    For each row (window) and each bin column:
      - footprint_area = footprint_df[col]
      - observed_melt  = melt_df[col]   (NaN treated as 0 when footprint_area > 0)
      - density        = observed_melt / footprint_area
      - unobserved     = max(0, total_bin_area - footprint_area)
      - projected      = density * unobserved

    Returns a copy of footprint_df with bin columns replaced by projected values.
    """
    # Start from a copy of footprint_df (carries template metadata columns)
    out_df = footprint_df.copy()

    for col in bin_cols:
        if col not in total_bin_areas:
            # Bin has no valid geometry — leave as NaN
            out_df[col] = np.nan
            continue

        total_area = total_bin_areas[col]

        fp_vals   = footprint_df[col] if col in footprint_df.columns else pd.Series(np.nan, index=footprint_df.index)
        melt_vals = melt_df[col]      if col in melt_df.columns      else pd.Series(np.nan, index=melt_df.index)

        projected = pd.array([np.nan] * len(out_df), dtype="float64")

        for i in range(len(out_df)):
            fp = fp_vals.iloc[i]

            if pd.isna(fp) or fp == 0.0:
                # No footprint data for this bin/window → cannot estimate density
                projected[i] = np.nan
                continue

            melt = melt_vals.iloc[i]
            # Treat NaN melt (missing melt file) as 0 when footprint is present
            if pd.isna(melt):
                melt = 0.0

            density      = melt / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)

        out_df[col] = projected

    n_filled = out_df[bin_cols].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label}: {n_filled}/{len(out_df)} window(s) "
        "with at least one bin projected"
    )
    return out_df


def combine_areas(
    observed_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    bin_cols: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the combined (observed + projected) area DataFrame.

    For each row (window) and each bin column:
      - Where projected is NaN (no footprint → density unknown):
          combined = NaN  (total cannot be estimated)
      - Where projected is not NaN:
          combined = observed (NaN treated as 0) + projected

    This is consistent with how project_melt treats missing observations:
    a missing melt file when a footprint exists is assumed to mean zero melt.

    Returns a copy of projected_df with bin columns replaced by combined values.
    """
    out_df = projected_df.copy()

    for col in bin_cols:
        proj_vals = projected_df[col] if col in projected_df.columns else pd.Series(np.nan, index=projected_df.index)
        obs_vals  = observed_df[col]  if col in observed_df.columns  else pd.Series(np.nan, index=observed_df.index)

        combined = proj_vals.copy()  # start as NaN where projected is NaN
        mask = proj_vals.notna()
        combined[mask] = obs_vals[mask].fillna(0.0) + proj_vals[mask]
        combined = combined.round(2)
        out_df[col] = combined

    n_filled = out_df[bin_cols].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label}: {n_filled}/{len(out_df)} window(s) "
        "with at least one bin combined"
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

    # --------------------------------------------------------
    # Load bin shapefiles and compute total bin areas
    # --------------------------------------------------------
    debug_print("\nLoading bin shapefiles …")
    ost_bins = load_bins(OST_BINS_SHP, "OST")
    ptm_bins = load_bins(PTM_BINS_SHP, "PTM")

    total_bin_areas = {}
    total_bin_areas.update(build_total_bin_areas(ost_bins, "OST"))
    total_bin_areas.update(build_total_bin_areas(ptm_bins, "PTM"))
    debug_print(f"  Total bin areas computed for {len(total_bin_areas)} bins")

    bin_cols = all_bin_columns(ost_bins, ptm_bins)
    debug_print(f"  Bin columns: {len(bin_cols)}")

    # Ensure projected output has all bin columns (add any missing ones as NaN)
    for col in bin_cols:
        for df in (footprint_df, slush_df, lake_df):
            if col not in df.columns:
                df[col] = np.nan

    # --------------------------------------------------------
    # Compute projected areas
    # --------------------------------------------------------
    debug_print("\nProjecting melt areas into unobserved bin zones …")
    proj_slush_df = project_melt(footprint_df, slush_df, bin_cols, total_bin_areas, "slush")
    proj_lake_df  = project_melt(footprint_df, lake_df,  bin_cols, total_bin_areas, "lake")

    # --------------------------------------------------------
    # Compute combined (observed + projected) areas
    # --------------------------------------------------------
    debug_print("\nCombining observed and projected melt areas …")
    total_slush_df = combine_areas(slush_df, proj_slush_df, bin_cols, "slush")
    total_lake_df  = combine_areas(lake_df,  proj_lake_df,  bin_cols, "lake")

    # --------------------------------------------------------
    # Enforce column order: template columns first, then bin columns
    # --------------------------------------------------------
    original_cols = list(pd.read_csv(FOOTPRINT_CSV, nrows=0).columns)
    for col in bin_cols:
        if col not in original_cols:
            original_cols.append(col)

    proj_slush_df  = proj_slush_df[original_cols]
    proj_lake_df   = proj_lake_df[original_cols]
    total_slush_df = total_slush_df[original_cols]
    total_lake_df  = total_lake_df[original_cols]

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
    aoi_config = [("OST", ost_bins), ("PTM", ptm_bins)]
    for aoi, bins_gdf in aoi_config:
        aoi_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        n_main   = (bins_gdf["sector"] == "main").sum()
        n_east   = (bins_gdf["sector"] == "east").sum()
        n_west   = (bins_gdf["sector"] == "west").sum()
        bin_desc = f"{n_main} main + {n_east} east + {n_west} west bins"

        for layer_label, df in (
            ("projected slush",      proj_slush_df),
            ("projected lake",       proj_lake_df),
            ("total combined slush", total_slush_df),
            ("total combined lake",  total_lake_df),
        ):
            n_blank   = df[aoi_cols].isna().all(axis=1).sum()
            n_covered = (df[aoi_cols] > 0).any(axis=1).sum()
            n_zero    = (
                df[aoi_cols].notna().any(axis=1)
                & ~(df[aoi_cols] > 0).any(axis=1)
            ).sum()
            debug_print(
                f"  {aoi} {layer_label} ({bin_desc}): "
                f"{n_covered} window(s) with ≥1 bin projected  |  "
                f"{n_zero} window(s) with data but zero projected area  |  "
                f"{n_blank} window(s) with no data (all blank)"
            )


if __name__ == "__main__":
    main()
