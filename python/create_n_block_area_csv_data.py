#!/usr/bin/env python3
"""
create_n_block_area_csv_data.py

NOTE:
This script remains available as the area-only generator.
For the unified one-command area+volume workflow, run
create_n_block_csv_data.py.

Generates 10 CSV files derived from 5-window block shapefiles produced by
create_n_day_blocks.py, equivalent to the per-window CSVs consumed by
compare_select_methods_by_visualize_melt_area.py.

All outputs use block_start / block_end in place of win_start / win_end.

Input block shapefile naming convention (inside AOI-specific folders):
    {YYYY-MM-DD}_{YYYY-MM-DD}_{layer_type}.shp
    e.g.  2013-05-01_2013-05-10_footprint.shp

Outputs (written to OUTPUT_DIR):
  Method A – aspect-split elevation-bin projection:
    5_win_block_select_by_apect_elev_bin_coverage.csv
    5_win_block_slush_area_by_aspect_elev_bins.csv
    5_win_block_lake_area_by_aspect_elev_bins.csv
    5_win_block_projected_slush_area_by_aspect_elev_bins.csv
    5_win_block_projected_lake_area_by_aspect_elev_bins.csv

  Method B – AOI-scale projection:
    5_win_block_total_summary_raw.csv
    5_win_block_projected_slush_area_by_AOI.csv
    5_win_block_projected_lake_area_by_AOI.csv
    5_win_block_total_slush_area_by_AOI.csv
    5_win_block_total_lake_area_by_AOI.csv

Dependencies: geopandas, pandas, numpy, shapely
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1) USER SETTINGS  ← edit these before running
# ============================================================

INPUT_PTM_DIR = Path(r"Q:\ThesisData\data\blocks\5_window\all_PTM_files")
INPUT_OST_DIR = Path(r"Q:\ThesisData\data\blocks\5_window\all_OST_files")

OUTPUT_DIR = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block")

# Aspect-split elevation bin shapefiles (from split_elevation_bins_by_aspect.py)
PTM_BINS_SHP = Path(r"Q:\ThesisData\data\study_areas\elevation_bins\PTM_aspect_elevation_bins.shp")
OST_BINS_SHP = Path(r"Q:\ThesisData\data\study_areas\elevation_bins\OST_aspect_elevation_bins.shp")

# AOI boundary shapefiles (used for footprint proportion + AOI-level projection)
PTM_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp")
OST_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp")

PROJECTED_CRS = "EPSG:3995"

DEBUG = True


# ============================================================
# 2) CONSTANTS
# ============================================================

# Regex matching block shapefile names: YYYY-MM-DD_YYYY-MM-DD_{layer}.shp
# Note: no AOI prefix — files live in AOI-specific folders.
_BLOCK_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(footprint|slush|lake)\.shp$",
    re.IGNORECASE,
)

AOIS = ["OST", "PTM"]

AOI_INPUT_DIRS = {
    "OST": INPUT_OST_DIR,
    "PTM": INPUT_PTM_DIR,
}

AOI_BINS_SHPS = {
    "OST": OST_BINS_SHP,
    "PTM": PTM_BINS_SHP,
}

AOI_BOUNDARY_SHPS = {
    "OST": OST_AOI_SHP,
    "PTM": PTM_AOI_SHP,
}


# ============================================================
# 3) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


def col_name(aoi: str, start_elv, end_elv, sector: str) -> str:
    """
    Return the bin column name matching the project convention:
      sector='main' → 'OST_0_100'
      sector='east' → 'OST_1000_1100_E'
      sector='west' → 'OST_1000_1100_W'
    """
    base = f"{aoi}_{int(start_elv)}_{int(end_elv)}"
    if sector == "east":
        return base + "_E"
    if sector == "west":
        return base + "_W"
    return base


def build_bin_columns(bins_gdf: gpd.GeoDataFrame, aoi: str) -> list:
    """Ordered list of bin column names for one AOI."""
    return [
        col_name(aoi, row["start_elv"], row["end_elv"], row["sector"])
        for _, row in bins_gdf.iterrows()
    ]


def all_bin_columns(bins_gdfs: dict) -> list:
    """All bin column names: OST bins first, then PTM."""
    cols = []
    for aoi in AOIS:
        cols.extend(build_bin_columns(bins_gdfs[aoi], aoi))
    return cols


def load_bins(shp_path: Path, aoi: str) -> gpd.GeoDataFrame:
    """Load and reproject an aspect-split elevation-bin shapefile."""
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] Bin shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"[{aoi}] Bin shapefile is empty: {shp_path}")
    for col in ("start_elv", "end_elv", "sector"):
        if col not in gdf.columns:
            raise KeyError(
                f"[{aoi}] Bin shapefile missing column '{col}'. "
                "Run split_elevation_bins_by_aspect.py first."
            )
    if gdf.crs is None:
        warnings.warn(f"[{aoi}] Bin shapefile has no CRS.")
    elif str(gdf.crs) != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    gdf = gdf.sort_values(["start_elv", "sector"]).reset_index(drop=True)
    n_main = (gdf["sector"] == "main").sum()
    n_east = (gdf["sector"] == "east").sum()
    n_west = (gdf["sector"] == "west").sum()
    debug_print(
        f"[{aoi}] Bins loaded: {n_main} main, {n_east} east, {n_west} west"
    )
    return gdf


def load_aoi_area(shp_path: Path, aoi: str) -> float:
    """Return the total area (m²) of an AOI boundary shapefile."""
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] AOI shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"[{aoi}] AOI shapefile is empty: {shp_path}")
    if gdf.crs is None:
        warnings.warn(f"[{aoi}] AOI shapefile has no CRS.")
    elif str(gdf.crs) != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    area = float(gdf.geometry.union_all().area)
    debug_print(f"[{aoi}] AOI total area: {area:,.0f} m²")
    return area


def build_bin_total_areas(bins_gdf: gpd.GeoDataFrame, aoi: str) -> dict:
    """Dict mapping bin col_name → total bin area in m² (from shapefile geometry)."""
    areas = {}
    for _, row in bins_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty or geom.area == 0.0:
            continue
        cname = col_name(aoi, row["start_elv"], row["end_elv"], row["sector"])
        areas[cname] = float(geom.area)
    return areas


# ============================================================
# 4) BLOCK DISCOVERY
# ============================================================

def discover_blocks(input_dirs: dict) -> dict:
    """
    Scan each AOI block folder for shapefiles matching the block naming pattern.

    Returns
    -------
    Nested dict:  {aoi: {(block_start_str, block_end_str): {layer_type: Path}}}
    """
    result = {aoi: {} for aoi in AOIS}

    for aoi, folder in input_dirs.items():
        if not folder.exists():
            warnings.warn(f"[{aoi}] Block folder not found: {folder}")
            continue
        for shp in sorted(folder.glob("*.shp")):
            m = _BLOCK_RE.match(shp.name)
            if not m:
                continue
            bs, be, layer = m.group(1), m.group(2), m.group(3).lower()
            result[aoi].setdefault((bs, be), {})[layer] = shp
        debug_print(f"[{aoi}] Discovered {len(result[aoi])} block period(s)")

    return result


def union_block_periods(blocks: dict) -> list:
    """
    Sorted list of all unique (block_start, block_end) string pairs
    across all AOIs.
    """
    periods: set = set()
    for aoi_blocks in blocks.values():
        periods.update(aoi_blocks.keys())
    return sorted(periods)


def read_block_shp(path: Path, aoi: str, layer: str) -> gpd.GeoDataFrame | None:
    """
    Read a block shapefile and reproject to PROJECTED_CRS.
    Returns None on read failure or if the file is empty.
    """
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        warnings.warn(f"[{aoi}] Cannot read {layer} {path.name}: {exc}")
        return None
    if gdf.empty:
        debug_print(f"  [{aoi}] Empty {layer} shapefile: {path.name}")
        return None
    if gdf.crs is None:
        warnings.warn(f"[{aoi}] {path.name} has no CRS.")
    elif str(gdf.crs) != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    return gdf


# ============================================================
# 5) SPATIAL COMPUTATIONS
# ============================================================

def _zero_bin_dict(bins_gdf: gpd.GeoDataFrame, aoi: str) -> dict:
    """Return a dict of col_name → 0.0 for every valid bin in bins_gdf."""
    return {
        col_name(aoi, r["start_elv"], r["end_elv"], r["sector"]): 0.0
        for _, r in bins_gdf.iterrows()
        if r["geometry"] is not None and not r["geometry"].is_empty
    }


def compute_footprint_bin_areas(
    bins_gdf: gpd.GeoDataFrame,
    fp_gdf: gpd.GeoDataFrame | None,
    aoi: str,
) -> dict:
    """
    Footprint area (m²) intersecting each elevation bin.

    fp_gdf is None  → all bins = NaN (no footprint for this block/AOI).
    fp_gdf is empty → all bins = 0.0 (footprint file exists but is empty).
    """
    if fp_gdf is None:
        return {}   # caller leaves these bins as NaN
    if fp_gdf.empty:
        return _zero_bin_dict(bins_gdf, aoi)

    fp_union = fp_gdf.union_all()
    result = {}
    for _, br in bins_gdf.iterrows():
        geom = br["geometry"]
        if geom is None or geom.is_empty or geom.area == 0.0:
            continue
        cname = col_name(aoi, br["start_elv"], br["end_elv"], br["sector"])
        result[cname] = float(geom.intersection(fp_union).area)
    return result


def compute_footprint_bin_proportions(
    bins_gdf: gpd.GeoDataFrame,
    fp_gdf: gpd.GeoDataFrame | None,
    aoi: str,
) -> dict:
    """
    Proportion (0.0–1.0) of each elevation bin covered by the footprint.

    fp_gdf is None  → {} (bins stay NaN in coverage CSV).
    fp_gdf is empty → all bins = 0.0.
    """
    if fp_gdf is None:
        return {}
    if fp_gdf.empty:
        return _zero_bin_dict(bins_gdf, aoi)

    fp_union = fp_gdf.union_all()
    result = {}
    for _, br in bins_gdf.iterrows():
        geom = br["geometry"]
        if geom is None or geom.is_empty or geom.area == 0.0:
            continue
        cname = col_name(aoi, br["start_elv"], br["end_elv"], br["sector"])
        result[cname] = float(geom.intersection(fp_union).area / geom.area)
    return result


def compute_melt_bin_areas(
    bins_gdf: gpd.GeoDataFrame,
    melt_gdf: gpd.GeoDataFrame | None,
    aoi: str,
) -> dict:
    """
    Intersection area (m²) of melt polygons with each elevation bin.

    melt_gdf is None  → {} (bins stay NaN).
    melt_gdf is empty → all bins = 0.0.
    """
    if melt_gdf is None:
        return {}
    if melt_gdf.empty:
        return _zero_bin_dict(bins_gdf, aoi)

    melt_union = melt_gdf.union_all()
    result = {}
    for _, br in bins_gdf.iterrows():
        geom = br["geometry"]
        if geom is None or geom.is_empty or geom.area == 0.0:
            continue
        cname = col_name(aoi, br["start_elv"], br["end_elv"], br["sector"])
        result[cname] = float(geom.intersection(melt_union).area)
    return result


# ============================================================
# 6) PROJECTION HELPERS
# ============================================================

def project_melt_bins(
    fp_df: pd.DataFrame,
    melt_df: pd.DataFrame,
    bin_cols: list,
    bin_total_areas: dict,
    meta_cols: list,
) -> pd.DataFrame:
    """
    Bin-level projection: estimate melt in the unobserved part of each bin.

    For each row (block) and bin column:
        density    = observed_melt / footprint_area
        unobserved = max(0, total_bin_area - footprint_area)
        projected  = density × unobserved

    NaN footprint or zero footprint → projected = NaN.
    NaN melt (file missing) when footprint exists → treated as 0.

    Returns a DataFrame with the same meta columns as fp_df plus bin columns
    filled with projected values.
    """
    out = fp_df[meta_cols].copy()
    for col in bin_cols:
        out[col] = np.nan

    for col in bin_cols:
        if col not in bin_total_areas:
            continue  # bin has no valid geometry — leave as NaN
        total_area = bin_total_areas[col]
        fp_vals   = fp_df[col]   if col in fp_df.columns   else pd.Series(np.nan, index=fp_df.index)
        melt_vals = melt_df[col] if col in melt_df.columns else pd.Series(np.nan, index=melt_df.index)

        projected = np.full(len(out), np.nan)
        for i in range(len(out)):
            fp = fp_vals.iloc[i]
            if pd.isna(fp) or fp == 0.0:
                continue
            melt = melt_vals.iloc[i]
            if pd.isna(melt):
                melt = 0.0
            density       = melt / fp
            unobserved    = max(0.0, total_area - fp)
            projected[i]  = round(density * unobserved, 2)
        out[col] = projected

    return out



def aggregate_bins_to_aoi(df: pd.DataFrame, aoi_bin_cols: list) -> pd.Series:
    """
    Sum all bin columns for one AOI per row.
    Rows where every bin is NaN → NaN (no data for this block/AOI).
    """
    cols = [c for c in aoi_bin_cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    subset = df[cols]
    result = subset.sum(axis=1, skipna=True)
    all_nan = subset.isna().all(axis=1)
    result[all_nan] = np.nan
    return result


def project_melt_aoi(
    fp_aoi: pd.Series,
    melt_aoi: pd.Series,
    total_aoi_area: float,
) -> pd.Series:
    """
    AOI-level projection: per-row estimated melt in the unobserved portion.

    NaN or zero footprint → NaN (density unknown).
    NaN melt when footprint exists → treated as 0.
    """
    projected = pd.Series(np.nan, index=fp_aoi.index)
    for i in fp_aoi.index:
        fp = fp_aoi.loc[i]
        if pd.isna(fp) or fp == 0.0:
            continue
        melt = melt_aoi.loc[i]
        if pd.isna(melt):
            melt = 0.0
        density         = melt / fp
        unobserved      = max(0.0, total_aoi_area - fp)
        projected.loc[i] = round(density * unobserved, 2)
    return projected


def combine_areas_aoi(
    obs_aoi: pd.Series,
    proj_aoi: pd.Series,
) -> pd.Series:
    """
    Total AOI area = observed + projected.
    NaN projected → NaN total. NaN observed → treated as 0.
    """
    total = proj_aoi.copy()
    mask  = proj_aoi.notna()
    total[mask] = obs_aoi[mask].fillna(0.0) + proj_aoi[mask]
    return total.round(2)


# ============================================================
# 7) MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load elevation bin shapefiles (done once)
    # ----------------------------------------------------------
    debug_print("Loading elevation bin shapefiles …")
    bins_gdfs = {aoi: load_bins(AOI_BINS_SHPS[aoi], aoi) for aoi in AOIS}
    bin_cols  = all_bin_columns(bins_gdfs)
    bin_total_areas: dict = {}
    for aoi in AOIS:
        bin_total_areas.update(build_bin_total_areas(bins_gdfs[aoi], aoi))
    debug_print(f"  Total bin columns: {len(bin_cols)}")

    # ----------------------------------------------------------
    # Load AOI total areas (done once)
    # ----------------------------------------------------------
    debug_print("Loading AOI boundary shapefiles …")
    aoi_total_areas = {aoi: load_aoi_area(AOI_BOUNDARY_SHPS[aoi], aoi) for aoi in AOIS}

    # ----------------------------------------------------------
    # Discover block shapefiles
    # ----------------------------------------------------------
    debug_print("\nDiscovering block shapefiles …")
    blocks = discover_blocks(AOI_INPUT_DIRS)
    all_periods = union_block_periods(blocks)

    if not all_periods:
        raise SystemExit(
            "No block shapefiles found. Check INPUT_PTM_DIR and INPUT_OST_DIR."
        )
    debug_print(f"  Total unique block periods: {len(all_periods)}")

    # ----------------------------------------------------------
    # Initialise output DataFrames
    # meta_cols used as the index columns in all bin-level CSVs
    # ----------------------------------------------------------
    meta_cols = ["block_start", "block_end"]
    meta = pd.DataFrame(all_periods, columns=meta_cols)

    # Bin-level DataFrames (all NaN initially)
    fp_area_df    = meta.copy()   # footprint area (m²) per bin — used for projection
    cov_df        = meta.copy()   # bin coverage proportions (Method A selection)
    slush_area_df = meta.copy()   # observed slush area (m²) per bin
    lake_area_df  = meta.copy()   # observed lake area (m²) per bin

    for col in bin_cols:
        for df in (fp_area_df, cov_df, slush_area_df, lake_area_df):
            df[col] = np.nan

    # AOI summary columns for total_summary_raw
    summary_cols = []
    for aoi in AOIS:
        for suffix in (
            "footprint_area_m2", "footprint_proportion",
            "slush_area_m2",     "slush_proportion",
            "lake_area_m2",      "lake_proportion",
            "lake_volume_m3",    "lake_count",
        ):
            col = f"{aoi}_{suffix}"
            meta[col] = np.nan
            summary_cols.append(col)

    # ----------------------------------------------------------
    # Process each block period
    # ----------------------------------------------------------
    debug_print("\nProcessing block periods …")
    n_periods = len(all_periods)

    for idx, (bs, be) in enumerate(all_periods):
        label = f"{bs} – {be}"
        debug_print(f"\n[{idx + 1}/{n_periods}] Block {label}")

        for aoi in AOIS:
            layers    = blocks[aoi].get((bs, be), {})
            fp_path    = layers.get("footprint")
            slush_path = layers.get("slush")
            lake_path  = layers.get("lake")

            # Read shapefiles (None if path absent or file unreadable)
            fp_gdf    = read_block_shp(fp_path,    aoi, "footprint") if fp_path    else None
            slush_gdf = read_block_shp(slush_path, aoi, "slush")     if slush_path else None
            lake_gdf  = read_block_shp(lake_path,  aoi, "lake")      if lake_path  else None

            # ---- Bin-level: footprint ----
            fp_areas = compute_footprint_bin_areas(bins_gdfs[aoi], fp_gdf, aoi)
            fp_props = compute_footprint_bin_proportions(bins_gdfs[aoi], fp_gdf, aoi)
            for cname, val in fp_areas.items():
                fp_area_df.at[idx, cname] = round(val, 2)
            for cname, val in fp_props.items():
                cov_df.at[idx, cname] = round(val, 6)

            # ---- Bin-level: slush ----
            slush_areas = compute_melt_bin_areas(bins_gdfs[aoi], slush_gdf, aoi)
            for cname, val in slush_areas.items():
                slush_area_df.at[idx, cname] = round(val, 2)

            # ---- Bin-level: lake ----
            lake_areas = compute_melt_bin_areas(bins_gdfs[aoi], lake_gdf, aoi)
            for cname, val in lake_areas.items():
                lake_area_df.at[idx, cname] = round(val, 2)

            # ---- AOI summary (total_summary_raw equivalent) ----
            study_area = aoi_total_areas[aoi]

            # Footprint
            if fp_gdf is not None:
                fp_tot = float(fp_gdf.geometry.area.sum())
                meta.at[idx, f"{aoi}_footprint_area_m2"]    = round(fp_tot, 2)
                meta.at[idx, f"{aoi}_footprint_proportion"] = round(fp_tot / study_area, 6)

            # Slush
            if slush_gdf is not None:
                sl_tot = float(slush_gdf.geometry.area.sum())
                meta.at[idx, f"{aoi}_slush_area_m2"]    = round(sl_tot, 2)
                meta.at[idx, f"{aoi}_slush_proportion"] = round(sl_tot / study_area, 6)

            # Lake
            if lake_gdf is not None:
                lk_tot = float(lake_gdf.geometry.area.sum())
                meta.at[idx, f"{aoi}_lake_area_m2"]    = round(lk_tot, 2)
                meta.at[idx, f"{aoi}_lake_proportion"] = round(lk_tot / study_area, 6)

                # Approximate total lake volume: vol_m3 (area-weighted mean) × polygon area
                if "vol_m3" in lake_gdf.columns:
                    poly_areas = lake_gdf.geometry.area
                    vol_vals   = pd.to_numeric(lake_gdf["vol_m3"], errors="coerce")
                    valid      = vol_vals.notna() & (poly_areas > 0)
                    if valid.any():
                        approx_vol = float((vol_vals[valid] * poly_areas[valid]).sum())
                        meta.at[idx, f"{aoi}_lake_volume_m3"] = round(approx_vol, 2)

                # Lake count: sum of n_obs (contributing input polygons per dissolved poly)
                if "n_obs" in lake_gdf.columns:
                    meta.at[idx, f"{aoi}_lake_count"] = int(
                        pd.to_numeric(lake_gdf["n_obs"], errors="coerce").sum()
                    )

            debug_print(
                f"  [{aoi}]  fp_path={'yes' if fp_path else 'no':3s}  "
                f"slush={'yes' if slush_path else 'no':3s}  "
                f"lake={'yes' if lake_path else 'no':3s}"
            )

    # ----------------------------------------------------------
    # Bin-level projections
    # ----------------------------------------------------------
    debug_print("\nComputing bin-level projected melt areas …")
    proj_slush_bins = project_melt_bins(
        fp_area_df, slush_area_df, bin_cols, bin_total_areas, meta_cols
    )
    proj_lake_bins = project_melt_bins(
        fp_area_df, lake_area_df, bin_cols, bin_total_areas, meta_cols
    )

    # ----------------------------------------------------------
    # AOI-level projections
    # ----------------------------------------------------------
    debug_print("Computing AOI-level projected melt areas …")

    proj_slush_aoi_df  = meta[meta_cols].copy()
    proj_lake_aoi_df   = meta[meta_cols].copy()
    total_slush_aoi_df = meta[meta_cols].copy()
    total_lake_aoi_df  = meta[meta_cols].copy()

    for aoi in AOIS:
        aoi_bin_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]

        fp_aoi    = aggregate_bins_to_aoi(fp_area_df,    aoi_bin_cols)
        slush_aoi = aggregate_bins_to_aoi(slush_area_df, aoi_bin_cols)
        lake_aoi  = aggregate_bins_to_aoi(lake_area_df,  aoi_bin_cols)

        proj_slush = project_melt_aoi(fp_aoi, slush_aoi, aoi_total_areas[aoi])
        proj_lake  = project_melt_aoi(fp_aoi, lake_aoi,  aoi_total_areas[aoi])

        proj_slush_aoi_df[aoi]  = proj_slush
        proj_lake_aoi_df[aoi]   = proj_lake
        total_slush_aoi_df[aoi] = combine_areas_aoi(slush_aoi, proj_slush)
        total_lake_aoi_df[aoi]  = combine_areas_aoi(lake_aoi,  proj_lake)

    # ----------------------------------------------------------
    # Write all 10 output CSVs
    # ----------------------------------------------------------
    debug_print("\nWriting output CSVs …")
    bin_col_order = meta_cols + bin_cols

    def write_csv(df: pd.DataFrame, filename: str, col_order: list | None = None) -> None:
        path = OUTPUT_DIR / filename
        if col_order is not None:
            present = [c for c in col_order if c in df.columns]
            extra   = [c for c in df.columns if c not in col_order]
            df = df[present + extra]
        df.to_csv(path, index=False)
        debug_print(f"  Written: {path.name}  ({len(df)} rows)")

    # Method A – bin-level
    write_csv(cov_df,           "5_win_block_select_by_apect_elev_bin_coverage.csv",        bin_col_order)
    write_csv(slush_area_df,    "5_win_block_slush_area_by_aspect_elev_bins.csv",           bin_col_order)
    write_csv(lake_area_df,     "5_win_block_lake_area_by_aspect_elev_bins.csv",            bin_col_order)
    write_csv(proj_slush_bins,  "5_win_block_projected_slush_area_by_aspect_elev_bins.csv", bin_col_order)
    write_csv(proj_lake_bins,   "5_win_block_projected_lake_area_by_aspect_elev_bins.csv",  bin_col_order)

    # Method B – AOI-level
    summary_col_order = meta_cols + summary_cols
    write_csv(meta,              "5_win_block_total_summary_raw.csv",           summary_col_order)
    write_csv(proj_slush_aoi_df, "5_win_block_projected_slush_area_by_AOI.csv")
    write_csv(proj_lake_aoi_df,  "5_win_block_projected_lake_area_by_AOI.csv")
    write_csv(total_slush_aoi_df,"5_win_block_total_slush_area_by_AOI.csv")
    write_csv(total_lake_aoi_df, "5_win_block_total_lake_area_by_AOI.csv")

    debug_print("\nDone.")


if __name__ == "__main__":
    main()
