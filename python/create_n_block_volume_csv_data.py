# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:35:29 2026

@author: luur4790

create_n_block_volume_csv_data.py

NOTE:
This script remains available as the volume-only generator.
For the unified one-command area+volume workflow, run
create_n_block_csv_data.py.

Generates lake-volume CSV files for 5-window block shapefiles produced by
create_n_day_blocks.py.  This is the block-level volume analogue of
create_lake_volume_csv_data.py and extends the area outputs from
create_n_block_csv_data.py.

Volume method (identical to create_lake_volume_csv_data.py)
------------------------------------------------------------
Lake shapefiles must carry a  mean_dpt_m  attribute (average depth, m).
For each block period and bin the observed volume is:

    observed_volume = Σ_i( mean_dpt_m[i] × intersection_area(polygon_i, bin) )

Polygons with a NaN mean_dpt_m value are treated as zero depth.

Projection method (identical in structure to the area/window scripts)
----------------------------------------------------------------------
    volume_density   = observed_lake_volume / footprint_area   [m³/m²]
    unobserved_area  = max(0, total_area − footprint_area)     [m²]
    projected_volume = volume_density × unobserved_area        [m³]

Conditions:
  • footprint_area NaN or 0  → projected = NaN
  • footprint_area > 0       → apply formula; NaN volume treated as 0

Input block shapefile naming convention (inside AOI-specific folders):
    {YYYY-MM-DD}_{YYYY-MM-DD}_{layer}.shp
    e.g.  2013-05-01_2013-05-10_lake.shp

Outputs
-------
New volume CSVs (written to OUTPUT_DIR):
    5_win_block_lake_volume_by_aspect_elev_bins.csv
    5_win_block_projected_lake_volume_by_aspect_elev_bins.csv
    5_win_block_total_lake_volume_by_aspect_elev_bins.csv
    5_win_block_projected_lake_volume_by_AOI.csv
    5_win_block_total_lake_volume_by_AOI.csv

Extended (in-place update):
    5_win_block_total_summary_raw.csv
        — populates  OST_lake_volume_m3  and  PTM_lake_volume_m3  columns
          (already present but empty when created by create_n_block_csv_data.py)

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

# AOI boundary shapefiles
PTM_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp")
OST_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp")

# Name of the average-depth attribute in the lake shapefiles.
# Formula:  mean_dpt_m [m] × intersection_area [m²] = volume [m³]
VOL_COLUMN = "mean_dpt_m"

# Path to the existing total_summary_raw CSV written by create_n_block_csv_data.py.
# The OST_lake_volume_m3 / PTM_lake_volume_m3 columns will be populated here.
TOTAL_SUMMARY_RAW_CSV = OUTPUT_DIR / "5_win_block_total_summary_raw.csv"

PROJECTED_CRS = "EPSG:3995"

DEBUG = True


# ============================================================
# 2) CONSTANTS
# ============================================================

# Regex matching block shapefile names: YYYY-MM-DD_YYYY-MM-DD_{layer}.shp
# No AOI prefix — files live in AOI-specific folders.
_BLOCK_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(footprint|lake)\.shp$",
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
        warnings.warn(f"[{aoi}] Bin shapefile has no CRS; area calculations may be unreliable.")
    elif str(gdf.crs) != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    gdf = gdf.sort_values(["start_elv", "sector"]).reset_index(drop=True)
    n_main = (gdf["sector"] == "main").sum()
    n_east = (gdf["sector"] == "east").sum()
    n_west = (gdf["sector"] == "west").sum()
    debug_print(f"[{aoi}] Bins loaded: {n_main} main, {n_east} east, {n_west} west")
    return gdf


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


def load_aoi_area(shp_path: Path, aoi: str) -> float:
    """Return the total area (m²) of an AOI boundary shapefile."""
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] AOI shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"[{aoi}] AOI shapefile is empty: {shp_path}")
    if gdf.crs is None:
        warnings.warn(f"[{aoi}] AOI shapefile has no CRS; area calculations may be unreliable.")
    elif str(gdf.crs) != PROJECTED_CRS:
        gdf = gdf.to_crs(PROJECTED_CRS)
    area = float(gdf.geometry.union_all().area)
    debug_print(f"[{aoi}] AOI total area: {area:,.0f} m²")
    return area


# ============================================================
# 4) BLOCK DISCOVERY
# ============================================================

def discover_blocks(input_dirs: dict) -> dict:
    """
    Scan each AOI block folder for footprint and lake shapefiles matching
    the block naming pattern.

    Returns
    -------
    Nested dict:  {aoi: {(block_start_str, block_end_str): {layer_type: Path}}}
    Only 'footprint' and 'lake' layers are collected (slush is not needed here).
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
# 5) VOLUME COMPUTATION
# ============================================================

def compute_bin_volumes(
    bins_gdf: gpd.GeoDataFrame,
    lake_gdf: gpd.GeoDataFrame | None,
    aoi: str,
    vol_col: str = VOL_COLUMN,
) -> tuple[dict, bool]:
    """
    For every bin feature in *bins_gdf*, compute the total lake volume (m³)
    contributed by the lake polygons in *lake_gdf*.

    Each lake polygon contributes:
        vol_col [m] × intersection_area(polygon, bin) [m²] = volume [m³]

    vol_col is treated as average depth (m); multiplying by intersection
    area gives volume (m³).  This mirrors create_lake_volume_csv_data.py.

    Polygons with a NaN vol_col value are treated as zero depth.
    Bins with null/empty/zero geometry are omitted (their columns stay NaN).

    Parameters
    ----------
    bins_gdf  : aspect-split elevation-bin GeoDataFrame (already reprojected)
    lake_gdf  : lake GeoDataFrame for this block / AOI, or None if absent
    aoi       : AOI label used for column naming and log messages
    vol_col   : name of the average-depth attribute in the lake shapefile

    Returns
    -------
    (volumes_dict, col_missing)
      volumes_dict : col_name -> float (m³), empty dict on failure / None input
      col_missing  : True if the shapefile was read but lacked vol_col
    """
    # No shapefile for this block/AOI — bins stay NaN (no footprint coverage)
    if lake_gdf is None:
        return {}, False

    # Empty shapefile — footprint existed but no lakes observed; all bins = 0
    if lake_gdf.empty:
        return (
            {
                col_name(aoi, r["start_elv"], r["end_elv"], r["sector"]): 0.0
                for _, r in bins_gdf.iterrows()
                if r["geometry"] is not None and not r["geometry"].is_empty
            },
            False,
        )

    if vol_col not in lake_gdf.columns:
        warnings.warn(
            f"[{aoi}] VOL_COLUMN='{vol_col}' not found in lake shapefile. "
            f"Available columns: {[c for c in lake_gdf.columns if c != 'geometry']}"
        )
        return {}, True  # caller tallies and reports

    # Pre-extract arrays for fast per-polygon access
    vol_vals    = pd.to_numeric(lake_gdf[vol_col], errors="coerce").fillna(0.0).to_numpy()
    geoms       = lake_gdf.geometry.to_numpy()
    lake_sindex = lake_gdf.sindex

    result = {}
    for _, bin_row in bins_gdf.iterrows():
        bin_geom = bin_row["geometry"]
        if bin_geom is None or bin_geom.is_empty or bin_geom.area == 0.0:
            continue

        cname      = col_name(aoi, bin_row["start_elv"], bin_row["end_elv"], bin_row["sector"])
        candidates = list(lake_sindex.intersection(bin_geom.bounds))

        total_volume = 0.0
        for j in candidates:
            lake_geom = geoms[j]
            if lake_geom is None or lake_geom.is_empty:
                continue
            inter = bin_geom.intersection(lake_geom)
            if not inter.is_empty:
                total_volume += vol_vals[j] * inter.area

        result[cname] = round(total_volume, 2)

    return result, False


def compute_aoi_total_volume(lake_gdf: gpd.GeoDataFrame | None, aoi: str, vol_col: str = VOL_COLUMN) -> float | None:
    """
    Compute total observed lake volume (m³) across the full AOI footprint:
        Σ_i( vol_col[i] × polygon_area[i] )

    Returns None when lake_gdf is None (no shapefile) or the vol_col is absent.
    Returns 0.0 when lake_gdf is present but empty (no lakes observed).
    """
    if lake_gdf is None:
        return None
    if lake_gdf.empty:
        return 0.0
    if vol_col not in lake_gdf.columns:
        return None
    poly_areas = lake_gdf.geometry.area
    vol_vals   = pd.to_numeric(lake_gdf[vol_col], errors="coerce")
    valid      = vol_vals.notna() & (poly_areas > 0)
    if not valid.any():
        return 0.0
    return round(float((vol_vals[valid] * poly_areas[valid]).sum()), 2)


# ============================================================
# 6) PROJECTION & COMBINATION HELPERS
# ============================================================

def aggregate_bins_to_aoi(df: pd.DataFrame, aoi_bin_cols: list) -> pd.Series:
    """
    Sum all bin columns for one AOI per row.
    Rows where every bin is NaN → NaN (no data for this block/AOI).
    """
    cols = [c for c in aoi_bin_cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    subset  = df[cols]
    result  = subset.sum(axis=1, skipna=True)
    all_nan = subset.isna().all(axis=1)
    result[all_nan] = np.nan
    return result


def project_volume(
    fp_area_df: pd.DataFrame,
    obs_vol_df: pd.DataFrame,
    bin_cols: list,
    bin_total_areas: dict,
    meta_cols: list,
) -> pd.DataFrame:
    """
    Bin-level volume projection: estimate volume in the unobserved part of each bin.

    For each row (block) and bin column:
        density    = observed_volume / footprint_area   [m³/m²]
        unobserved = max(0, total_bin_area − footprint_area)
        projected  = density × unobserved               [m³]

    NaN or zero footprint → projected = NaN.
    NaN volume when footprint exists → treated as 0.

    Returns a DataFrame with meta columns plus bin columns filled with
    projected values.
    """
    out = fp_area_df[meta_cols].copy()
    for col in bin_cols:
        out[col] = np.nan

    for col in bin_cols:
        if col not in bin_total_areas:
            continue  # bin has no valid geometry — leave as NaN
        total_area = bin_total_areas[col]
        fp_vals  = fp_area_df[col]  if col in fp_area_df.columns  else pd.Series(np.nan, index=fp_area_df.index)
        vol_vals = obs_vol_df[col]  if col in obs_vol_df.columns  else pd.Series(np.nan, index=obs_vol_df.index)

        projected = np.full(len(out), np.nan)
        for i in range(len(out)):
            fp = fp_vals.iloc[i]
            if pd.isna(fp) or fp == 0.0:
                continue
            vol = vol_vals.iloc[i]
            if pd.isna(vol):
                vol = 0.0
            density      = vol / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)
        out[col] = projected

    return out


def combine_volumes(
    obs_df: pd.DataFrame,
    proj_df: pd.DataFrame,
    bin_cols: list,
    meta_cols: list,
) -> pd.DataFrame:
    """
    Total (observed + projected) volume at bin level.

    For each row and bin:
      • projected NaN → total NaN  (density unknown; cannot estimate total)
      • projected not NaN → total = observed (NaN → 0) + projected
    """
    out = proj_df[meta_cols].copy()
    for col in bin_cols:
        proj_vals = proj_df[col] if col in proj_df.columns else pd.Series(np.nan, index=proj_df.index)
        obs_vals  = obs_df[col]  if col in obs_df.columns  else pd.Series(np.nan, index=obs_df.index)

        combined       = proj_vals.copy()
        mask           = proj_vals.notna()
        combined[mask] = obs_vals[mask].fillna(0.0) + proj_vals[mask]
        out[col]       = combined.round(2)

    return out


def project_volume_aoi(
    fp_area_df: pd.DataFrame,
    obs_vol_df: pd.DataFrame,
    bin_cols: list,
    aoi_total_areas: dict,
    meta_cols: list,
) -> pd.DataFrame:
    """
    AOI-level volume projection.

    For each row (block) and AOI:
        fp_aoi    = sum of footprint areas across all bins in the AOI
        vol_aoi   = sum of observed volumes across all bins
        density   = vol_aoi / fp_aoi
        unobserved = max(0, total_AOI_area − fp_aoi)
        projected = density × unobserved

    NaN or zero fp_aoi → projected = NaN.
    NaN vol_aoi when fp_aoi exists → treated as 0.
    """
    out = fp_area_df[meta_cols].copy()

    for aoi in AOIS:
        aoi_bin_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        total_area   = aoi_total_areas[aoi]
        fp_aoi       = aggregate_bins_to_aoi(fp_area_df,  aoi_bin_cols)
        vol_aoi      = aggregate_bins_to_aoi(obs_vol_df,  aoi_bin_cols)

        projected = np.full(len(out), np.nan)
        for i in range(len(out)):
            fp = fp_aoi.iloc[i]
            if pd.isna(fp) or fp == 0.0:
                continue
            vol = vol_aoi.iloc[i]
            if pd.isna(vol):
                vol = 0.0
            density      = vol / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)
        out[aoi] = projected

    return out


def combine_volumes_aoi(
    obs_vol_df: pd.DataFrame,
    proj_aoi_df: pd.DataFrame,
    bin_cols: list,
) -> pd.DataFrame:
    """
    Total (observed + projected) volume at AOI level.

    For each row and AOI:
      • projected NaN → total NaN
      • projected not NaN → total = observed_AOI (NaN → 0) + projected
    """
    out = proj_aoi_df.copy()

    for aoi in AOIS:
        aoi_bin_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        obs_aoi      = aggregate_bins_to_aoi(obs_vol_df, aoi_bin_cols)
        proj_col     = proj_aoi_df[aoi]

        combined       = proj_col.copy()
        mask           = proj_col.notna()
        combined[mask] = obs_aoi[mask].fillna(0.0) + proj_col[mask]
        out[aoi]       = combined.round(2)

    return out


# ============================================================
# 7) MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load elevation bin shapefiles (done once)
    # ----------------------------------------------------------
    debug_print("Loading elevation bin shapefiles …")
    bins_gdfs     = {aoi: load_bins(AOI_BINS_SHPS[aoi], aoi) for aoi in AOIS}
    bin_cols      = all_bin_columns(bins_gdfs)
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
    blocks     = discover_blocks(AOI_INPUT_DIRS)
    all_periods = union_block_periods(blocks)

    if not all_periods:
        raise SystemExit(
            "No block shapefiles found. Check INPUT_PTM_DIR and INPUT_OST_DIR."
        )
    debug_print(f"  Total unique block periods: {len(all_periods)}")

    # ----------------------------------------------------------
    # Startup diagnostic: inspect the first available lake file
    # to confirm VOL_COLUMN exists and list all attributes.
    # ----------------------------------------------------------
    debug_print(f"\nVolume column setting: VOL_COLUMN = '{VOL_COLUMN}'")
    _checked = False
    for _aoi in AOIS:
        for (_bs, _be), _layers in blocks[_aoi].items():
            if "lake" not in _layers:
                continue
            try:
                _sample   = gpd.read_file(_layers["lake"])
                _non_geom = [c for c in _sample.columns if c != "geometry"]
                debug_print(
                    f"  First lake file ({_aoi}): {_layers['lake'].name}\n"
                    f"    Attributes found: {_non_geom}"
                )
                if VOL_COLUMN not in _sample.columns:
                    print(
                        f"\n*** WARNING: VOL_COLUMN='{VOL_COLUMN}' is NOT present in the lake "
                        f"shapefiles.\n"
                        f"    Available columns: {_non_geom}\n"
                        f"    Set VOL_COLUMN at the top of this script to the correct field name.\n"
                        f"    All observed volumes will remain NaN until this is fixed. ***\n"
                    )
                else:
                    debug_print(
                        f"    '{VOL_COLUMN}' column found — proceeding with volume computation."
                    )
                _checked = True
            except Exception:
                pass
            if _checked:
                break
        if _checked:
            break

    # ----------------------------------------------------------
    # Initialise output DataFrames
    # ----------------------------------------------------------
    meta_cols = ["block_start", "block_end"]
    meta      = pd.DataFrame(all_periods, columns=meta_cols)

    # Footprint area per bin (m²) — computed from block shapefiles for projection
    fp_area_df  = meta.copy()
    # Observed lake volume per bin (m³)
    obs_vol_df  = meta.copy()

    for col in bin_cols:
        fp_area_df[col] = np.nan
        obs_vol_df[col] = np.nan

    # ----------------------------------------------------------
    # Process each block period
    # ----------------------------------------------------------
    debug_print("\nProcessing block periods …")
    n_periods = len(all_periods)
    missing_col_counts = {aoi: 0 for aoi in AOIS}

    for idx, (bs, be) in enumerate(all_periods):
        label = f"{bs} – {be}"
        debug_print(f"\n[{idx + 1}/{n_periods}] Block {label}")

        for aoi in AOIS:
            layers    = blocks[aoi].get((bs, be), {})
            fp_path   = layers.get("footprint")
            lake_path = layers.get("lake")

            # Read shapefiles (None if path absent or unreadable)
            fp_gdf   = read_block_shp(fp_path,   aoi, "footprint") if fp_path   else None
            lake_gdf = read_block_shp(lake_path,  aoi, "lake")      if lake_path else None

            # ---- Footprint bin areas (needed for projection) ----
            if fp_gdf is not None and not fp_gdf.empty:
                fp_union = fp_gdf.union_all()
                for _, br in bins_gdfs[aoi].iterrows():
                    bin_geom = br["geometry"]
                    if bin_geom is None or bin_geom.is_empty or bin_geom.area == 0.0:
                        continue
                    cname = col_name(aoi, br["start_elv"], br["end_elv"], br["sector"])
                    fp_area_df.at[idx, cname] = round(float(bin_geom.intersection(fp_union).area), 2)
            elif fp_gdf is not None and fp_gdf.empty:
                # File present but empty → set footprint area to 0 for all bins
                for _, br in bins_gdfs[aoi].iterrows():
                    bin_geom = br["geometry"]
                    if bin_geom is None or bin_geom.is_empty:
                        continue
                    cname = col_name(aoi, br["start_elv"], br["end_elv"], br["sector"])
                    fp_area_df.at[idx, cname] = 0.0

            # ---- Observed lake volumes per bin ----
            volumes, col_missing = compute_bin_volumes(bins_gdfs[aoi], lake_gdf, aoi)
            if col_missing:
                missing_col_counts[aoi] += 1
            else:
                for cname, val in volumes.items():
                    obs_vol_df.at[idx, cname] = val

            n_nonzero = sum(1 for v in volumes.values() if v > 0)
            debug_print(
                f"  [{aoi}]  fp={'yes' if fp_path else 'no ':3s}  "
                f"lake={'yes' if lake_path else 'no ':3s}  "
                f"bins_with_vol>0={n_nonzero}"
            )

    for aoi in AOIS:
        if missing_col_counts[aoi] > 0:
            print(
                f"  [{aoi}] {missing_col_counts[aoi]} lake file(s) missing "
                f"'{VOL_COLUMN}' — those blocks left as NaN. "
                f"Set VOL_COLUMN to the correct depth field name."
            )

    # ----------------------------------------------------------
    # Bin-level projections
    # ----------------------------------------------------------
    debug_print("\nProjecting lake volumes into unobserved bin zones …")
    proj_vol_bins_df  = project_volume(fp_area_df, obs_vol_df, bin_cols, bin_total_areas, meta_cols)

    debug_print("Combining observed and projected bin volumes …")
    total_vol_bins_df = combine_volumes(obs_vol_df, proj_vol_bins_df, bin_cols, meta_cols)

    # ----------------------------------------------------------
    # AOI-level projections
    # ----------------------------------------------------------
    debug_print("\nProjecting lake volumes into unobserved AOI zones …")
    proj_vol_aoi_df  = project_volume_aoi(fp_area_df, obs_vol_df, bin_cols, aoi_total_areas, meta_cols)

    debug_print("Combining observed and projected AOI volumes …")
    total_vol_aoi_df = combine_volumes_aoi(obs_vol_df, proj_vol_aoi_df, bin_cols)

    # ----------------------------------------------------------
    # Extend total_summary_raw with lake volume columns
    # ----------------------------------------------------------
    debug_print(f"\nUpdating {TOTAL_SUMMARY_RAW_CSV.name} with lake volume columns …")
    if TOTAL_SUMMARY_RAW_CSV.exists():
        summary_df = pd.read_csv(TOTAL_SUMMARY_RAW_CSV)
        for idx, (bs, be) in enumerate(all_periods):
            # Locate the matching row in summary_df by block dates
            mask = (summary_df["block_start"] == bs) & (summary_df["block_end"] == be)
            if not mask.any():
                debug_print(f"  No matching row in summary CSV for {bs}–{be}; skipping.")
                continue
            row_idx = summary_df.index[mask][0]

            for aoi in AOIS:
                layers   = blocks[aoi].get((bs, be), {})
                lake_path = layers.get("lake")
                lake_gdf  = read_block_shp(lake_path, aoi, "lake") if lake_path else None
                vol = compute_aoi_total_volume(lake_gdf, aoi)
                if vol is not None:
                    summary_df.at[row_idx, f"{aoi}_lake_volume_m3"] = vol

        summary_df.to_csv(TOTAL_SUMMARY_RAW_CSV, index=False)
        debug_print(f"  Updated: {TOTAL_SUMMARY_RAW_CSV.name}")
    else:
        warnings.warn(
            f"total_summary_raw CSV not found at {TOTAL_SUMMARY_RAW_CSV}. "
            "Run create_n_block_csv_data.py first to generate it."
        )

    # ----------------------------------------------------------
    # Write the 5 volume CSVs
    # ----------------------------------------------------------
    debug_print("\nWriting volume CSVs …")
    bin_col_order = meta_cols + bin_cols

    def write_csv(df: pd.DataFrame, filename: str, col_order: list | None = None) -> None:
        path = OUTPUT_DIR / filename
        if col_order is not None:
            present = [c for c in col_order if c in df.columns]
            extra   = [c for c in df.columns  if c not in col_order]
            df = df[present + extra]
        df.to_csv(path, index=False)
        debug_print(f"  Written: {path.name}  ({len(df)} rows)")

    write_csv(obs_vol_df,        "5_win_block_lake_volume_by_aspect_elev_bins.csv",           bin_col_order)
    write_csv(proj_vol_bins_df,  "5_win_block_projected_lake_volume_by_aspect_elev_bins.csv", bin_col_order)
    write_csv(total_vol_bins_df, "5_win_block_total_lake_volume_by_aspect_elev_bins.csv",     bin_col_order)
    write_csv(proj_vol_aoi_df,   "5_win_block_projected_lake_volume_by_AOI.csv")
    write_csv(total_vol_aoi_df,  "5_win_block_total_lake_volume_by_AOI.csv")

    # ----------------------------------------------------------
    # QA summary
    # ----------------------------------------------------------
    debug_print("\n── QA Summary ──────────────────────────────────────────")
    for aoi in AOIS:
        aoi_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        bins_gdf = bins_gdfs[aoi]
        n_main   = (bins_gdf["sector"] == "main").sum()
        n_east   = (bins_gdf["sector"] == "east").sum()
        n_west   = (bins_gdf["sector"] == "west").sum()
        bin_desc = f"{n_main} main + {n_east} east + {n_west} west bins"

        for layer_label, df in [
            ("observed volume  (bins)",  obs_vol_df),
            ("projected volume (bins)",  proj_vol_bins_df),
            ("total volume     (bins)",  total_vol_bins_df),
        ]:
            present   = [c for c in aoi_cols if c in df.columns]
            n_blank   = df[present].isna().all(axis=1).sum()
            n_covered = (df[present] > 0).any(axis=1).sum()
            n_zero    = (
                df[present].notna().any(axis=1) & ~(df[present] > 0).any(axis=1)
            ).sum()
            debug_print(
                f"  {aoi} {layer_label} ({bin_desc}): "
                f"{n_covered} block(s) ≥1 bin > 0  |  "
                f"{n_zero} block(s) all-zero  |  "
                f"{n_blank} block(s) no data"
            )

        for layer_label, df in [
            ("projected volume (AOI)", proj_vol_aoi_df),
            ("total volume     (AOI)", total_vol_aoi_df),
        ]:
            if aoi not in df.columns:
                continue
            n_blank   = df[aoi].isna().sum()
            n_covered = (df[aoi] > 0).sum()
            n_zero    = (df[aoi].notna() & ~(df[aoi] > 0)).sum()
            debug_print(
                f"  {aoi} {layer_label}: "
                f"{n_covered} block(s) > 0  |  "
                f"{n_zero} block(s) zero  |  "
                f"{n_blank} block(s) no data"
            )

    debug_print("\nDone.")


if __name__ == "__main__":
    main()
