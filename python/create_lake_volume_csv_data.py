# -*- coding: utf-8 -*-
"""
create_lake_volume_csv_data.py

Computes observed lake volume (m³) per aspect-split elevation bin for every
melt-season two-day window, then projects that volume into the unobserved
portions of each bin and each full AOI.

This is the volume analogue of combining:
  - melt_area_by_aspect_elev_bins.py        (observed values from shapefiles)
  - project_melt_area_by_aspect_elev_bins.py (bin-level projection)
  - project_melt_area_by_AOI.py              (AOI-level projection)

and produces all input CSVs required by a volume-visualisation script
equivalent to compare_select_methods_by_visualize_melt_area.py.

Observed volume per bin
-----------------------
Lake shapefiles must carry a  vol_m3  attribute (average depth, m³/m²).
For each window and bin the observed volume is:

    observed_volume = sum_i( vol_m3[i] × intersection_area(polygon_i, bin) )

where the sum runs over all lake polygons that intersect the bin.
Polygons with a NaN vol_m3 value are treated as 0.

Projection method (identical in structure to the area scripts)
--------------------------------------------------------------
    volume_density   = observed_lake_volume / footprint_area   [m³/m²]
    unobserved_area  = max(0, total_area − footprint_area)     [m²]
    projected_volume = volume_density × unobserved_area        [m³]

Conditions (same as area scripts):
  • footprint_area NaN or 0  → projected = NaN  (density cannot be estimated)
  • footprint_area > 0       → apply formula above
    – observed_volume NaN (missing shapefile when footprint exists) → treated as 0

Inputs
------
  all_windows_template.csv                 — window row structure
  footprint_area_by_aspect_elev_bins.csv   — footprint area (m²) per bin/window
  {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_lake.shp — lake polygons per window (need vol_m3)
  PTM_aspect_elevation_bins.shp / OST_aspect_elevation_bins.shp
  PTM_AOI_1500m.shp / OST_AOI_1500m.shp

Outputs (saved to OUTPUT_DIR; names mirror the area CSVs with "area" → "volume")
-------
  lake_volume_by_aspect_elev_bins.csv           — observed volume per bin (m³)
  projected_lake_volume_by_aspect_elev_bins.csv — projected volume, unobserved zone (m³)
  total_lake_volume_by_aspect_elev_bins.csv     — total (obs + proj) per bin (m³)
  projected_lake_volume_by_AOI.csv              — projected volume per AOI (m³)
  total_lake_volume_by_AOI.csv                  — total (obs + proj) per AOI (m³)
"""

import re
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

PTM_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp")
OST_AOI_SHP = Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp")

# Folders containing per-window lake shapefiles.
# Files must be named:  {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_lake.shp
# and carry a  vol_m3  attribute (average depth, m³/m²).
PTM_FOLDER = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_PTM_files")
OST_FOLDER = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_OST_files")

TEMPLATE_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_templates\all_windows_template.csv")
FOOTPRINT_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\footprint_area_by_aspect_elev_bins.csv")

OUTPUT_DIR = Path(r"Q:\ThesisData\data\csv_files\csv_outputs")

# Name of the mean-depth attribute in the lake shapefiles.
# The script expects this column to hold average lake depth (m) so that
#   mean_depth × intersection_area = volume (m³) per bin.
# Your shapefiles use "mean_dpt_m".  Do NOT use the "volume_m3" field
# here — that would be dimensionally incorrect (m³ × m² ≠ m³).
VOL_COLUMN = "mean_dpt_m"

OBS_VOL_BINS_CSV   = OUTPUT_DIR / "lake_volume_by_aspect_elev_bins.csv"
PROJ_VOL_BINS_CSV  = OUTPUT_DIR / "projected_lake_volume_by_aspect_elev_bins.csv"
TOTAL_VOL_BINS_CSV = OUTPUT_DIR / "total_lake_volume_by_aspect_elev_bins.csv"
PROJ_VOL_AOI_CSV   = OUTPUT_DIR / "projected_lake_volume_by_AOI.csv"
TOTAL_VOL_AOI_CSV  = OUTPUT_DIR / "total_lake_volume_by_AOI.csv"

# Projected CRS used for all spatial calculations (must match the source CSVs)
PROJECTED_CRS = "EPSG:3995"

AOIS = ["OST", "PTM"]

AOI_LAKE_FOLDERS = {
    "OST": OST_FOLDER,
    "PTM": PTM_FOLDER,
}

AOI_BINS_SHPS = {
    "OST": OST_BINS_SHP,
    "PTM": PTM_BINS_SHP,
}

AOI_BOUNDARY_SHPS = {
    "OST": OST_AOI_SHP,
    "PTM": PTM_AOI_SHP,
}

DEBUG = True


# ============================================================
# 2) LOW-LEVEL HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


# Filename pattern:  {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_lake.shp
_LAKE_RE = re.compile(
    r"^(?P<aoi>[A-Z]+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"lake\.shp$",
    re.IGNORECASE,
)


def col_name(aoi: str, start_elv, end_elv, sector: str) -> str:
    """
    Return the output column name for an aspect-split elevation bin.

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


def all_bin_columns(ost_bins: gpd.GeoDataFrame, ptm_bins: gpd.GeoDataFrame) -> list:
    """Return ordered column names: OST bins first, then PTM."""
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
        debug_print(f"[{aoi}] Reprojecting bins from {gdf.crs} to {PROJECTED_CRS}")
        gdf = gdf.to_crs(PROJECTED_CRS)
    gdf = gdf.sort_values(["start_elv", "sector"]).reset_index(drop=True)
    n_main = (gdf["sector"] == "main").sum()
    n_east = (gdf["sector"] == "east").sum()
    n_west = (gdf["sector"] == "west").sum()
    debug_print(
        f"[{aoi}] Loaded {len(gdf)} bin feature(s): "
        f"{n_main} main, {n_east} east, {n_west} west"
    )
    return gdf


def build_bin_total_areas(bins_gdf: gpd.GeoDataFrame, aoi: str) -> dict:
    """
    Return a dict mapping col_name -> total bin area (m²) from shapefile geometry.
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


def load_aoi_area(shp_path: Path, aoi: str) -> float:
    """
    Read an AOI shapefile, reproject to PROJECTED_CRS, and return the total
    area (m²) of all features unioned together.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] AOI shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"[{aoi}] AOI shapefile is empty: {shp_path}")
    if gdf.crs is None:
        warnings.warn(f"[{aoi}] AOI shapefile has no CRS; area calculations may be unreliable.")
    elif str(gdf.crs) != PROJECTED_CRS:
        debug_print(f"[{aoi}] Reprojecting AOI from {gdf.crs} to {PROJECTED_CRS}")
        gdf = gdf.to_crs(PROJECTED_CRS)
    total_area = float(gdf.geometry.union_all().area)
    debug_print(f"[{aoi}] AOI total area: {total_area:,.0f} m²")
    return total_area


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce win_start / win_end to ISO YYYY-MM-DD strings."""
    out = df.copy()
    for col in ("win_start", "win_end"):
        if col not in out.columns:
            raise KeyError(f"CSV is missing required column: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


# ============================================================
# 3) LAKE FILE DISCOVERY & OBSERVED-VOLUME COMPUTATION
# ============================================================

def scan_lake_files(folder: Path, aoi_prefix: str) -> dict:
    """
    Scan *folder* for lake shapefiles matching the expected naming pattern.

    Returns
    -------
    dict mapping (win_start_str, win_end_str) -> Path
    where date strings are ISO format YYYY-MM-DD.
    """
    result = {}
    prefix_upper = aoi_prefix.upper()

    if not folder.exists():
        raise FileNotFoundError(f"[{aoi_prefix}] Lake folder not found: {folder}")

    for shp_path in folder.glob("*.shp"):
        m = _LAKE_RE.match(shp_path.name)
        if m and m.group("aoi").upper() == prefix_upper:
            key = (m.group("start"), m.group("end"))
            if key in result:
                warnings.warn(
                    f"[{aoi_prefix}] Duplicate lake file for {key[0]}–{key[1]}; "
                    f"keeping first: {result[key].name}"
                )
            else:
                result[key] = shp_path

    debug_print(f"[{aoi_prefix}] Lake files found: {len(result)}")
    return result


def compute_bin_volumes(
    bins_gdf: gpd.GeoDataFrame,
    melt_path: Path,
    aoi: str,
    vol_col: str = "vol_m3",
) -> tuple[dict, bool]:
    """
    For every bin feature in *bins_gdf*, compute the total lake volume (m³)
    contributed by the lake polygons in *melt_path*.

    Each lake polygon contributes:
        vol_col_attribute × intersection_area(polygon, bin)

    vol_col is treated as average depth (m), so multiplying by the
    intersection area (m²) gives volume (m³).  This mirrors the approach
    used in create_n_block_csv_data.py for AOI-level volume.

    Polygons with a NaN vol_col value are treated as zero depth.
    Bins with null/empty/zero geometry are omitted (their columns stay NaN).

    Parameters
    ----------
    bins_gdf  : aspect-split elevation-bin GeoDataFrame (already reprojected)
    melt_path : path to the lake shapefile for this window / AOI
    aoi       : AOI label used for column naming and log messages
    vol_col   : name of the depth/volume attribute in the lake shapefile

    Returns
    -------
    (volumes_dict, col_missing)
      volumes_dict : col_name -> float (m³), or empty dict on failure
      col_missing  : True if the shapefile was read successfully but lacked vol_col
    """
    try:
        lake_gdf = gpd.read_file(melt_path)
    except Exception as exc:
        warnings.warn(f"[{aoi}] Cannot read lake shapefile {melt_path.name}: {exc}")
        return {}, False

    # Empty shapefile → footprint existed but no lakes observed; set all bins to 0
    if lake_gdf.empty:
        debug_print(f"  [{aoi}] Empty lake shapefile: {melt_path.name} — all bins set to 0.0")
        return (
            {
                col_name(aoi, r["start_elv"], r["end_elv"], r["sector"]): 0.0
                for _, r in bins_gdf.iterrows()
                if r["geometry"] is not None and not r["geometry"].is_empty
            },
            False,
        )

    if vol_col not in lake_gdf.columns:
        return {}, True   # caller tallies and reports

    if lake_gdf.crs is None:
        warnings.warn(f"[{aoi}] {melt_path.name} has no CRS; reprojection skipped.")
    elif str(lake_gdf.crs) != PROJECTED_CRS:
        lake_gdf = lake_gdf.to_crs(PROJECTED_CRS)

    # Pre-extract arrays for fast per-polygon access
    vol_vals  = pd.to_numeric(lake_gdf[vol_col], errors="coerce").fillna(0.0).to_numpy()
    geoms     = lake_gdf.geometry.to_numpy()

    # Build a spatial index over lake polygons for fast candidate lookup
    lake_sindex = lake_gdf.sindex

    result = {}
    for _, bin_row in bins_gdf.iterrows():
        bin_geom = bin_row["geometry"]
        if bin_geom is None or bin_geom.is_empty or bin_geom.area == 0.0:
            continue

        cname = col_name(aoi, bin_row["start_elv"], bin_row["end_elv"], bin_row["sector"])

        # Query spatial index for candidate polygons whose bounding box overlaps
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


# ============================================================
# 4) PROJECTION & COMBINATION HELPERS
# ============================================================

def aggregate_bins_to_aoi(df: pd.DataFrame, aoi: str) -> pd.Series:
    """
    Sum all columns starting with '{aoi}_' across the DataFrame per row.

    Per-row behaviour (mirrors project_melt_area_by_AOI.py):
      • All bins NaN  → NaN  (no data for this window / AOI)
      • At least one bin non-NaN → sum of non-NaN bins (NaN bins treated as 0)
    """
    cols = [c for c in df.columns if c.startswith(f"{aoi}_")]
    if not cols:
        debug_print(f"  [{aoi}] No bin columns found with prefix '{aoi}_' — returning NaN series")
        return pd.Series(np.nan, index=df.index)
    subset  = df[cols]
    result  = subset.sum(axis=1, skipna=True)
    all_nan = subset.isna().all(axis=1)
    result[all_nan] = np.nan
    return result


def project_volume(
    footprint_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    bin_cols: list,
    bin_total_areas: dict,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the projected-volume DataFrame at bin level.

    For each row (window) and each bin column:
        volume_density   = observed_volume / footprint_area   [m³/m²]
        unobserved_area  = max(0, total_bin_area − footprint_area)
        projected_volume = volume_density × unobserved_area   [m³]

    Conditions (identical to project_melt in project_melt_area_by_aspect_elev_bins.py):
      • footprint_area NaN or 0  → projected = NaN
      • footprint_area > 0       → apply formula; NaN volume treated as 0

    Returns a copy of footprint_df with bin columns replaced by projected values.
    """
    out_df = footprint_df.copy()

    for col in bin_cols:
        if col not in bin_total_areas:
            out_df[col] = np.nan
            continue

        total_area = bin_total_areas[col]
        fp_vals  = footprint_df[col] if col in footprint_df.columns \
                   else pd.Series(np.nan, index=footprint_df.index)
        vol_vals = volume_df[col]    if col in volume_df.columns    \
                   else pd.Series(np.nan, index=volume_df.index)

        projected = np.full(len(out_df), np.nan)
        for i in range(len(out_df)):
            fp = fp_vals.iloc[i]
            if pd.isna(fp) or fp == 0.0:
                continue
            vol = vol_vals.iloc[i]
            if pd.isna(vol):
                vol = 0.0
            density      = vol / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)

        out_df[col] = projected

    n_filled = out_df[bin_cols].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label} (bins): {n_filled}/{len(out_df)} window(s) "
        "with at least one bin projected"
    )
    return out_df


def combine_volumes(
    observed_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    bin_cols: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the total (observed + projected) volume DataFrame at bin level.

    For each row and bin column:
      • projected NaN → total NaN   (density unknown; cannot estimate total)
      • projected not NaN → total = observed (NaN → 0) + projected

    Mirrors combine_areas in project_melt_area_by_aspect_elev_bins.py.
    """
    out_df = projected_df.copy()

    for col in bin_cols:
        proj_vals = projected_df[col] if col in projected_df.columns \
                    else pd.Series(np.nan, index=projected_df.index)
        obs_vals  = observed_df[col]  if col in observed_df.columns  \
                    else pd.Series(np.nan, index=observed_df.index)

        combined      = proj_vals.copy()
        mask          = proj_vals.notna()
        combined[mask] = obs_vals[mask].fillna(0.0) + proj_vals[mask]
        out_df[col]   = combined.round(2)

    n_filled = out_df[bin_cols].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label} (bins): {n_filled}/{len(out_df)} window(s) "
        "with at least one bin combined"
    )
    return out_df


def project_volume_aoi(
    footprint_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    aoi_names: list,
    total_aoi_areas: dict,
    meta_cols: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the projected-volume DataFrame at AOI level.

    For each row (window) and AOI:
        fp_aoi           = sum of footprint areas across all bins in the AOI
        observed_vol_aoi = sum of observed volumes across all bins
        volume_density   = observed_vol_aoi / fp_aoi
        unobserved_area  = max(0, total_AOI_area − fp_aoi)
        projected_volume = volume_density × unobserved_area

    Mirrors project_melt_aoi in project_melt_area_by_AOI.py.
    """
    out_df = footprint_df[meta_cols].copy()

    for aoi in aoi_names:
        total_area = total_aoi_areas[aoi]
        fp_aoi  = aggregate_bins_to_aoi(footprint_df, aoi)
        vol_aoi = aggregate_bins_to_aoi(volume_df,    aoi)

        projected = np.full(len(out_df), np.nan)
        for i in range(len(out_df)):
            fp = fp_aoi.iloc[i]
            if pd.isna(fp) or fp == 0.0:
                continue
            vol = vol_aoi.iloc[i]
            if pd.isna(vol):
                vol = 0.0
            density      = vol / fp
            unobserved   = max(0.0, total_area - fp)
            projected[i] = round(density * unobserved, 2)

        out_df[aoi] = projected

    n_filled = out_df[aoi_names].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label} (AOI): {n_filled}/{len(out_df)} window(s) "
        "with at least one AOI projected"
    )
    return out_df


def combine_volumes_aoi(
    observed_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    aoi_names: list,
    layer_label: str,
) -> pd.DataFrame:
    """
    Build the total (observed + projected) volume DataFrame at AOI level.

    For each row and AOI:
      • projected NaN → total NaN
      • projected not NaN → total = observed_AOI (NaN → 0) + projected

    Mirrors combine_areas_aoi in project_melt_area_by_AOI.py.
    """
    out_df = projected_df.copy()

    for aoi in aoi_names:
        obs_aoi  = aggregate_bins_to_aoi(observed_df, aoi)
        proj_col = projected_df[aoi]

        combined       = proj_col.copy()
        mask           = proj_col.notna()
        combined[mask] = obs_aoi[mask].fillna(0.0) + proj_col[mask]
        out_df[aoi]    = combined.round(2)

    n_filled = out_df[aoi_names].notna().any(axis=1).sum()
    debug_print(
        f"  {layer_label} (AOI): {n_filled}/{len(out_df)} window(s) "
        "with at least one AOI combined"
    )
    return out_df


# ============================================================
# 5) MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Validate required inputs
    # --------------------------------------------------------
    for p in (TEMPLATE_CSV, FOOTPRINT_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Required input CSV not found: {p}")
    for aoi in AOIS:
        for p in (AOI_BINS_SHPS[aoi], AOI_BOUNDARY_SHPS[aoi]):
            if not p.exists():
                raise FileNotFoundError(f"[{aoi}] Required shapefile not found: {p}")

    # --------------------------------------------------------
    # Load shapefiles
    # --------------------------------------------------------
    debug_print("Loading elevation bin shapefiles …")
    ost_bins = load_bins(OST_BINS_SHP, "OST")
    ptm_bins = load_bins(PTM_BINS_SHP, "PTM")
    bin_cols = all_bin_columns(ost_bins, ptm_bins)
    debug_print(f"  Total bin columns: {len(bin_cols)}")

    bin_total_areas: dict = {}
    bin_total_areas.update(build_bin_total_areas(ost_bins, "OST"))
    bin_total_areas.update(build_bin_total_areas(ptm_bins, "PTM"))

    debug_print("\nLoading AOI boundary shapefiles …")
    total_aoi_areas = {aoi: load_aoi_area(AOI_BOUNDARY_SHPS[aoi], aoi) for aoi in AOIS}

    # --------------------------------------------------------
    # Load template and footprint CSVs
    # --------------------------------------------------------
    debug_print("\nLoading template and footprint CSVs …")
    template_df   = normalize_date_columns(pd.read_csv(TEMPLATE_CSV))
    footprint_df  = normalize_date_columns(pd.read_csv(FOOTPRINT_CSV))
    debug_print(f"  Template rows : {len(template_df)}")
    debug_print(f"  Footprint rows: {len(footprint_df)}")

    if len(template_df) != len(footprint_df):
        warnings.warn(
            "Template and footprint CSVs have different row counts. "
            "Rows are matched by position — verify both share the same window set."
        )

    # Ensure footprint_df has all expected bin columns (add missing ones as NaN)
    for col in bin_cols:
        if col not in footprint_df.columns:
            footprint_df[col] = np.nan

    # --------------------------------------------------------
    # Scan lake shapefiles
    # --------------------------------------------------------
    debug_print("\nScanning lake shapefile folders …")
    lake_map = {aoi: scan_lake_files(AOI_LAKE_FOLDERS[aoi], aoi) for aoi in AOIS}

    # --------------------------------------------------------
    # Startup diagnostic: inspect the first available lake file
    # to confirm the volume column exists and show all attributes.
    # --------------------------------------------------------
    debug_print(f"\nVolume column setting: VOL_COLUMN = '{VOL_COLUMN}'")
    _checked_first = False
    for _aoi in AOIS:
        if lake_map[_aoi]:
            _first_path = next(iter(lake_map[_aoi].values()))
            try:
                _sample = gpd.read_file(_first_path)
                _non_geom = [c for c in _sample.columns if c != "geometry"]
                debug_print(
                    f"  First lake file ({_aoi}): {_first_path.name}\n"
                    f"    Attributes found: {_non_geom}"
                )
                if VOL_COLUMN not in _sample.columns:
                    print(
                        f"\n*** WARNING: VOL_COLUMN='{VOL_COLUMN}' is NOT present in the lake "
                        f"shapefiles.\n"
                        f"    Available columns: {_non_geom}\n"
                        f"    Set VOL_COLUMN at the top of this script to the correct field name,\n"
                        f"    OR add a '{VOL_COLUMN}' attribute (average depth in m) to the "
                        f"shapefiles.\n"
                        f"    All observed volumes will remain NaN until this is fixed. ***\n"
                    )
                else:
                    debug_print(
                        f"    '{VOL_COLUMN}' column found — proceeding with volume computation."
                    )
                _checked_first = True
            except Exception:
                pass
        if _checked_first:
            break

    # --------------------------------------------------------
    # Build observed-volume DataFrame (bin level)
    # Rows follow the template; bins filled from shapefiles.
    # --------------------------------------------------------
    debug_print("\nComputing observed lake volumes per bin …")
    obs_vol_df = template_df.copy()
    for col in bin_cols:
        obs_vol_df[col] = np.nan

    total = len(template_df)
    missing_col_counts: dict = {aoi: 0 for aoi in AOIS}   # files lacking VOL_COLUMN

    for idx, row in template_df.iterrows():
        key   = (row["win_start"], row["win_end"])
        label = f"{key[0]} – {key[1]}"

        for aoi, bins_gdf in [("OST", ost_bins), ("PTM", ptm_bins)]:
            if key not in lake_map[aoi]:
                continue
            try:
                volumes, col_missing = compute_bin_volumes(
                    bins_gdf, lake_map[aoi][key], aoi, vol_col=VOL_COLUMN
                )
                if col_missing:
                    missing_col_counts[aoi] += 1
                    continue
                for cname, val in volumes.items():
                    obs_vol_df.at[idx, cname] = val

                n_nonzero = sum(1 for v in volumes.values() if v > 0)
                debug_print(
                    f"[{idx + 1}/{total}] {aoi} lake {label}: "
                    f"{n_nonzero}/{len(volumes)} bin(s) with volume > 0"
                )
            except Exception as exc:
                warnings.warn(f"[{aoi}] lake {label}: {exc}")

    for aoi in AOIS:
        if missing_col_counts[aoi] > 0:
            print(
                f"  [{aoi}] {missing_col_counts[aoi]} lake file(s) missing "
                f"'{VOL_COLUMN}' — those windows left as NaN. "
                f"Set VOL_COLUMN to the correct depth field name."
            )

    # Align obs_vol_df rows with footprint_df by position for projection step
    # (same assumption as project_melt_area_by_aspect_elev_bins.py)
    obs_vol_aligned = obs_vol_df.copy()
    for col in bin_cols:
        if col not in obs_vol_aligned.columns:
            obs_vol_aligned[col] = np.nan

    # --------------------------------------------------------
    # Bin-level projections
    # --------------------------------------------------------
    debug_print("\nProjecting lake volumes into unobserved bin zones …")
    proj_vol_bins_df = project_volume(
        footprint_df, obs_vol_aligned, bin_cols, bin_total_areas, "lake"
    )

    debug_print("\nCombining observed and projected bin volumes …")
    total_vol_bins_df = combine_volumes(
        obs_vol_aligned, proj_vol_bins_df, bin_cols, "lake"
    )

    # --------------------------------------------------------
    # AOI-level projections
    # --------------------------------------------------------
    debug_print("\nProjecting lake volumes into unobserved AOI zones …")
    meta_cols = [c for c in footprint_df.columns
                 if not any(c.startswith(f"{aoi}_") for aoi in AOIS)]

    proj_vol_aoi_df = project_volume_aoi(
        footprint_df, obs_vol_aligned, AOIS, total_aoi_areas, meta_cols, "lake"
    )

    debug_print("\nCombining observed and projected AOI volumes …")
    total_vol_aoi_df = combine_volumes_aoi(
        obs_vol_aligned, proj_vol_aoi_df, AOIS, "lake"
    )

    # --------------------------------------------------------
    # Enforce column order: template / meta columns first, then bin columns
    # --------------------------------------------------------
    original_cols = list(pd.read_csv(TEMPLATE_CSV, nrows=0).columns)
    for col in bin_cols:
        if col not in original_cols:
            original_cols.append(col)

    obs_vol_df        = obs_vol_df[original_cols]
    proj_vol_bins_df  = proj_vol_bins_df[[c for c in original_cols if c in proj_vol_bins_df.columns]]
    total_vol_bins_df = total_vol_bins_df[[c for c in original_cols if c in total_vol_bins_df.columns]]

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    obs_vol_df.to_csv(OBS_VOL_BINS_CSV,    index=False)
    proj_vol_bins_df.to_csv(PROJ_VOL_BINS_CSV,  index=False)
    total_vol_bins_df.to_csv(TOTAL_VOL_BINS_CSV, index=False)
    proj_vol_aoi_df.to_csv(PROJ_VOL_AOI_CSV,   index=False)
    total_vol_aoi_df.to_csv(TOTAL_VOL_AOI_CSV,  index=False)

    debug_print(f"\nObserved lake volume (bins) written to:      {OBS_VOL_BINS_CSV}")
    debug_print(f"Projected lake volume (bins) written to:     {PROJ_VOL_BINS_CSV}")
    debug_print(f"Total lake volume (bins) written to:         {TOTAL_VOL_BINS_CSV}")
    debug_print(f"Projected lake volume (AOI) written to:      {PROJ_VOL_AOI_CSV}")
    debug_print(f"Total lake volume (AOI) written to:          {TOTAL_VOL_AOI_CSV}")

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

        for layer_label, df, cols in [
            ("observed volume (bins)",  obs_vol_df,        aoi_cols),
            ("projected volume (bins)", proj_vol_bins_df,  aoi_cols),
            ("total volume (bins)",     total_vol_bins_df, aoi_cols),
        ]:
            present   = [c for c in cols if c in df.columns]
            n_blank   = df[present].isna().all(axis=1).sum()
            n_covered = (df[present] > 0).any(axis=1).sum()
            n_zero    = (
                df[present].notna().any(axis=1) & ~(df[present] > 0).any(axis=1)
            ).sum()
            debug_print(
                f"  {aoi} {layer_label} ({bin_desc}): "
                f"{n_covered} window(s) with ≥1 bin > 0  |  "
                f"{n_zero} window(s) with data but all zero  |  "
                f"{n_blank} window(s) with no data (all blank)"
            )

        for layer_label, df in [
            ("projected volume (AOI)", proj_vol_aoi_df),
            ("total volume (AOI)",     total_vol_aoi_df),
        ]:
            if aoi not in df.columns:
                continue
            n_blank   = df[aoi].isna().sum()
            n_covered = (df[aoi] > 0).sum()
            n_zero    = (df[aoi].notna() & ~(df[aoi] > 0)).sum()
            debug_print(
                f"  {aoi} {layer_label}: "
                f"{n_covered} window(s) with volume > 0  |  "
                f"{n_zero} window(s) with data but zero volume  |  "
                f"{n_blank} window(s) with no data (blank)"
            )

    debug_print("\nDone.")


if __name__ == "__main__":
    main()
