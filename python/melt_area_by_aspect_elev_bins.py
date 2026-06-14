"""
melt_area_by_aspect_elev_bins.py

For each melt-season two-day window, computes the area (m²) and proportion
of each aspect-split elevation bin that is covered by lake and slush polygons.

Elevation bins are taken from the aspect-split shapefiles produced by
split_elevation_bins_by_aspect.py:
    PTM_aspect_elevation_bins.shp  /  OST_aspect_elevation_bins.shp

Each feature has start_elv, end_elv, sector ('main' | 'west' | 'east').

Output column naming matches select_by_apect_elev_bin_coverage.csv:

Bins below 1000 m have sector='main' and keep the plain column name:
    OST_0_100, OST_100_200, …, OST_900_1000

Bins at or above 1000 m are split east / west and get an E or W suffix:
    OST_1000_1100_E, OST_1000_1100_W, …, OST_1400_1500_E, OST_1400_1500_W
    PTM_1000_1100_E, PTM_1000_1100_W, …, PTM_1400_1500_E, PTM_1400_1500_W

Blank cells : no melt shapefile exists for that window / AOI.
Zero cells  : melt shapefile exists but does not overlap the bin.

Outputs
-------
    slush_area_by_aspect_elev_bins.csv       — intersection area in m²
    lake_area_by_aspect_elev_bins.csv        — intersection area in m²
    slush_proportion_by_aspect_elev_bins.csv — fraction of bin area (0.0–1.0)
    lake_proportion_by_aspect_elev_bins.csv  — fraction of bin area (0.0–1.0)
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

# Folders containing per-window lake and slush shapefiles.
# Files are expected to be named:
#   {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_lake.shp
#   {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_slush.shp
PTM_FOLDER = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_PTM_files")
OST_FOLDER = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_OST_files")

TEMPLATE_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_templates\all_windows_template.csv")

SLUSH_AREA_CSV       = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\slush_area_by_aspect_elev_bins.csv")
LAKE_AREA_CSV        = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\lake_area_by_aspect_elev_bins.csv")
SLUSH_PROPORTION_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\slush_proportion_by_aspect_elev_bins.csv")
LAKE_PROPORTION_CSV  = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\lake_proportion_by_aspect_elev_bins.csv")

# Projected CRS used for all area calculations
PROJECTED_CRS = "EPSG:3995"

DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


# Filename patterns: AOI_YYYY-MM-DD_YYYY-MM-DD_{type}.shp
_LAKE_RE = re.compile(
    r"^(?P<aoi>[A-Z]+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"lake\.shp$",
    re.IGNORECASE,
)
_SLUSH_RE = re.compile(
    r"^(?P<aoi>[A-Z]+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"slush\.shp$",
    re.IGNORECASE,
)


def col_name(aoi: str, start_elv, end_elv, sector: str) -> str:
    """
    Return the output column name for an aspect-split elevation bin.

    sector='main'  → 'OST_0_100'          (no suffix)
    sector='east'  → 'OST_1000_1100_E'
    sector='west'  → 'OST_1000_1100_W'
    """
    base = f"{aoi}_{int(start_elv)}_{int(end_elv)}"
    if sector == "east":
        return base + "_E"
    if sector == "west":
        return base + "_W"
    return base  # 'main'


def build_bin_columns(bins_gdf: gpd.GeoDataFrame, aoi: str) -> list:
    """
    Return an ordered list of column names from an aspect-split bin GeoDataFrame.

    The GDF is expected to be sorted by (start_elv, sector), which is the
    order produced by split_elevation_bins_by_aspect.py.
    """
    cols = []
    for _, row in bins_gdf.iterrows():
        cols.append(col_name(aoi, row["start_elv"], row["end_elv"], row["sector"]))
    return cols


def all_bin_columns(ost_bins: gpd.GeoDataFrame, ptm_bins: gpd.GeoDataFrame) -> list:
    """Return ordered column names for OST first, then PTM."""
    return build_bin_columns(ost_bins, "OST") + build_bin_columns(ptm_bins, "PTM")


def scan_melt_files(folder: Path, aoi_prefix: str, pattern: re.Pattern, layer: str) -> dict:
    """
    Scan *folder* for melt shapefiles (lake or slush) matching *aoi_prefix*.

    Returns
    -------
    dict mapping (win_start_str, win_end_str) -> Path
    where date strings are ISO format YYYY-MM-DD.
    """
    result = {}
    prefix_upper = aoi_prefix.upper()

    if not folder.exists():
        raise FileNotFoundError(f"[{aoi_prefix}] Melt folder not found: {folder}")

    for shp_path in folder.glob("*.shp"):
        m = pattern.match(shp_path.name)
        if m and m.group("aoi").upper() == prefix_upper:
            start_str = m.group("start")
            end_str   = m.group("end")
            key = (start_str, end_str)
            if key in result:
                warnings.warn(
                    f"[{aoi_prefix}] Duplicate {layer} file for {start_str}–{end_str}; "
                    f"keeping first: {result[key].name}"
                )
            else:
                result[key] = shp_path

    return result


def load_bins(shp_path: Path, aoi: str) -> gpd.GeoDataFrame:
    """
    Read an aspect-split elevation-bin shapefile, reproject to PROJECTED_CRS,
    and return a GeoDataFrame sorted by (start_elv, sector).

    Expected columns: start_elv, end_elv, sector, geometry.
    """
    if not shp_path.exists():
        raise FileNotFoundError(f"[{aoi}] Aspect-split bin shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"[{aoi}] Aspect-split bin shapefile contains no features: {shp_path}")

    for col in ("start_elv", "end_elv", "sector"):
        if col not in gdf.columns:
            raise KeyError(
                f"[{aoi}] Aspect-split bin shapefile is missing column '{col}'. "
                "Run split_elevation_bins_by_aspect.py to generate the required files."
            )

    if gdf.crs is None:
        warnings.warn(
            f"[{aoi}] Aspect-split bin shapefile has no CRS; "
            "area calculations may be unreliable."
        )
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


def compute_bin_stats(
    bins_gdf: gpd.GeoDataFrame,
    melt_path: Path,
    aoi: str,
    layer: str,
) -> tuple[dict, dict]:
    """
    For every bin feature in *bins_gdf*, compute the area (m²) and proportion
    of that feature covered by the melt polygons (lake or slush) in *melt_path*.

    Returns
    -------
    (areas, proportions) — two dicts mapping col_name -> float.
      areas       : intersection area in m²  (>= 0.0)
      proportions : intersection area / bin area  (0.0–1.0)
    Bins with zero/null geometry are omitted (their columns stay NaN).
    """
    melt_gdf = gpd.read_file(melt_path)

    if melt_gdf.empty:
        debug_print(f"  [{aoi}] {layer.capitalize()} file is empty: {melt_path.name}")
        empty = {
            col_name(aoi, row["start_elv"], row["end_elv"], row["sector"]): 0.0
            for _, row in bins_gdf.iterrows()
        }
        return empty, empty.copy()

    if melt_gdf.crs is None:
        warnings.warn(
            f"[{aoi}] {layer.capitalize()} {melt_path.name} has no CRS; "
            "reprojection skipped — area may be wrong."
        )
    elif str(melt_gdf.crs) != PROJECTED_CRS:
        melt_gdf = melt_gdf.to_crs(PROJECTED_CRS)

    melt_union = melt_gdf.union_all()

    areas       = {}
    proportions = {}
    for _, bin_row in bins_gdf.iterrows():
        cname    = col_name(aoi, bin_row["start_elv"], bin_row["end_elv"], bin_row["sector"])
        bin_geom = bin_row["geometry"]

        if bin_geom is None or bin_geom.is_empty:
            continue  # degenerate bin — column stays NaN

        bin_area = bin_geom.area
        if bin_area == 0.0:
            continue

        inter_area        = bin_geom.intersection(melt_union).area
        areas[cname]       = float(inter_area)
        proportions[cname] = float(inter_area / bin_area)

    return areas, proportions


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce win_start / win_end to ISO YYYY-MM-DD strings."""
    out = df.copy()
    for col in ("win_start", "win_end"):
        if col not in out.columns:
            raise KeyError(f"Template CSV is missing required column: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


# ============================================================
# 3) MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Load template
    # --------------------------------------------------------
    if not TEMPLATE_CSV.exists():
        raise FileNotFoundError(f"Template CSV not found: {TEMPLATE_CSV}")

    template_df = pd.read_csv(TEMPLATE_CSV)
    template_df = normalize_date_columns(template_df)
    debug_print(f"Template rows: {len(template_df)}")

    # --------------------------------------------------------
    # Load aspect-split elevation bin shapefiles (done once)
    # --------------------------------------------------------
    debug_print("\nLoading aspect-split elevation bin shapefiles …")
    ost_bins = load_bins(OST_BINS_SHP, "OST")
    ptm_bins = load_bins(PTM_BINS_SHP, "PTM")

    aoi_config = [("OST", ost_bins), ("PTM", ptm_bins)]

    # Build ordered column list
    bin_cols = all_bin_columns(ost_bins, ptm_bins)
    debug_print(f"Total output bin columns: {len(bin_cols)}")

    # Four output DataFrames (area + proportion, slush + lake) — all NaN initially
    slush_area_df  = template_df.copy()
    lake_area_df   = template_df.copy()
    slush_prop_df  = template_df.copy()
    lake_prop_df   = template_df.copy()
    for col in bin_cols:
        for df in (slush_area_df, lake_area_df, slush_prop_df, lake_prop_df):
            if col not in df.columns:
                df[col] = np.nan

    # --------------------------------------------------------
    # Scan melt file folders
    # --------------------------------------------------------
    debug_print("\nScanning melt file folders …")
    slush_map = {
        "OST": scan_melt_files(OST_FOLDER, "OST", _SLUSH_RE, "slush"),
        "PTM": scan_melt_files(PTM_FOLDER, "PTM", _SLUSH_RE, "slush"),
    }
    lake_map = {
        "OST": scan_melt_files(OST_FOLDER, "OST", _LAKE_RE, "lake"),
        "PTM": scan_melt_files(PTM_FOLDER, "PTM", _LAKE_RE, "lake"),
    }
    debug_print(f"  OST slush files found: {len(slush_map['OST'])}")
    debug_print(f"  PTM slush files found: {len(slush_map['PTM'])}")
    debug_print(f"  OST lake files found:  {len(lake_map['OST'])}")
    debug_print(f"  PTM lake files found:  {len(lake_map['PTM'])}")

    # --------------------------------------------------------
    # Fill output DataFrames row by row
    # --------------------------------------------------------
    debug_print("\nProcessing windows …")
    total = len(template_df)

    for idx, row in template_df.iterrows():
        key   = (row["win_start"], row["win_end"])
        label = f"{key[0]} – {key[1]}"

        for aoi, bins_gdf in aoi_config:
            # --- Slush ---
            if key in slush_map[aoi]:
                try:
                    areas, props = compute_bin_stats(bins_gdf, slush_map[aoi][key], aoi, "slush")
                    for cname, area in areas.items():
                        slush_area_df.at[idx, cname] = round(area, 2)
                    for cname, prop in props.items():
                        slush_prop_df.at[idx, cname] = round(prop, 6)

                    n_nonzero = sum(1 for v in areas.values() if v > 0)
                    debug_print(
                        f"[{idx + 1}/{total}] {aoi} slush {label}: "
                        f"{n_nonzero}/{len(areas)} bin(s) with melt area"
                    )
                except Exception as exc:
                    warnings.warn(f"[{aoi}] slush {label}: {exc}")

            # --- Lake ---
            if key in lake_map[aoi]:
                try:
                    areas, props = compute_bin_stats(bins_gdf, lake_map[aoi][key], aoi, "lake")
                    for cname, area in areas.items():
                        lake_area_df.at[idx, cname] = round(area, 2)
                    for cname, prop in props.items():
                        lake_prop_df.at[idx, cname] = round(prop, 6)

                    n_nonzero = sum(1 for v in areas.values() if v > 0)
                    debug_print(
                        f"[{idx + 1}/{total}] {aoi} lake  {label}: "
                        f"{n_nonzero}/{len(areas)} bin(s) with melt area"
                    )
                except Exception as exc:
                    warnings.warn(f"[{aoi}] lake {label}: {exc}")

    # --------------------------------------------------------
    # Enforce column order: template columns first, then bin columns
    # --------------------------------------------------------
    original_cols = list(pd.read_csv(TEMPLATE_CSV, nrows=0).columns)
    for col in bin_cols:
        if col not in original_cols:
            original_cols.append(col)

    slush_area_df = slush_area_df[original_cols]
    lake_area_df  = lake_area_df[original_cols]
    slush_prop_df = slush_prop_df[original_cols]
    lake_prop_df  = lake_prop_df[original_cols]

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    for csv_path in (SLUSH_AREA_CSV, LAKE_AREA_CSV, SLUSH_PROPORTION_CSV, LAKE_PROPORTION_CSV):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    slush_area_df.to_csv(SLUSH_AREA_CSV,       index=False)
    lake_area_df.to_csv(LAKE_AREA_CSV,          index=False)
    slush_prop_df.to_csv(SLUSH_PROPORTION_CSV,  index=False)
    lake_prop_df.to_csv(LAKE_PROPORTION_CSV,    index=False)

    debug_print(f"\nSlush area written to:       {SLUSH_AREA_CSV}")
    debug_print(f"Lake area written to:        {LAKE_AREA_CSV}")
    debug_print(f"Slush proportion written to: {SLUSH_PROPORTION_CSV}")
    debug_print(f"Lake proportion written to:  {LAKE_PROPORTION_CSV}")

    # --------------------------------------------------------
    # QA summary
    # --------------------------------------------------------
    debug_print("\n── QA Summary ──────────────────────────────────────────")
    for aoi, bins_gdf in aoi_config:
        aoi_cols = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        n_main   = (bins_gdf["sector"] == "main").sum()
        n_east   = (bins_gdf["sector"] == "east").sum()
        n_west   = (bins_gdf["sector"] == "west").sum()
        bin_desc = f"{n_main} main + {n_east} east + {n_west} west bins"

        for layer_label, df in (
            ("slush area",       slush_area_df),
            ("lake area",        lake_area_df),
            ("slush proportion", slush_prop_df),
            ("lake proportion",  lake_prop_df),
        ):
            n_blank   = df[aoi_cols].isna().all(axis=1).sum()
            n_covered = (df[aoi_cols] > 0).any(axis=1).sum()
            n_zero    = (
                df[aoi_cols].notna().any(axis=1)
                & ~(df[aoi_cols] > 0).any(axis=1)
            ).sum()
            debug_print(
                f"  {aoi} {layer_label} ({bin_desc}): "
                f"{n_covered} window(s) with ≥1 bin covered  |  "
                f"{n_zero} window(s) with data but no bin overlap  |  "
                f"{n_blank} window(s) with no data (all blank)"
            )


if __name__ == "__main__":
    main()
