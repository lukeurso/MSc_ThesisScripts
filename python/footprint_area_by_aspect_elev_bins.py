"""
footprint_area_by_aspect_elev_bins.py

For each melt-season two-day window, computes the footprint area (m²) that
falls within each aspect-split elevation bin for that window.

Elevation bins are taken from the aspect-split shapefiles produced by
split_elevation_bins_by_aspect.py:
    PTM_aspect_elevation_bins.shp  /  OST_aspect_elevation_bins.shp

Each feature has start_elv, end_elv, sector ('main' | 'west' | 'east').

Bins below 1000 m have sector='main' and keep the plain column name:
    OST_0_100, OST_100_200, …, OST_900_1000

Bins at or above 1000 m are split east / west and get an E or W suffix:
    OST_1000_1100_E, OST_1000_1100_W, …, OST_1400_1500_E, OST_1400_1500_W
    PTM_1000_1100_E, PTM_1000_1100_W, …, PTM_1400_1500_E, PTM_1400_1500_W

Areas are in square metres (projected CRS: EPSG:3995).

Blank cells : no footprint exists for that window / AOI.
Zero cells  : footprint exists but does not overlap the bin.
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

PTM_FOLDER   = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_PTM_files")
OST_FOLDER   = Path(r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_OST_files")

TEMPLATE_CSV = Path(r"Q:\ThesisData\data\csv_files\csv_templates\all_windows_template.csv")
OUTPUT_CSV   = Path(r"Q:\ThesisData\data\csv_files\csv_outputs\footprint_area_by_aspect_elev_bins.csv")

# Projected CRS used for all area calculations
PROJECTED_CRS = "EPSG:3995"

DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


# Filename pattern: AOI_YYYY-MM-DD_YYYY-MM-DD_footprint.shp
_FOOTPRINT_RE = re.compile(
    r"^(?P<aoi>[A-Z]+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"footprint\.shp$",
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


def scan_footprints(folder: Path, aoi_prefix: str) -> dict:
    """
    Scan *folder* for footprint shapefiles matching *aoi_prefix*.

    Returns
    -------
    dict mapping (win_start_str, win_end_str) -> Path
    where date strings are ISO format YYYY-MM-DD.
    """
    result = {}
    prefix_upper = aoi_prefix.upper()

    if not folder.exists():
        raise FileNotFoundError(f"[{aoi_prefix}] Footprint folder not found: {folder}")

    for shp_path in folder.glob("*.shp"):
        m = _FOOTPRINT_RE.match(shp_path.name)
        if m and m.group("aoi").upper() == prefix_upper:
            start_str = m.group("start")
            end_str   = m.group("end")
            key = (start_str, end_str)
            if key in result:
                warnings.warn(
                    f"[{aoi_prefix}] Duplicate footprint for {start_str}–{end_str}; "
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


def compute_bin_areas(
    bins_gdf: gpd.GeoDataFrame,
    footprint_path: Path,
    aoi: str,
) -> dict:
    """
    For every bin feature in *bins_gdf* (including east/west splits), compute
    the footprint area (m²) that intersects that feature.

    Returns
    -------
    dict mapping col_name -> area in m² (float).
    Bins with zero/null geometry are omitted (their column stays NaN).
    """
    fp_gdf = gpd.read_file(footprint_path)

    if fp_gdf.empty:
        debug_print(f"  [{aoi}] Footprint file is empty: {footprint_path.name}")
        return {
            col_name(aoi, row["start_elv"], row["end_elv"], row["sector"]): 0.0
            for _, row in bins_gdf.iterrows()
        }

    if fp_gdf.crs is None:
        warnings.warn(
            f"[{aoi}] Footprint {footprint_path.name} has no CRS; "
            "reprojection skipped — area may be wrong."
        )
    elif str(fp_gdf.crs) != PROJECTED_CRS:
        fp_gdf = fp_gdf.to_crs(PROJECTED_CRS)

    foot_union = fp_gdf.union_all()

    result = {}
    for _, bin_row in bins_gdf.iterrows():
        cname    = col_name(aoi, bin_row["start_elv"], bin_row["end_elv"], bin_row["sector"])
        bin_geom = bin_row["geometry"]

        if bin_geom is None or bin_geom.is_empty:
            continue  # degenerate bin — column stays NaN

        if bin_geom.area == 0.0:
            continue

        intersection = bin_geom.intersection(foot_union)
        result[cname] = float(intersection.area)

    return result


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

    # Add all bin columns to the template (all NaN initially)
    bin_cols = all_bin_columns(ost_bins, ptm_bins)
    debug_print(f"Total output bin columns: {len(bin_cols)}")
    for col in bin_cols:
        if col not in template_df.columns:
            template_df[col] = np.nan

    # --------------------------------------------------------
    # Scan footprint folders
    # --------------------------------------------------------
    debug_print("\nScanning footprint folders …")
    fp_map = {
        "OST": scan_footprints(OST_FOLDER, "OST"),
        "PTM": scan_footprints(PTM_FOLDER, "PTM"),
    }
    debug_print(f"  OST footprints found: {len(fp_map['OST'])}")
    debug_print(f"  PTM footprints found: {len(fp_map['PTM'])}")

    # --------------------------------------------------------
    # Fill template row by row
    # --------------------------------------------------------
    debug_print("\nProcessing windows …")
    total = len(template_df)

    for idx, row in template_df.iterrows():
        key   = (row["win_start"], row["win_end"])
        label = f"{key[0]} – {key[1]}"

        for aoi, bins_gdf in aoi_config:
            if key not in fp_map[aoi]:
                # No footprint → all bin columns for this AOI stay NaN
                continue

            try:
                areas = compute_bin_areas(bins_gdf, fp_map[aoi][key], aoi)
                for cname, area in areas.items():
                    template_df.at[idx, cname] = round(area, 2)

                n_covered = sum(1 for v in areas.values() if v > 0)
                debug_print(
                    f"[{idx + 1}/{total}] {aoi} {label}: "
                    f"{n_covered}/{len(areas)} bin(s) with footprint area"
                )
            except Exception as exc:
                warnings.warn(f"[{aoi}] {label}: {exc}")

    # --------------------------------------------------------
    # Enforce column order: template columns first, then bin columns
    # --------------------------------------------------------
    original_cols = list(pd.read_csv(TEMPLATE_CSV, nrows=0).columns)
    for col in bin_cols:
        if col not in original_cols:
            original_cols.append(col)
    template_df = template_df[original_cols]

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    template_df.to_csv(OUTPUT_CSV, index=False)
    debug_print(f"\nOutput written to: {OUTPUT_CSV}")

    # --------------------------------------------------------
    # QA summary
    # --------------------------------------------------------
    debug_print("\n── QA Summary ──────────────────────────────────────────")
    for aoi, bins_gdf in aoi_config:
        aoi_cols  = [c for c in bin_cols if c.startswith(f"{aoi}_")]
        n_blank   = template_df[aoi_cols].isna().all(axis=1).sum()
        n_covered = (template_df[aoi_cols] > 0).any(axis=1).sum()
        n_zero    = (
            template_df[aoi_cols].notna().any(axis=1)
            & ~(template_df[aoi_cols] > 0).any(axis=1)
        ).sum()

        n_main = (bins_gdf["sector"] == "main").sum()
        n_east = (bins_gdf["sector"] == "east").sum()
        n_west = (bins_gdf["sector"] == "west").sum()

        debug_print(
            f"  {aoi} ({n_main} main + {n_east} east + {n_west} west bins): "
            f"{n_covered} window(s) with ≥1 bin with footprint area  |  "
            f"{n_zero} window(s) with footprint but no bin overlap  |  "
            f"{n_blank} window(s) with no footprint (all blank)"
        )


if __name__ == "__main__":
    main()
