# -*- coding: utf-8 -*-
"""
footprint_table.py

Summarize GEE-exported footprint shapefiles to a CSV using a template CSV
as the master list of mosaic windows (2013–2025).

Reads combined lake/slush/footprint shapefiles from all_OST_files and
all_PTM_files folders, isolates the footprint layer, and computes:
    footprint_area_m2   = sum(geometry area in projected CRS) per window
    footprint_proportion = footprint_area_m2 / study_area_m2

Missing windows in the shapefile data remain null in the output CSV
(left-merge against the template).

Template columns populated:
    OST_footprint_area_m2
    OST_footprint_proportion
    PTM_footprint_area_m2
    PTM_footprint_proportion
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1) USER SETTINGS 
# ============================================================

# Folder paths for combined OST / PTM shapefiles (lake + slush + footprint)
OST_FOLDER = Path(r"Q:\ThesisData\data\raw_files\all_OST_files")
PTM_FOLDER = Path(r"Q:\ThesisData\data\raw_files\all_PTM_files")

# Study area shapefiles used to compute footprint proportion
OST_STUDY_AREA_SHP = Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp")
PTM_STUDY_AREA_SHP = Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp")

# Template CSV (pre-populated win_start / win_end rows, empty data columns)
TEMPLATE_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_templates\footprint_summary_template.csv"
)

# Output CSV
OUTPUT_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\footprint_summary.csv"
)

# Expected projected CRS for area calculations
EXPECTED_CRS = "EPSG:3995"

# Optional: if True, prints progress and warnings
DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    """Print only when DEBUG is True."""
    if DEBUG:
        print(*args)


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert win_start and win_end to ISO yyyy-mm-dd strings so joins
    work consistently across shapefiles and the template CSV.
    """
    out = df.copy()
    for col in ["win_start", "win_end"]:
        if col not in out.columns:
            raise KeyError(f"Required column missing: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def empty_summary_df(columns: list) -> pd.DataFrame:
    """Return an empty DataFrame with the requested columns."""
    return pd.DataFrame(columns=columns)


def infer_layer_type_from_name(shp_path: Path) -> str | None:
    """
    Infer layer type from filename if needed.
    Expected names usually end with _lake, _slush, or _footprint.
    """
    name = shp_path.stem.lower()
    if name.endswith("_lake"):
        return "lake"
    if name.endswith("_slush"):
        return "slush"
    if name.endswith("_footprint"):
        return "footprint"
    return None


def read_one_shapefile(shp_path: Path) -> gpd.GeoDataFrame | None:
    """
    Read a single shapefile safely. Returns None if read fails.
    """
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            debug_print(f"  Empty shapefile skipped: {shp_path.name}")
        return gdf
    except Exception as exc:
        warnings.warn(f"Could not read {shp_path.name}: {exc}")
        return None


def validate_required_window_fields(gdf: gpd.GeoDataFrame, shp_path: Path) -> bool:
    """Ensure win_start and win_end fields exist."""
    required = {"win_start", "win_end"}
    missing = required - set(gdf.columns)
    if missing:
        warnings.warn(
            f"{shp_path.name} missing required fields: {sorted(missing)}. Skipping."
        )
        return False
    return True


def determine_layer_type(gdf: gpd.GeoDataFrame, shp_path: Path) -> str | None:
    """
    Prefer layer_type attribute if present and non-null.
    Otherwise infer from filename.
    """
    if "layer_type" in gdf.columns:
        non_null = gdf["layer_type"].dropna().astype(str).str.lower().unique().tolist()
        if len(non_null) == 1:
            return non_null[0]
        if len(non_null) > 1:
            warnings.warn(
                f"{shp_path.name} has multiple layer_type values {non_null}. "
                f"Using first value."
            )
            return non_null[0]
    return infer_layer_type_from_name(shp_path)


def get_study_area_m2(shp_path: Path, prefix: str) -> float:
    """
    Read a study area shapefile and return the total area in m²
    (reprojects to EXPECTED_CRS if needed).
    """
    if not shp_path.exists():
        raise FileNotFoundError(
            f"[{prefix}] Study area shapefile not found: {shp_path}"
        )

    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"[{prefix}] Study area shapefile is empty: {shp_path}")

    if gdf.crs is None:
        warnings.warn(
            f"[{prefix}] Study area shapefile has no CRS. "
            "Area may be wrong if geometry is not already in metres."
        )
    elif str(gdf.crs) != EXPECTED_CRS:
        debug_print(
            f"[{prefix}] Reprojecting study area from {gdf.crs} to {EXPECTED_CRS}"
        )
        gdf = gdf.to_crs(EXPECTED_CRS)

    total_area = float(gdf.geometry.area.sum())
    debug_print(f"[{prefix}] Study area: {total_area:,.1f} m²")
    return total_area


# ============================================================
# 3) FOOTPRINT SUMMARIZATION
# ============================================================

def summarize_footprint(gdf: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """
    Summarize footprint polygons by win_start / win_end:
        footprint_area_m2 = sum(geometry.area)

    Reprojects to EXPECTED_CRS if needed.
    """
    work = gdf.copy()

    if work.crs is None:
        warnings.warn(
            f"[{prefix}] Footprint shapefile has no CRS. Area may be wrong."
        )
    elif str(work.crs) != EXPECTED_CRS:
        debug_print(
            f"[{prefix}] Reprojecting footprint from {work.crs} to {EXPECTED_CRS}"
        )
        work = work.to_crs(EXPECTED_CRS)

    work["footprint_area_m2"] = work.geometry.area

    summary = (
        work.groupby(["win_start", "win_end"], dropna=False, as_index=False)
        .agg(footprint_area_m2=("footprint_area_m2", "sum"))
    )

    return summary


# ============================================================
# 4) PER-AOI FOLDER PROCESSING
# ============================================================

def process_aoi_folder(folder: Path, prefix: str) -> pd.DataFrame:
    """
    Read all shapefiles from *folder*, keep only footprint-type features,
    and return a summary DataFrame with columns:
        win_start, win_end, footprint_area_m2
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"[{prefix}] Shapefile folder not found: {folder}"
        )

    shp_files = sorted(folder.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(
            f"[{prefix}] No .shp files found in: {folder}"
        )

    debug_print(f"[{prefix}] Found {len(shp_files)} shapefile(s).")

    footprint_gdfs = []

    for shp_path in shp_files:
        debug_print(f"[{prefix}] Reading: {shp_path.name}")

        gdf = read_one_shapefile(shp_path)
        if gdf is None or gdf.empty:
            continue

        if not validate_required_window_fields(gdf, shp_path):
            continue

        layer_type = determine_layer_type(gdf, shp_path)
        if layer_type != "footprint":
            # Skip lake and slush layers silently
            continue

        # Normalize date keys
        gdf["win_start"] = pd.to_datetime(
            gdf["win_start"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        gdf["win_end"] = pd.to_datetime(
            gdf["win_end"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # Drop rows with broken date keys
        bad_keys = gdf["win_start"].isna() | gdf["win_end"].isna()
        if bad_keys.any():
            warnings.warn(
                f"{shp_path.name}: dropping {bad_keys.sum()} row(s) with "
                "invalid win_start or win_end."
            )
            gdf = gdf.loc[~bad_keys].copy()

        if gdf.empty:
            continue

        footprint_gdfs.append(gdf)

    if footprint_gdfs:
        footprint_all = gpd.GeoDataFrame(
            pd.concat(footprint_gdfs, ignore_index=True),
            crs=footprint_gdfs[0].crs,
        )
        footprint_summary = summarize_footprint(footprint_all, prefix)
    else:
        debug_print(f"[{prefix}] No footprint shapefiles found.")
        footprint_summary = empty_summary_df(
            ["win_start", "win_end", "footprint_area_m2"]
        )

    return footprint_summary


# ============================================================
# 5) MERGE INTO TEMPLATE
# ============================================================

def merge_into_template(
    template_df: pd.DataFrame,
    footprint_summary: pd.DataFrame,
    study_area_m2: float,
    prefix: str,
) -> pd.DataFrame:
    """
    Left-merge footprint summary onto the template and populate:
        {prefix}_footprint_area_m2
        {prefix}_footprint_proportion

    Windows present in the template but not in the shapefile data
    remain null (left-join behaviour).
    """
    out = template_df.copy()

    out = out.merge(footprint_summary, on=["win_start", "win_end"], how="left")

    col_area = f"{prefix}_footprint_area_m2"
    col_prop = f"{prefix}_footprint_proportion"

    out[col_area] = out["footprint_area_m2"]
    out[col_prop] = out["footprint_area_m2"] / study_area_m2

    out = out.drop(columns=["footprint_area_m2"], errors="ignore")

    return out


# ============================================================
# 6) MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Read template
    # --------------------------------------------------------
    if not TEMPLATE_CSV.exists():
        raise FileNotFoundError(f"Template CSV not found: {TEMPLATE_CSV}")

    template_df = pd.read_csv(TEMPLATE_CSV)
    template_df = normalize_date_columns(template_df)

    debug_print(f"Template rows: {len(template_df)}")
    debug_print(f"Template columns: {list(template_df.columns)}")

    # --------------------------------------------------------
    # Get study area totals for proportion calculation
    # --------------------------------------------------------
    ost_study_area_m2 = get_study_area_m2(OST_STUDY_AREA_SHP, "OST")
    ptm_study_area_m2 = get_study_area_m2(PTM_STUDY_AREA_SHP, "PTM")

    # --------------------------------------------------------
    # Process OST
    # --------------------------------------------------------
    debug_print("\n--- Processing AOI: OST ---")
    ost_footprint_summary = process_aoi_folder(OST_FOLDER, "OST")

    result_df = merge_into_template(
        template_df=template_df,
        footprint_summary=ost_footprint_summary,
        study_area_m2=ost_study_area_m2,
        prefix="OST",
    )

    # --------------------------------------------------------
    # Process PTM
    # --------------------------------------------------------
    debug_print("\n--- Processing AOI: PTM ---")
    ptm_footprint_summary = process_aoi_folder(PTM_FOLDER, "PTM")

    result_df = merge_into_template(
        template_df=result_df,
        footprint_summary=ptm_footprint_summary,
        study_area_m2=ptm_study_area_m2,
        prefix="PTM",
    )

    # --------------------------------------------------------
    # Enforce original template column order
    # --------------------------------------------------------
    template_cols = list(pd.read_csv(TEMPLATE_CSV, nrows=0).columns)
    for col in template_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    result_df = result_df[template_cols]

    # --------------------------------------------------------
    # Write output CSV
    # --------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    debug_print(f"\nOutput written to: {OUTPUT_CSV}")

    # --------------------------------------------------------
    # QA preview
    # --------------------------------------------------------
    debug_print("\nPreview:")
    debug_print(result_df.head(10))

    debug_print("\nMissing-value counts:")
    for col in [
        "OST_footprint_area_m2",
        "OST_footprint_proportion",
        "PTM_footprint_area_m2",
        "PTM_footprint_proportion",
    ]:
        if col in result_df.columns:
            debug_print(f"  {col}: {result_df[col].isna().sum()} null(s)")


if __name__ == "__main__":
    main()
