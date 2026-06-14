#!/usr/bin/env python3
"""
create_n_block_elevation_csv_data.py

Generates an elevation summary CSV for 5-window block shapefiles.

For each block period and AOI the script reads footprint, slush, and lake
shapefiles and extracts the 'elev_mean' per-polygon attribute (added by
add_elevation_to_shapefiles.py).

Columns produced per AOI and layer:
  {AOI}_{layer}_elev_mean    – area-weighted mean of elev_mean across polygons
  {AOI}_{layer}_elev_max     – highest elev_mean value across polygons

For slush and lake layers, area-weighted percentile columns are also written:
  {AOI}_{layer}_elev_p10
  {AOI}_{layer}_elev_p25
  {AOI}_{layer}_elev_p50
  {AOI}_{layer}_elev_p75
  {AOI}_{layer}_elev_p90

Percentiles use the midpoint-convention weighted interpolation (each polygon's
elev_mean is weighted by its area).

NaN is written whenever a shapefile is absent, empty, or lacks 'elev_mean'.

Input shapefiles (inside AOI-specific folders):
    {YYYY-MM-DD}_{YYYY-MM-DD}_{footprint|slush|lake}.shp

Dependencies: geopandas, pandas, numpy
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

INPUT_PTM_DIR = Path(r"Q:\ThesisData\data\blocks\5_window\all_PTM_files_anchored")
INPUT_OST_DIR = Path(r"Q:\ThesisData\data\blocks\5_window\all_OST_files_anchored")

OUTPUT_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_elevation_data.csv"
)

PROJECTED_CRS = "EPSG:3995"

ELEV_COL = "elev_mean"

DEBUG = True


# ============================================================
# 2) CONSTANTS
# ============================================================

_BLOCK_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(footprint|slush|lake)\.shp$",
    re.IGNORECASE,
)

AOIS = ["OST", "PTM"]
LAYERS = ["footprint", "slush", "lake"]
PERCENTILE_LAYERS = {"slush", "lake"}
PERCENTILES = [10, 25, 50, 75, 90]

AOI_INPUT_DIRS = {
    "OST": INPUT_OST_DIR,
    "PTM": INPUT_PTM_DIR,
}


# ============================================================
# 3) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


def discover_blocks(input_dirs: dict) -> dict:
    """
    Scan each AOI folder for block shapefiles matching the naming convention.

    Returns
    -------
    {aoi: {(block_start_str, block_end_str): {layer_type: Path}}}
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
    """Sorted list of all unique (block_start, block_end) pairs across all AOIs."""
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
# 4) ELEVATION STATISTICS
# ============================================================

def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentiles) -> np.ndarray:
    """
    Area-weighted percentiles using the midpoint-convention interpolation.

    Each value occupies a weight interval; its representative quantile point
    is the midpoint of that interval normalised to [0, 1].  np.interp then
    interpolates between sorted values.

    Parameters
    ----------
    values      : 1-D array of elev_mean values (pre-filtered, no NaN)
    weights     : 1-D array of polygon areas (same length, all > 0)
    percentiles : sequence of values in [0, 100]

    Returns
    -------
    1-D array of interpolated values, one per requested percentile.
    """
    order    = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_cumsum = np.cumsum(w_sorted)
    w_total  = w_cumsum[-1]
    # Midpoint of each polygon's cumulative-weight interval, normalised to [0,1]
    quantile_points = (w_cumsum - w_sorted / 2.0) / w_total
    return np.interp(np.asarray(percentiles) / 100.0, quantile_points, v_sorted)


def compute_elev_stats(
    gdf: gpd.GeoDataFrame | None,
    aoi: str,
    layer: str,
) -> dict:
    """
    Compute area-weighted mean, max, and (for PERCENTILE_LAYERS) area-weighted
    p10/p25/p50/p75/p90 of the per-polygon elev_mean attribute.

    Returns
    -------
    dict with keys:
      elev_mean       – area-weighted mean
      elev_max        – maximum elev_mean across all polygons
      elev_p10 … p90 – area-weighted percentiles (None for footprint layer)

    All values are None when gdf is None, ELEV_COL is absent, or no valid
    polygons are present.
    """
    pct_keys = {f"elev_p{p}": None for p in PERCENTILES}
    empty    = {"elev_mean": None, "elev_max": None, **pct_keys}

    if gdf is None:
        return empty

    if ELEV_COL not in gdf.columns:
        warnings.warn(
            f"[{aoi}] '{ELEV_COL}' not found in {layer} shapefile. "
            f"Available columns: {[c for c in gdf.columns if c != 'geometry']}"
        )
        return empty

    elev_vals = pd.to_numeric(gdf[ELEV_COL], errors="coerce")
    areas     = gdf.geometry.area

    valid = elev_vals.notna() & (areas > 0)
    if not valid.any():
        return empty

    e = elev_vals[valid].to_numpy(dtype=float)
    w = areas[valid].to_numpy(dtype=float)

    weighted_mean = float(np.average(e, weights=w))
    elev_max      = float(e.max())

    pct_values = _weighted_percentile(e, w, PERCENTILES)
    pct_result = {
        f"elev_p{p}": round(float(v), 2)
        for p, v in zip(PERCENTILES, pct_values)
    }

    return {"elev_mean": round(weighted_mean, 2), "elev_max": round(elev_max, 2), **pct_result}


# ============================================================
# 5) OUTPUT COLUMN ORDER
# ============================================================

def build_output_columns() -> list:
    """
    Ordered list of all output columns (excluding block_start / block_end).

    footprint: elev_mean, elev_max
    slush/lake: elev_mean, elev_max, elev_p10, elev_p25, elev_p50, elev_p75, elev_p90
    """
    cols = []
    for aoi in AOIS:
        for layer in LAYERS:
            cols.append(f"{aoi}_{layer}_elev_mean")
            cols.append(f"{aoi}_{layer}_elev_max")
            if layer in PERCENTILE_LAYERS:
                for p in PERCENTILES:
                    cols.append(f"{aoi}_{layer}_elev_p{p}")
    return cols


# ============================================================
# 6) MAIN
# ============================================================

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Discover block shapefiles
    # ----------------------------------------------------------
    debug_print("\nDiscovering block shapefiles …")
    blocks      = discover_blocks(AOI_INPUT_DIRS)
    all_periods = union_block_periods(blocks)

    if not all_periods:
        raise SystemExit(
            "No block shapefiles found. "
            "Check INPUT_PTM_DIR and INPUT_OST_DIR."
        )
    debug_print(f"  Total unique block periods: {len(all_periods)}")

    # Startup diagnostic: inspect first available file to confirm ELEV_COL exists.
    debug_print(f"\nElevation column setting: ELEV_COL = '{ELEV_COL}'")
    _checked = False
    for _aoi in AOIS:
        for _layers in blocks[_aoi].values():
            for _ltype, _path in _layers.items():
                try:
                    _sample   = gpd.read_file(_path)
                    _non_geom = [c for c in _sample.columns if c != "geometry"]
                    debug_print(
                        f"  First file ({_aoi} / {_ltype}): {_path.name}\n"
                        f"    Attributes: {_non_geom}"
                    )
                    if ELEV_COL not in _sample.columns:
                        print(
                            f"\n*** WARNING: ELEV_COL='{ELEV_COL}' is NOT present.\n"
                            f"    Available columns: {_non_geom}\n"
                            f"    Run add_elevation_to_shapefiles.py first, or set "
                            f"ELEV_COL to the correct field name. ***\n"
                        )
                    else:
                        debug_print(f"    '{ELEV_COL}' found — proceeding.")
                    _checked = True
                except Exception:
                    pass
                if _checked:
                    break
            if _checked:
                break
        if _checked:
            break

    # ----------------------------------------------------------
    # Process each block period
    # ----------------------------------------------------------
    meta_cols  = ["block_start", "block_end"]
    extra_cols = build_output_columns()
    rows       = []

    debug_print("\nProcessing block periods …")
    n_periods = len(all_periods)

    for idx, (bs, be) in enumerate(all_periods):
        debug_print(f"\n[{idx + 1}/{n_periods}] Block {bs} – {be}")
        row: dict = {"block_start": bs, "block_end": be}

        for aoi in AOIS:
            layers_found = blocks[aoi].get((bs, be), {})

            for layer in LAYERS:
                path  = layers_found.get(layer)
                gdf   = read_block_shp(path, aoi, layer) if path else None
                stats = compute_elev_stats(gdf, aoi, layer)

                row[f"{aoi}_{layer}_elev_mean"] = stats["elev_mean"]
                row[f"{aoi}_{layer}_elev_max"]  = stats["elev_max"]

                if layer in PERCENTILE_LAYERS:
                    for p in PERCENTILES:
                        row[f"{aoi}_{layer}_elev_p{p}"] = stats[f"elev_p{p}"]

                debug_print(
                    f"  [{aoi}] {layer:10s}  "
                    f"mean={stats['elev_mean']}  "
                    f"max={stats['elev_max']}  "
                    f"p10={stats['elev_p10']}  p50={stats['elev_p50']}  p90={stats['elev_p90']}"
                )

        rows.append(row)

    # ----------------------------------------------------------
    # Build and write output CSV
    # ----------------------------------------------------------
    df = pd.DataFrame(rows, columns=meta_cols + extra_cols)
    df.to_csv(OUTPUT_CSV, index=False)

    debug_print(
        f"\nWritten: {OUTPUT_CSV}\n"
        f"  {len(df)} rows × {len(df.columns)} columns"
    )
    debug_print("\nDone.")


if __name__ == "__main__":
    main()
