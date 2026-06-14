# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:11:26 2026

@author: luur4790

create_n_day_blocks.py

Aggregates per-window Landsat melt-feature shapefiles (footprint, slush,
lake) into fixed-length temporal blocks of N*2 days.

For each block of N consecutive 2-day windows:
  - Footprint polygons are dissolved into a geometry-only union (n_obs field
    records how many input polygons contributed to each output polygon).
  - Slush polygons are dissolved; elev_mn is preserved as an area-weighted
    mean of contributing input polygons, using each polygon's original area
    as the weight.
  - Lake polygons are dissolved; elev_mn, mn_dpt_m, and vol_m3 are
    preserved as area-weighted means.
  - All output polygons are tagged with blk_start and blk_end strings.
  - Outputs are written as shapefiles (.shp).

Input shapefile naming convention (flat folder, all layer types co-located):
    {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_{layer_type}.shp
    e.g. OST_2013-05-01_2013-05-02_footprint.shp

Dependencies: geopandas, shapely, pandas
"""

# ============================================================
# Configuration 
# ============================================================

N_WINDOWS = 5   # number of 2-day windows per block  (block length = N × 2 days)

INPUT_PTM_DIR  = r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_PTM_files"
INPUT_OST_DIR  = r"Q:\ThesisData\data\data_correction_steps\2_elevation\all_OST_files"
OUTPUT_PTM_DIR = r"Q:\ThesisData\data\blocks\5_window\all_PTM_files_anchored"
OUTPUT_OST_DIR = r"Q:\ThesisData\data\blocks\5_window\all_OST_files_anchored"

BLOCK_START_DATE = "2013-05-01"   # anchor start date (YYYY-MM-DD)
BLOCK_END_DATE   = None           # None → auto-detect from latest window found

# ============================================================

import logging
import re
import warnings
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


# ============================================================
# Constants
# ============================================================

# Inputs are expected in geographic coordinates (project data: EPSG:4326).
# Missing CRS metadata on input files will be interpreted as this CRS.
INPUT_CRS = "EPSG:4326"

# All area calculations and geometry operations use this projected CRS.
TARGET_CRS = "EPSG:3995"

# Regex matching the project's window shapefile naming convention:
#   {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_{layer_type}.shp
_WIN_RE = re.compile(
    r"^.+_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(footprint|slush|lake)\.shp$",
    re.IGNORECASE,
)

# Fields to area-weight-average per layer type (uses original input field names).
# Footprint is geometry-only (empty list); n_obs is added for all types.
LAYER_FIELDS: dict[str, list[str]] = {
    "footprint": [],
    "slush":     ["elev_mean"],
    "lake":      ["elev_mean", "mean_dpt_m", "volume_m3"],
}

# Abbreviated output field names (shapefile field names max 10 characters).
FIELD_RENAME = {
    "block_start": "blk_start",
    "block_end":   "blk_end",
    "elev_mean":   "elev_mean",
    "mean_dpt_m":  "mean_dpt_m",
    "volume_m3":   "volume_m3",
}

# AOI input/output pairs processed in a single run.
AOI_PAIRS = [
    (Path(INPUT_PTM_DIR), Path(OUTPUT_PTM_DIR)),
    (Path(INPUT_OST_DIR), Path(OUTPUT_OST_DIR)),
]


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Window discovery
# ============================================================

def discover_windows(
    input_dir: Path,
) -> dict[tuple[date, date], dict[str, Path]]:
    """
    Scan *input_dir* for shapefiles matching the project naming convention
    ``{AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_{layer_type}.shp``.

    Returns
    -------
    dict keyed by (win_start, win_end) date tuples.
    Each value is a dict mapping layer_type -> Path.
    """
    windows: dict[tuple[date, date], dict[str, Path]] = {}

    for shp in sorted(input_dir.glob("*.shp")):
        m = _WIN_RE.match(shp.name)
        if not m:
            continue
        try:
            d1 = date.fromisoformat(m.group(1))
            d2 = date.fromisoformat(m.group(2))
        except ValueError:
            log.warning("Cannot parse dates from filename: %s", shp.name)
            continue
        layer = m.group(3).lower()
        windows.setdefault((d1, d2), {})[layer] = shp

    log.info("Discovered %d window(s) in %s", len(windows), input_dir)
    return windows


# ============================================================
# Block construction
# ============================================================

def build_blocks(
    windows: dict[tuple[date, date], dict[str, Path]],
    start: date,
    end: date,
    n_windows: int,
) -> list[dict]:
    """
    Group windows into fixed-length blocks anchored to *start* each year.

    For every year in [start.year, end.year], block construction restarts at
    the same month/day as *start*. This makes the first block of each year
    consistent (e.g. 05-01 when start="2013-05-01").

    Within each year, block boundaries are non-overlapping and each block is
    N*2 days long:
        Block 0 : [year_anchor,              year_anchor + N*2 - 1]
        Block 1 : [year_anchor + N*2,        year_anchor + N*4 - 1]
        ...

    A window is assigned to the block whose date range contains the
    window's start date.

    Returns a list of block dicts, each containing:
        block_start : date
        block_end   : date  (inclusive last day)
        windows     : sorted list of (win_start, win_end) tuples
        files       : dict  layer_type -> list[Path]
    """
    block_len = timedelta(days=n_windows * 2)
    blocks: list[dict] = []
    anchor_month = start.month
    anchor_day = start.day

    for year in range(start.year, end.year + 1):
        year_anchor = date(year, anchor_month, anchor_day)
        if year_anchor > end:
            break

        block_start = year_anchor
        year_end = min(end, date(year, 12, 31))

        while block_start <= year_end:
            block_end = block_start + block_len - timedelta(days=1)

            block_wins = sorted(
                key for key in windows
                if block_start <= key[0] <= block_end
            )

            if block_wins:
                layer_paths: dict[str, list[Path]] = {}
                for key in block_wins:
                    for layer, path in windows[key].items():
                        layer_paths.setdefault(layer, []).append(path)

                blocks.append({
                    "block_start": block_start,
                    "block_end":   block_end,
                    "windows":     block_wins,
                    "files":       layer_paths,
                })
                log.info(
                    "Block %s – %s: %d window(s) [%s]",
                    block_start, block_end, len(block_wins),
                    ", ".join(str(w[0]) for w in block_wins),
                )

            block_start += block_len

    return blocks


# ============================================================
# Geometry helpers
# ============================================================

def _clean(geom):
    """Return buffer(0)-cleaned geometry, or None if empty/invalid."""
    if geom is None or geom.is_empty:
        return None
    try:
        g = geom.buffer(0)
        return g if not g.is_empty else None
    except Exception:
        return None


def _explode(geom) -> list[Polygon]:
    """Split any geometry into a flat list of Polygon parts."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    # GeometryCollection or other compound type
    parts: list[Polygon] = []
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            parts.extend(_explode(g))
    return parts


# ============================================================
# Area-weighted attribute merge
# ============================================================

def _area_weighted_attrs(
    output_poly: Polygon,
    input_gdf: gpd.GeoDataFrame,
    sindex,
    fields: list[str],
) -> dict:
    """
    For one output polygon, find intersecting input polygons via the
    spatial index and compute area-weighted means of *fields*.

    The weight for each contributing input polygon is its pre-computed
    original area stored in column ``_orig_area_m2``.

    Returns a dict with keys: n_obs, and one key per field in *fields*.
    """
    # Bounding-box candidates from spatial index
    candidate_idx = list(sindex.intersection(output_poly.bounds))
    if not candidate_idx:
        return {"n_obs": 0} | {f: None for f in fields}

    candidates = input_gdf.iloc[candidate_idx]
    mask_intersect = candidates.geometry.intersects(output_poly)
    intersecting = candidates.loc[mask_intersect]

    n_obs = len(intersecting)
    if n_obs == 0:
        return {"n_obs": 0} | {f: None for f in fields}

    result: dict = {"n_obs": n_obs}

    for field in fields:
        if field not in intersecting.columns:
            result[field] = None
            continue

        vals = pd.to_numeric(intersecting[field], errors="coerce")
        weights = intersecting["_orig_area_m2"]

        # Only rows where both value and weight are finite and positive
        valid = vals.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            result[field] = None
        else:
            v = vals[valid]
            w = weights[valid]
            result[field] = float((v * w).sum() / w.sum())

    return result


# ============================================================
# Per-layer block processing
# ============================================================

def process_layer_block(
    paths: list[Path],
    layer_type: str,
    block_start: date,
    block_end: date,
) -> gpd.GeoDataFrame | None:
    """
    Merge shapefiles for one *layer_type* within one block.

    Algorithm
    ---------
    1. Read and concatenate all input GeoDataFrames.
    2. Reproject to TARGET_CRS.
    3. Clean geometries with buffer(0); drop empties.
    4. Record each polygon's original area for use as weighting factor.
    5. Dissolve all polygons via unary_union.
    6. Explode union result into individual contiguous Polygon parts.
    7. For each output polygon, find intersecting inputs (spatial index)
       and compute area-weighted means of the layer's attribute fields.
    8. Tag every output polygon with block_start, block_end, n_obs.

    Returns a GeoDataFrame in TARGET_CRS, or None if no valid output.
    """
    fields = LAYER_FIELDS[layer_type]

    # ----------------------------------------------------------
    # Step 1 – Read and concatenate
    # ----------------------------------------------------------
    gdfs: list[gpd.GeoDataFrame] = []
    for p in paths:
        try:
            gdf = gpd.read_file(p)
            if gdf.empty:
                continue

            if gdf.crs is None:
                log.warning(
                    "  [%s] %s has no CRS; assuming %s.",
                    layer_type, p.name, INPUT_CRS
                )
                gdf = gdf.set_crs(INPUT_CRS)

            if str(gdf.crs) != TARGET_CRS:
                gdf = gdf.to_crs(TARGET_CRS)

            gdfs.append(gdf)
        except Exception as exc:
            log.warning("Cannot read %s: %s", p.name, exc)

    if not gdfs:
        log.warning(
            "  [%s] No readable files in block %s – %s",
            layer_type, block_start, block_end,
        )
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True),
            crs=TARGET_CRS,
        )

    if combined.empty:
        return None

    # ----------------------------------------------------------
    # Step 2 – CRS already normalized per file above
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Step 3 – Clean geometries
    # ----------------------------------------------------------
    combined["geometry"] = combined["geometry"].apply(_clean)
    combined = combined[
        combined["geometry"].notna() & ~combined["geometry"].is_empty
    ].copy()

    if combined.empty:
        log.warning(
            "  [%s] All geometries invalid after buffer(0) in block %s – %s",
            layer_type, block_start, block_end,
        )
        return None

    # ----------------------------------------------------------
    # Step 4 – Original area for weighting
    # ----------------------------------------------------------
    combined["_orig_area_m2"] = combined.geometry.area

    # ----------------------------------------------------------
    # Step 5 – Union
    # ----------------------------------------------------------
    union_geom = unary_union(combined.geometry.tolist())
    if union_geom is None or union_geom.is_empty:
        return None

    # ----------------------------------------------------------
    # Step 6 – Explode to individual polygons
    # ----------------------------------------------------------
    parts = _explode(union_geom)
    if not parts:
        return None

    log.info(
        "  [%s] %d input polygon(s) → %d output polygon(s)",
        layer_type, len(combined), len(parts),
    )

    # ----------------------------------------------------------
    # Step 7 – Area-weighted attribute means (all layer types,
    #           footprint has fields=[] so only n_obs is computed)
    # ----------------------------------------------------------
    sindex = combined.sindex
    rows: list[dict] = []
    for poly in parts:
        attrs = _area_weighted_attrs(poly, combined, sindex, fields)
        rows.append({"geometry": poly, **attrs})

    # ----------------------------------------------------------
    # Step 8 – Assemble output GeoDataFrame
    # ----------------------------------------------------------
    out = gpd.GeoDataFrame(rows, crs=TARGET_CRS)
    out["block_start"] = block_start.isoformat()
    out["block_end"]   = block_end.isoformat()

    # Column order: block_start, block_end, n_obs, [value fields], geometry
    col_order = ["block_start", "block_end", "n_obs"] + fields + ["geometry"]
    out = out.reindex(columns=col_order)

    # Rename to abbreviated field names for shapefile compatibility
    out = out.rename(columns=FIELD_RENAME)

    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    start = date.fromisoformat(BLOCK_START_DATE)
    end   = date.fromisoformat(BLOCK_END_DATE) if BLOCK_END_DATE else None

    if N_WINDOWS < 1:
        log.error("N_WINDOWS must be >= 1")
        raise SystemExit(1)

    for input_dir, output_dir in AOI_PAIRS:
        log.info("=" * 60)
        log.info("AOI input  : %s", input_dir)
        log.info("AOI output : %s", output_dir)
        log.info("=" * 60)

        if not input_dir.exists():
            log.error("Input directory not found: %s", input_dir)
            raise SystemExit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------
        # Discover windows
        # ----------------------------------------------------------
        windows = discover_windows(input_dir)
        if not windows:
            log.error(
                "No window shapefiles found in %s  "
                "(expected names like  OST_2013-05-01_2013-05-02_footprint.shp)",
                input_dir,
            )
            raise SystemExit(1)

        # Resolve end date
        aoi_end = end
        if aoi_end is None:
            aoi_end = max(key[0] for key in windows)
            log.info("BLOCK_END_DATE not set; using latest window start: %s", aoi_end)

        # ----------------------------------------------------------
        # Build blocks
        # ----------------------------------------------------------
        log.info(
            "Building blocks of %d window(s) each (%d days) anchored to %s",
            N_WINDOWS, N_WINDOWS * 2, start,
        )
        blocks = build_blocks(windows, start, aoi_end, N_WINDOWS)

        if not blocks:
            log.warning(
                "No blocks produced — no windows in the date range %s – %s",
                start, aoi_end,
            )
            continue

        log.info("Total blocks to process: %d", len(blocks))

        # ----------------------------------------------------------
        # Process each block
        # ----------------------------------------------------------
        for block in blocks:
            bs: date = block["block_start"]
            be: date = block["block_end"]
            n_wins = len(block["windows"])

            log.info("=== Block %s – %s  (%d window(s)) ===", bs, be, n_wins)

            for layer_type in ("footprint", "slush", "lake"):
                paths: list[Path] = block["files"].get(layer_type, [])
                if not paths:
                    log.info("  [%s] No files in this block.", layer_type)
                    continue

                out_gdf = process_layer_block(paths, layer_type, bs, be)
                if out_gdf is None or out_gdf.empty:
                    log.info("  [%s] Empty output — skipping.", layer_type)
                    continue

                out_name = f"{bs}_{be}_{layer_type}.shp"
                out_path = output_dir / out_name
                out_gdf.to_file(out_path, driver="ESRI Shapefile")
                log.info(
                    "  [%s] Wrote %d polygon(s) → %s",
                    layer_type, len(out_gdf), out_name,
                )

    log.info("Done.")


if __name__ == "__main__":
    main()
