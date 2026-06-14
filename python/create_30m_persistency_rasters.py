#!/usr/bin/env python3
"""
create_30m_persistency_rasters.py

Builds per-AOI lake, slush, and combined persistency rasters at 30 m resolution
from the 5-band depth-band rasters produced by add_depth_band.py.

For each AOI and each window that passes the Method B coverage filter (footprint
proportion > COVERAGE_THRESHOLD from select_by_meltzone_coverage.csv), band 1
(classification: 1=lake, 2=slush, 3=other) is accumulated pixel-by-pixel:

    lake_persistency     – count of valid windows where pixel was classified lake
    slush_persistency    – count of valid windows where pixel was classified slush
    combined_persistency – count of valid windows where pixel was lake OR slush

Pixels outside the mosaic footprint have NaN in the classification band and
contribute 0 to all counts without any explicit footprint masking.

Input rasters (from add_depth_band.py):
    Band 1 – classification   float32  1=lake, 2=slush, 3=other, NaN=no data
    Band 2 – B4_ring          float32
    Band 3 – B8_ring          float32
    Band 4 – mosaic_footprint float32  1=valid, NaN=no data
    Band 5 – px_dpt_m         float32  per-pixel depth (m), NaN outside lake pixels

Output rasters (int32, nodata=0, native CRS/resolution of input rasters):
    {prefix}_lake_persistency.tif
    {prefix}_slush_persistency.tif
    {prefix}_combined_persistency.tif
"""

import re
import logging
import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_ROOT = Path(r"Q:\ThesisData\data\raster_data\with_depth_band")
OUTPUT_DIR = Path(r"Q:\ThesisData\data\raster_data\30_persistency_rasters")

CSV_PATH = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\select_by_meltzone_coverage.csv"
)

AOI_CONFIGS = [
    {
        "prefix":         "PTM",
        "input_dir":      INPUT_ROOT / "all_PTM_classified_depth_rasters",
        "proportion_col": "PTM_proportion",
        "mask_shp":       Path(r"Q:\ThesisData\data\data_correction_steps\1_visual\PTM_intersect_polygons\PTM_intersector.shp"),
    },
    {
        "prefix":         "OST",
        "input_dir":      INPUT_ROOT / "all_OST_classified_depth_rasters",
        "proportion_col": "OST_proportion",
        "mask_shp":       None,
    },
]

# Must match the threshold used in the existing pipeline (Method B filter).
COVERAGE_THRESHOLD = 0.2

# Restrict to a single year ('YYYY') or None to process all available years.
YEAR_FILTER = None

# Classification band values (matching add_depth_band.py / export_v06.js)
LAKE_CLASS  = 1
SLUSH_CLASS = 2


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

FNAME_RE = re.compile(
    r'^(?P<aoi>[A-Z]+)_(?P<win_start>\d{4}-\d{2}-\d{2})_(?P<win_end>\d{4}-\d{2}-\d{2})\.tif$'
)


def parse_filename(path: Path):
    """Return (aoi, win_start_str, win_end_str) parsed from filename, or None."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.group('aoi'), m.group('win_start'), m.group('win_end')


def load_valid_windows(csv_path: Path, proportion_col: str) -> set:
    """
    Return a set of (win_start, win_end) string tuples that pass the
    Method B coverage filter (proportion > COVERAGE_THRESHOLD).
    Date strings are normalised to YYYY-MM-DD to match parsed filenames.
    """
    df = pd.read_csv(csv_path)
    for col in ('win_start', 'win_end'):
        if col not in df.columns:
            raise KeyError(f"Coverage CSV missing required column: '{col}'")
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')

    if proportion_col not in df.columns:
        raise KeyError(
            f"Coverage CSV missing proportion column: '{proportion_col}'\n"
            f"Available columns: {list(df.columns)}"
        )

    mask  = pd.to_numeric(df[proportion_col], errors='coerce').fillna(0) > COVERAGE_THRESHOLD
    valid = df[mask][['win_start', 'win_end']].dropna()
    return set(zip(valid['win_start'], valid['win_end']))


def _geotiff_profile(width: int, height: int, crs, transform: Affine) -> dict:
    return {
        'driver':     'GTiff',
        'dtype':      'int32',
        'width':      width,
        'height':     height,
        'count':      1,
        'crs':        crs,
        'transform':  transform,
        'compress':   'deflate',
        'tiled':      True,
        'blockxsize': 256,
        'blockysize': 256,
        'nodata':     0,
    }


# ============================================================
# ERROR MASK
# ============================================================

def build_error_mask(mask_shp: Path, ref_transform, ref_shape: tuple,
                     ref_crs) -> np.ndarray:
    """
    Rasterize error polygons onto the reference grid.
    Returns a boolean array (True = exclude pixel) aligned to ref_shape.
    The shapefile is reprojected to ref_crs before rasterization.
    """
    gdf = gpd.read_file(mask_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        log.warning(f"Error mask shapefile contains no valid geometries: {mask_shp}")
        return np.zeros(ref_shape, dtype=bool)

    burned = rasterize(
        shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype='uint8',
        all_touched=False,
    )
    return burned.astype(bool)


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_aoi(prefix: str, input_dir: Path, proportion_col: str,
                valid_windows: set, mask_shp: Path | None = None) -> None:

    tifs = sorted(input_dir.glob(f"{prefix}_*.tif"))
    if YEAR_FILTER:
        year_start = len(prefix) + 1           # index of first date char after 'AOI_'
        tifs = [t for t in tifs if t.name[year_start:year_start + 4] == YEAR_FILTER]

    if not tifs:
        log.warning(
            f"[{prefix}] No TIF files found in {input_dir}" +
            (f" for year {YEAR_FILTER}" if YEAR_FILTER else "")
        )
        return

    # Filter to windows that pass the Method B coverage threshold.
    passing = []
    for t in tifs:
        parsed = parse_filename(t)
        if parsed is None:
            log.warning(f"[{prefix}] Unexpected filename format, skipping: {t.name}")
            continue
        _, win_start, win_end = parsed
        if (win_start, win_end) in valid_windows:
            passing.append(t)

    if not passing:
        log.warning(f"[{prefix}] No rasters pass the coverage threshold.")
        return

    log.info(
        f"[{prefix}] {len(passing)} of {len(tifs)} rasters pass coverage filter."
        + (f" (year filter: {YEAR_FILTER})" if YEAR_FILTER else "")
    )

    # Establish the reference grid from the first valid raster.
    # All depth-band rasters for an AOI share the same GEE export region,
    # CRS, and scale, so they should already be on the same pixel grid.
    with rasterio.open(passing[0]) as ref:
        ref_shape     = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs       = ref.crs

    log.info(
        f"[{prefix}] Reference grid: {ref_shape[1]}×{ref_shape[0]} px  "
        f"CRS={ref_crs.to_epsg()}"
    )

    # Build error mask once against the reference grid.
    error_mask = None
    if mask_shp is not None:
        if not mask_shp.exists():
            log.error(f"[{prefix}] Error mask shapefile not found: {mask_shp}")
        else:
            error_mask = build_error_mask(mask_shp, ref_transform, ref_shape, ref_crs)
            log.info(f"[{prefix}] Error mask: {int(error_mask.sum())} pixels excluded.")

    height, width = ref_shape
    lake_pers     = np.zeros((height, width), dtype=np.int32)
    slush_pers    = np.zeros((height, width), dtype=np.int32)
    combined_pers = np.zeros((height, width), dtype=np.int32)

    n_processed = 0
    first_sig = None
    n_identical_to_first = 0
    for tif in passing:
        with rasterio.open(tif) as src:
            same_grid = (
                (src.height, src.width) == ref_shape and
                src.crs == ref_crs and
                src.transform == ref_transform
            )

            class_src = src.read(1, masked=True).filled(np.nan).astype(np.float32)

            footprint_src = None
            if src.count >= 4:
                footprint_src = src.read(4, masked=True).filled(np.nan).astype(np.float32)

            if same_grid:
                class_band = class_src
                footprint = footprint_src
            else:
                log.info(f"[{prefix}] Reprojecting to reference grid: {tif.name}")
                class_band = np.full(ref_shape, np.nan, dtype=np.float32)
                reproject(
                    source=class_src,
                    destination=class_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )

                footprint = None
                if footprint_src is not None:
                    footprint = np.full(ref_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=footprint_src,
                        destination=footprint,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        src_nodata=np.nan,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                    )

            # Enforce explicit mosaic footprint validity (band 4).
            if footprint is not None:
                class_band[~np.isclose(footprint, 1.0, atol=1e-6)] = np.nan

        if error_mask is not None:
            class_band[error_mask] = np.nan

        valid_vals = class_band[np.isfinite(class_band)]
        if valid_vals.size:
            uniq, cnts = np.unique(valid_vals.astype(np.int16), return_counts=True)
            class_summary = ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnts)])
            log.info(f"[{prefix}] {tif.name} class counts (valid px): {class_summary}")

        # Signature check: if all windows have identical class grids, persistency
        # will naturally saturate at n_processed for stable lake/slush pixels.
        sig = hashlib.sha1(np.nan_to_num(class_band, nan=-9999.0).tobytes()).hexdigest()
        if first_sig is None:
            first_sig = sig
        elif sig == first_sig:
            n_identical_to_first += 1

        lake_arr  = np.isclose(class_band, LAKE_CLASS, atol=1e-6).astype(np.int32)
        slush_arr = np.isclose(class_band, SLUSH_CLASS, atol=1e-6).astype(np.int32)

        lake_pers     += lake_arr
        slush_pers    += slush_arr
        combined_pers += ((lake_arr | slush_arr) > 0).astype(np.int32)
        n_processed   += 1

    log.info(
        f"[{prefix}] Stacked {n_processed} rasters.  "
        f"Max — lake: {lake_pers.max()}, slush: {slush_pers.max()}, "
        f"combined: {combined_pers.max()}"
    )
    if n_processed > 1 and n_identical_to_first == (n_processed - 1):
        log.warning(
            f"[{prefix}] All processed class rasters are byte-identical to the first "
            f"window after masking/alignment. This points to an upstream export/input issue, "
            f"not persistency summation."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = _geotiff_profile(width, height, ref_crs, ref_transform)

    for arr, layer in [
        (lake_pers,     'lake_persistency'),
        (slush_pers,    'slush_persistency'),
        (combined_pers, 'combined_persistency'),
    ]:
        out_path = OUTPUT_DIR / f"{prefix}_{layer}.tif"
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(arr, 1)
        log.info(f"[{prefix}] Saved → {out_path.name}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {CSV_PATH}")

    for cfg in AOI_CONFIGS:
        prefix         = cfg['prefix']
        input_dir      = cfg['input_dir']
        proportion_col = cfg['proportion_col']

        log.info('=' * 60)
        log.info(f"AOI: {prefix}")

        if not input_dir.exists():
            log.error(f"[{prefix}] Input directory not found: {input_dir}")
            continue

        valid_windows = load_valid_windows(CSV_PATH, proportion_col)
        log.info(f"[{prefix}] {len(valid_windows)} windows pass coverage filter in CSV.")

        process_aoi(prefix, input_dir, proportion_col, valid_windows,
                    mask_shp=cfg.get('mask_shp'))

    log.info('=' * 60)
    log.info('Done.')


if __name__ == '__main__':
    main()
