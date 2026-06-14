#!/usr/bin/env python3
"""
add_depth_band.py

Adds a per-pixel lake depth band (px_dpt_m) to each classified raster,
replicating the Pope et al. (2016) two-band depth model from volume_v06.js.

Depth model (per pixel, lake class only):
    z_red = (ln(Ad_red - Rinf_red) - ln(Rw_red - Rinf_red)) / g_red
    z_pan = (ln(Ad_pan - Rinf_pan) - ln(Rw_pan - Rinf_pan)) / g_pan
    depth = (z_red + z_pan) / 2, clamped to [0, 20] m

Ad (bottom reflectance) = mean of B4_ring / B8_ring in the 1-pixel ring
    around each connected lake region (matches per-lake buffer in volume_v06.js).

Rinf (deep-water asymptote) = matched from Rinf LUT CSV by closest scene
    date to each raster's window_start, filtering to ok==1 rows only.

Input raster bands (from export_v06.js):
    1 – classification    int8   1=lake, 2=slush, 3=other
    2 – B4_ring           float  red TOA reflectance, lake + 1-px ring
    3 – B8_ring           float  pan TOA reflectance (30 m), lake + 1-px ring
    4 – mosaic_footprint  byte   1 where valid observation exists

Output: same raster + band 5 (px_dpt_m, float32, NaN outside lake pixels).
Note: all bands are written as float32 in the output; classification values
      1/2/3 and footprint 0/1 are preserved exactly as float32.
"""

import re
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label, binary_dilation


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_ROOT  = Path(r"Q:\ThesisData\data\raster_data\raw_classified_layers")
OUTPUT_ROOT = Path(r"Q:\ThesisData\data\raster_data\with_depth_band")
LUT_DIR     = Path(r"Q:\ThesisData\data\deepwater_rinf_lookuptables")

AOI_KEYS = ['PTM']           # set to ['OST', 'PTM'] for full run

# Restrict processing to a single year (string 'YYYY') or None for all years.
YEAR_FILTER = '2025'

LUT_FILENAMES = {
    'OST': 'Rinf_LUT_OST_L8L9_v1.csv',
    'PTM': 'Rinf_LUT_PTM_L8L9_v1.csv',
}

INPUT_DIR_TEMPLATE  = "all_{aoi}_classified_rasters"
OUTPUT_DIR_TEMPLATE = "all_{aoi}_classified_depth_rasters"

# Depth model constants (Pope et al. 2016) — mirrors volume_v06.js
G_RED     = 0.80
G_PAN     = 0.36
EPS       = 1e-6
MIN_DEPTH = 0.0
MAX_DEPTH = 20.0

WATER_CLASS   = 1    # classification band value for lake pixels
MAX_DIFF_DAYS = 30   # max days between raster window_start and matched LUT row

# 8-connected structuring element for 1-pixel ring dilation
DILATE_STRUCT = np.ones((3, 3), dtype=bool)


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
# FILENAME PARSING
# ============================================================

FNAME_RE = re.compile(
    r'^(?P<aoi>[A-Z]+)_(?P<win_start>\d{4}-\d{2}-\d{2})_(?P<win_end>\d{4}-\d{2}-\d{2})\.tif$'
)


def parse_filename(path: Path):
    """Return (aoi, win_start_date, win_end_date) parsed from filename, or None."""
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return (
        m.group('aoi'),
        datetime.strptime(m.group('win_start'), '%Y-%m-%d'),
        datetime.strptime(m.group('win_end'),   '%Y-%m-%d'),
    )


# ============================================================
# RINF LUT HELPERS
# ============================================================

def load_lut(lut_path: Path) -> pd.DataFrame:
    """Load CSV, keep ok==1 rows, parse date column."""
    df = pd.read_csv(lut_path)
    log.info(f"LUT columns: {list(df.columns)}")

    # Normalise column names: strip whitespace, map common GEE export variants
    df.columns = df.columns.str.strip()
    col_map = {c: c for c in df.columns}
    for col in df.columns:
        lower = col.lower()
        if lower in ('rinf_red', 'rinfinity_red', 'rinf_r'):
            col_map[col] = 'Rinf_red'
        elif lower in ('rinf_pan', 'rinfinity_pan', 'rinf_p'):
            col_map[col] = 'Rinf_pan'
    df = df.rename(columns=col_map)

    required = {'ok', 'date', 'Rinf_red', 'Rinf_pan'}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Expected columns not found in {lut_path.name}: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df = df[df['ok'] == 1].copy()
    df = df.dropna(subset=['Rinf_red', 'Rinf_pan'])
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.sort_values('date_dt').reset_index(drop=True)
    log.info(f"LUT loaded: {len(df)} ok==1 rows  ({lut_path.name})")
    return df


def match_rinf(lut_df: pd.DataFrame, win_start: datetime):
    """
    Return (Rinf_red, Rinf_pan) for the LUT row closest to win_start.
    Raises ValueError if the closest row exceeds MAX_DIFF_DAYS.
    Mirrors attachClosestLut() in volume_v06.js.
    """
    diffs = (lut_df['date_dt'] - pd.Timestamp(win_start)).abs()
    idx   = diffs.idxmin()
    best  = diffs[idx]
    if best > pd.Timedelta(days=MAX_DIFF_DAYS):
        raise ValueError(
            f"Closest LUT row is {best.days} days from {win_start.date()} "
            f"(limit {MAX_DIFF_DAYS} days)"
        )
    row = lut_df.loc[idx]
    log.info(
        f"  Rinf match: {row['date']}  diff={best.days}d  "
        f"Rinf_red={row['Rinf_red']:.5f}  Rinf_pan={row['Rinf_pan']:.5f}"
    )
    return float(row['Rinf_red']), float(row['Rinf_pan'])


# ============================================================
# DEPTH MODEL
# ============================================================

def depth_from_band(Rw: np.ndarray, Ad: np.ndarray, Rinf: float, g: float) -> np.ndarray:
    """
    Per-pixel depth for one band (Pope et al. 2016).
    Validity conditions (mirrors depthFromBand() in volume_v06.js):
        Rw  > Rinf + EPS
        Ad  > Rinf + EPS
        Ad  > Rw   + EPS
    Returns float32 array; NaN where conditions are not met or inputs are NaN.
    """
    valid = (
        np.isfinite(Rw) & np.isfinite(Ad) &
        (Rw > Rinf + EPS) &
        (Ad > Rinf + EPS) &
        (Ad > Rw   + EPS)
    )
    z = np.full(Rw.shape, np.nan, dtype=np.float32)
    z[valid] = (np.log(Ad[valid] - Rinf) - np.log(Rw[valid] - Rinf)) / g
    return z


# ============================================================
# PER-LAKE Ad COMPUTATION
# ============================================================

def compute_ad_images(
    lake_mask: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
) -> tuple:
    """
    For each connected lake region, compute Ad_red and Ad_pan as the mean of
    B4_ring / B8_ring in the 1-pixel ring around the region, then paint those
    scalar values back onto the lake pixels.

    Mirrors the per-lake buffer + reduceRegion workflow in volume_v06.js.
    Single-pixel lakes (< 2 px) are skipped (matches GEE filter count >= 2).

    Returns (Ad_red_img, Ad_pan_img) as float32 arrays, NaN where undefined.
    """
    labeled, n_lakes = label(lake_mask, structure=DILATE_STRUCT)

    Ad_red_img = np.full(lake_mask.shape, np.nan, dtype=np.float32)
    Ad_pan_img = np.full(lake_mask.shape, np.nan, dtype=np.float32)

    for lake_id in range(1, n_lakes + 1):
        this_lake = labeled == lake_id
        if this_lake.sum() < 2:
            continue

        ring = binary_dilation(this_lake, structure=DILATE_STRUCT) & ~this_lake

        ring_b4 = b4[ring & np.isfinite(b4)]
        ring_b8 = b8[ring & np.isfinite(b8)]

        Ad_red = float(np.mean(ring_b4)) if ring_b4.size > 0 else np.nan
        Ad_pan = float(np.mean(ring_b8)) if ring_b8.size > 0 else np.nan

        Ad_red_img[this_lake] = Ad_red
        Ad_pan_img[this_lake] = Ad_pan

    return Ad_red_img, Ad_pan_img


# ============================================================
# PER-RASTER PROCESSING
# ============================================================

def process_raster(src_path: Path, dst_path: Path, lut_df: pd.DataFrame) -> bool:
    """
    Read one classified raster, compute px_dpt_m depth band, write output.
    Returns True on success, False if skipped.
    """
    parsed = parse_filename(src_path)
    if parsed is None:
        log.warning(f"Skipping — unexpected filename: {src_path.name}")
        return False

    aoi, win_start, win_end = parsed

    try:
        Rinf_red, Rinf_pan = match_rinf(lut_df, win_start)
    except ValueError as exc:
        log.warning(f"Skipping {src_path.name}: {exc}")
        return False

    with rasterio.open(src_path) as src:
        profile   = src.profile.copy()
        meta_tags = src.tags()
        src_nodata = src.nodata

        # Band 1: classification  Band 2: B4_ring  Band 3: B8_ring
        # Read as float32 — nodata may be NaN; NaN == WATER_CLASS is False so
        # masked pixels are safely excluded from the lake mask.
        class_band = src.read(1).astype(np.float32)
        b4         = src.read(2).astype(np.float32)
        b8         = src.read(3).astype(np.float32)

    # Replace source nodata with NaN for arithmetic
    if src_nodata is not None:
        b4[b4 == src_nodata] = np.nan
        b8[b8 == src_nodata] = np.nan

    lake_mask = class_band == WATER_CLASS
    n_lake_px = int(lake_mask.sum())
    log.info(f"  Lake pixels: {n_lake_px}")

    # Per-lake Ad from 1-pixel ring
    Ad_red_img, Ad_pan_img = compute_ad_images(lake_mask, b4, b8)

    # Rw: only lake pixels contribute to depth
    Rw_red = np.where(lake_mask, b4, np.nan).astype(np.float32)
    Rw_pan = np.where(lake_mask, b8, np.nan).astype(np.float32)

    # Depth per band, then average (NaN propagates if either band is invalid)
    z_red = depth_from_band(Rw_red, Ad_red_img, Rinf_red, G_RED)
    z_pan = depth_from_band(Rw_pan, Ad_pan_img, Rinf_pan, G_PAN)

    depth = (z_red + z_pan) / 2.0
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)  # NaN remains NaN through clip
    depth[~lake_mask] = np.nan

    valid_depth_px = int(np.isfinite(depth).sum())
    log.info(f"  Valid depth pixels: {valid_depth_px} / {n_lake_px}")

    # ----------------------------------------------------------
    # Write output: original 4 bands (as float32) + depth band
    # All bands promoted to float32; classification values 1/2/3
    # and footprint 0/1 are exactly representable.
    # ----------------------------------------------------------
    out_profile = profile.copy()
    out_profile.update(
        count=profile['count'] + 1,
        dtype='float32',
        nodata=float('nan'),
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, 'w', **out_profile) as dst:
        with rasterio.open(src_path) as src:
            for band_idx in range(1, profile['count'] + 1):
                data = src.read(band_idx).astype(np.float32)
                if src_nodata is not None:
                    data[data == src_nodata] = np.nan
                dst.write(data, band_idx)

        dst.write(depth, profile['count'] + 1)

        dst.update_tags(**meta_tags)
        dst.update_tags(
            win_start=win_start.strftime('%Y-%m-%d'),
            win_end=win_end.strftime('%Y-%m-%d'),
            depth_band_index=str(profile['count'] + 1),
            depth_band_name='px_dpt_m',
            depth_Rinf_red=str(Rinf_red),
            depth_Rinf_pan=str(Rinf_pan),
            depth_model='Pope et al. 2016, g_red=0.80, g_pan=0.36',
        )

    log.info(f"  Written: {dst_path}")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    n_ok   = 0
    n_skip = 0

    for aoi in AOI_KEYS:
        log.info(f"{'='*60}")
        log.info(f"AOI: {aoi}")

        lut_path = LUT_DIR / LUT_FILENAMES[aoi]
        if not lut_path.exists():
            log.error(f"LUT not found: {lut_path}")
            continue

        lut_df = load_lut(lut_path)

        in_dir  = INPUT_ROOT  / INPUT_DIR_TEMPLATE.format(aoi=aoi)
        out_dir = OUTPUT_ROOT / OUTPUT_DIR_TEMPLATE.format(aoi=aoi)

        tifs = sorted(in_dir.glob(f"{aoi}_*.tif"))
        if YEAR_FILTER:
            tifs = [t for t in tifs if t.name[len(aoi)+1:len(aoi)+5] == YEAR_FILTER]
        if not tifs:
            log.warning(f"No TIF files found in {in_dir}" +
                        (f" for year {YEAR_FILTER}" if YEAR_FILTER else ""))
            continue

        log.info(f"Found {len(tifs)} rasters" +
                 (f" for {YEAR_FILTER}" if YEAR_FILTER else "") +
                 f" in {in_dir}")

        for tif in tifs:
            log.info(f"Processing {tif.name}")
            dst = out_dir / tif.name
            if process_raster(tif, dst, lut_df):
                n_ok += 1
            else:
                n_skip += 1

    log.info(f"{'='*60}")
    log.info(f"Done.  Processed: {n_ok}   Skipped: {n_skip}")


if __name__ == '__main__':
    main()
