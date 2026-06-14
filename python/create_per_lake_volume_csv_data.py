# -*- coding: utf-8 -*-
"""
create_per_lake_volume_csv_data.py

Computes per-lake metrics for every two-day window, using:
  - Per-window footprint and lake rasters from rasterize_window_shapefiles.py
  - Persistent lake polygon shapefiles from create_persistent_lake_vectors.py

Output CSVs  (same row / column layout for all four files)
-----------------------------------------------------------
  per_lake_volume.csv       volume [m³]       Σ(depth_m × pixel_area_m²) over obs lake pixels
  per_lake_mean_depth.csv   mean depth [m]    mean of Band 2 values over obs lake pixels
  per_lake_obs_pixels.csv   obs pixel count   number of lake-present pixels inside polygon
  per_lake_obs_fraction.csv obs area fraction obs_pixels / total polygon pixels  (0 – 1)

Columns : win_start, win_end, P001, P002, … (PTM lakes), C001, C002, … (OST lakes)
One row per window from all_windows_template.csv.

Cell logic (applied identically to all four outputs)
-----------------------------------------------------
  footprint coverage of the lake polygon < 99 %   →  blank (NaN) in all four CSVs
  footprint coverage ≥ 99 % and no lake pixels     →  0.0 in all four CSVs
  footprint coverage ≥ 99 % and lake pixels exist  →  computed value

"Footprint coverage" for a lake is defined as:
    (pixels inside the persistent lake polygon covered by the window footprint raster)
    / (total pixels inside the persistent lake polygon)

Volume formula
--------------
  volume_m3 = pixel_area_m2 × Σ depth_m[px]
  where the sum is over pixels that are:
    (a) inside the persistent lake polygon  AND
    (b) flagged as lake-present in Band 1 of the window's lake raster

  Because depth_m (Band 2) is a uniform field (same mean_dpt_m value for every
  pixel in the lake observation), this is equivalent to:
    volume_m3 = mean_dpt_m × obs_pixels × pixel_area_m2

depth_m is taken from Band 2 of the lake raster (mean_dpt_m attribute, nodata = 0).
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


# ============================================================
# 1) USER SETTINGS
# ============================================================

TEMPLATE_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_templates\all_windows_template.csv"
)

OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events"
)

# Minimum footprint coverage fraction (0–1) for a lake cell to be computed.
# Windows / lakes with coverage below this threshold are left blank (NaN).
COVERAGE_THRESHOLD = 0.99

RESOLUTION_M  = 30              # pixel edge length [m], must match rasterize_window_shapefiles.py
PIXEL_AREA_M2 = RESOLUTION_M ** 2   # 8 100 m²

TARGET_CRS = "EPSG:3413"

# PTM first so PTM lake columns (P001, P002, …) precede OST columns (C001, C002, …).   
# r"Q:\ThesisData\data\drainage\persistent_lakes_30m_converted"

AOI_CONFIGS = [
    {
        "prefix":     "PTM",
        "lake_shp":   Path(r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles\PTM_persistent_lake_polygons.shp"),
        "raster_dir": Path(r"Q:\ThesisData\data\raster_data\converted_shapefiles\all_30m_ptm_converted_rasters"),
    },
    {
        "prefix":     "OST",
        "lake_shp":   Path(r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles\OST_persistent_lake_polygons.shp"),
        "raster_dir": Path(r"Q:\ThesisData\data\raster_data\converted_shapefiles\all_30m_OST_converted_rasters"),
    },
]

DEBUG = True


# ============================================================
# 2) HELPERS
# ============================================================

def debug_print(*args):
    if DEBUG:
        print(*args)


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce win_start / win_end to ISO YYYY-MM-DD strings."""
    out = df.copy()
    for col in ("win_start", "win_end"):
        if col not in out.columns:
            raise KeyError(f"CSV is missing required column: {col}")
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


# ============================================================
# 3) REFERENCE GRID & LAKE MASK PREPARATION
# ============================================================

def load_reference_grid(raster_dir: Path, prefix: str):
    """
    Return (transform, (height, width)) from the AOI reference raster.
    Falls back to the first available footprint raster if the reference file
    is absent (e.g. if rasterize_window_shapefiles.py was not run with that flag).
    """
    ref_path = raster_dir / f"{prefix}_AOI_reference.tif"
    if ref_path.exists():
        with rasterio.open(ref_path) as src:
            return src.transform, (src.height, src.width)

    for tif in sorted(raster_dir.glob(f"converted_{prefix}_*_footprint.tif")):
        with rasterio.open(tif) as src:
            debug_print(f"  [{prefix}] Reference grid inferred from {tif.name}")
            return src.transform, (src.height, src.width)

    raise FileNotFoundError(
        f"[{prefix}] No reference raster or footprint raster found in {raster_dir}.\n"
        "Run rasterize_window_shapefiles.py first."
    )


def build_lake_masks(
    lake_shp: Path,
    prefix: str,
    transform,
    shape: tuple,
) -> tuple[list, dict, dict]:
    """
    Rasterize each persistent lake polygon onto the AOI reference grid.

    Returns
    -------
    lake_ids  : list of lake_id strings, sorted (e.g. ['P001', 'P002', …])
    masks     : dict  lake_id -> boolean ndarray (height, width)
                True where a pixel centre falls inside the polygon.
    px_totals : dict  lake_id -> int  (total pixel count inside polygon)
    """
    if not lake_shp.exists():
        raise FileNotFoundError(
            f"[{prefix}] Persistent lake shapefile not found: {lake_shp}\n"
            "Run create_persistent_lake_vectors.py first."
        )

    gdf = gpd.read_file(lake_shp)
    if gdf.empty:
        warnings.warn(f"[{prefix}] Lake shapefile is empty: {lake_shp}")
        return [], {}, {}

    if "lake_id" not in gdf.columns:
        raise KeyError(
            f"[{prefix}] Lake shapefile missing 'lake_id' column: {lake_shp}"
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS.upper():
        gdf = gdf.to_crs(TARGET_CRS)

    height, width = shape
    lake_ids  = sorted(gdf["lake_id"].tolist())
    masks     = {}
    px_totals = {}

    for _, row in gdf.iterrows():
        lid  = row["lake_id"]
        geom = row.geometry
        if geom is None or geom.is_empty:
            warnings.warn(f"[{prefix}] Lake {lid} has empty geometry — skipping.")
            continue

        arr = rasterize(
            [(geom.__geo_interface__, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mask     = arr.astype(bool)
        total_px = int(mask.sum())

        if total_px == 0:
            warnings.warn(
                f"[{prefix}] Lake {lid}: polygon rasterized to 0 pixels at "
                f"{RESOLUTION_M} m resolution — skipping."
            )
            continue

        masks[lid]     = mask
        px_totals[lid] = total_px

    valid_ids = [lid for lid in lake_ids if lid in masks]
    debug_print(
        f"[{prefix}] {len(valid_ids)} lake(s) rasterized from {lake_shp.name}"
    )
    return valid_ids, masks, px_totals


# ============================================================
# 4) PER-WINDOW METRIC COMPUTATION
# ============================================================

def compute_lake_metrics_for_window(
    prefix: str,
    win_start: str,
    win_end: str,
    raster_dir: Path,
    lake_ids: list,
    masks: dict,
    px_totals: dict,
) -> tuple[dict, dict, dict, dict]:
    """
    Compute all four per-lake metrics for one AOI / window.

    Returns (volumes, mean_depths, obs_pixels, obs_fractions)
    ---------------------------------------------------------
    Each is a dict  lake_id -> float (or np.nan where coverage is insufficient).

    All lakes default to NaN.  A cell is filled only when footprint coverage
    of the lake polygon meets or exceeds COVERAGE_THRESHOLD.

      no lake pixels detected  →  0.0 in all four dicts
      lake pixels detected     →
        volumes      = Σ(depth_m[px]) × PIXEL_AREA_M2        [m³]
        mean_depths  = mean(depth_m[px])                      [m]
        obs_pixels   = count of lake-present pixels inside polygon
        obs_fractions = obs_pixels / total polygon pixels     [0–1]
    """
    nan_result = {lid: np.nan for lid in lake_ids}

    stem    = f"converted_{prefix}_{win_start}_{win_end}"
    fp_path = raster_dir / f"{stem}_footprint.tif"
    lk_path = raster_dir / f"{stem}_lake.tif"

    if not fp_path.exists():
        return nan_result, nan_result.copy(), nan_result.copy(), nan_result.copy()

    with rasterio.open(fp_path) as fp_src:
        fp_arr = fp_src.read(1)  # uint8: 1 = footprint, 0 = background

    # Lake raster: Band 1 = presence (1.0 = lake), Band 2 = mean depth [m] (nodata = 0)
    lk_b1 = lk_b2 = None
    if lk_path.exists():
        with rasterio.open(lk_path) as lk_src:
            lk_b1 = lk_src.read(1).astype(np.float32)
            lk_b2 = lk_src.read(2).astype(np.float32)

    fp_bool = fp_arr == 1

    volumes       = {lid: np.nan for lid in lake_ids}
    mean_depths   = {lid: np.nan for lid in lake_ids}
    obs_pixels    = {lid: np.nan for lid in lake_ids}
    obs_fractions = {lid: np.nan for lid in lake_ids}

    for lid in lake_ids:
        if lid not in masks:
            continue

        mask     = masks[lid]
        total_px = px_totals[lid]

        covered_px = int((mask & fp_bool).sum())
        coverage   = covered_px / total_px

        if coverage < COVERAGE_THRESHOLD:
            continue   # insufficient coverage → stay NaN

        if lk_b1 is None:
            continue   # lake raster missing → stay NaN

        lake_px   = mask & (lk_b1 > 0)
        n_lake_px = int(lake_px.sum())

        if n_lake_px == 0:
            volumes[lid]       = 0.0
            mean_depths[lid]   = 0.0
            obs_pixels[lid]    = 0.0
            obs_fractions[lid] = 0.0
        else:
            depth_vals         = lk_b2[lake_px]
            volumes[lid]       = round(float(depth_vals.sum()) * PIXEL_AREA_M2, 2)
            mean_depths[lid]   = round(float(depth_vals.mean()), 4)
            obs_pixels[lid]    = float(n_lake_px)
            obs_fractions[lid] = round(n_lake_px / total_px, 6)

    return volumes, mean_depths, obs_pixels, obs_fractions


# ============================================================
# 5) MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Load window template
    # --------------------------------------------------------
    if not TEMPLATE_CSV.exists():
        raise FileNotFoundError(f"Template CSV not found: {TEMPLATE_CSV}")

    template_df = normalize_date_columns(pd.read_csv(TEMPLATE_CSV))
    debug_print(f"Template windows loaded: {len(template_df)}")

    # --------------------------------------------------------
    # Build per-AOI lake masks (done once, reused for every window)
    # --------------------------------------------------------
    aoi_data = {}

    for cfg in AOI_CONFIGS:
        prefix     = cfg["prefix"]
        raster_dir = cfg["raster_dir"]

        debug_print(f"\n[{prefix}] Loading reference grid …")
        transform, shape = load_reference_grid(raster_dir, prefix)

        debug_print(f"[{prefix}] Rasterizing persistent lake polygons …")
        lake_ids, masks, px_totals = build_lake_masks(
            cfg["lake_shp"], prefix, transform, shape
        )

        aoi_data[prefix] = {
            "lake_ids":   lake_ids,
            "masks":      masks,
            "px_totals":  px_totals,
            "raster_dir": raster_dir,
        }

    # --------------------------------------------------------
    # Column order: PTM lakes first, then OST lakes
    # --------------------------------------------------------
    ptm_ids      = aoi_data.get("PTM", {}).get("lake_ids", [])
    ost_ids      = aoi_data.get("OST", {}).get("lake_ids", [])
    all_lake_ids = ptm_ids + ost_ids

    debug_print(
        f"\nPTM lake columns ({len(ptm_ids)}): "
        f"{ptm_ids[:6]}{'…' if len(ptm_ids) > 6 else ''}"
    )
    debug_print(
        f"OST lake columns ({len(ost_ids)}): "
        f"{ost_ids[:6]}{'…' if len(ost_ids) > 6 else ''}"
    )

    # --------------------------------------------------------
    # Accumulate results — one matrix per metric
    # --------------------------------------------------------
    n_rows  = len(template_df)
    n_lakes = len(all_lake_ids)

    vol_matrix      = np.full((n_rows, n_lakes), np.nan, dtype=np.float64)
    depth_matrix    = np.full((n_rows, n_lakes), np.nan, dtype=np.float64)
    px_matrix       = np.full((n_rows, n_lakes), np.nan, dtype=np.float64)
    frac_matrix     = np.full((n_rows, n_lakes), np.nan, dtype=np.float64)

    lid_to_col = {lid: j for j, lid in enumerate(all_lake_ids)}

    total = n_rows
    for i, (_, row) in enumerate(template_df.iterrows()):
        win_start = row["win_start"]
        win_end   = row["win_end"]

        if DEBUG and (i + 1) % 50 == 0:
            debug_print(f"  Window {i + 1}/{total}: {win_start} – {win_end}")

        for prefix, data in aoi_data.items():
            volumes, mean_depths, obs_pixels, obs_fractions = compute_lake_metrics_for_window(
                prefix     = prefix,
                win_start  = win_start,
                win_end    = win_end,
                raster_dir = data["raster_dir"],
                lake_ids   = data["lake_ids"],
                masks      = data["masks"],
                px_totals  = data["px_totals"],
            )
            for lid in data["lake_ids"]:
                j = lid_to_col[lid]
                if not np.isnan(volumes[lid]):
                    vol_matrix[i, j]   = volumes[lid]
                    depth_matrix[i, j] = mean_depths[lid]
                    px_matrix[i, j]    = obs_pixels[lid]
                    frac_matrix[i, j]  = obs_fractions[lid]

    meta = template_df[["win_start", "win_end"]]

    outputs = [
        (vol_matrix,   "per_lake_volume.csv",        "volume [m³]"),
        (depth_matrix, "per_lake_mean_depth.csv",    "mean depth [m]"),
        (px_matrix,    "per_lake_obs_pixels.csv",    "obs pixel count"),
        (frac_matrix,  "per_lake_obs_fraction.csv",  "obs area fraction"),
    ]

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for matrix, filename, label in outputs:
        lake_df = pd.DataFrame(matrix, columns=all_lake_ids, index=template_df.index)
        out_df  = pd.concat([meta, lake_df], axis=1)
        out_path = OUTPUT_DIR / filename
        out_df.to_csv(out_path, index=False)
        print(f"Written: {out_path}  ({label})")
        print(f"Shape:   {out_df.shape[0]} rows × {out_df.shape[1]} columns")

    # --------------------------------------------------------
    # QA summary
    # --------------------------------------------------------
    if DEBUG:
        vol_df = pd.DataFrame(vol_matrix, columns=all_lake_ids)
        print("\n── QA Summary ──────────────────────────────────────────")
        for prefix, ids in [("PTM", ptm_ids), ("OST", ost_ids)]:
            if not ids:
                continue
            sub = vol_df[ids]
            n_blank    = sub.isna().all(axis=1).sum()
            n_with_vol = (sub > 0).any(axis=1).sum()
            n_all_zero = (sub.notna().any(axis=1) & ~(sub > 0).any(axis=1)).sum()
            print(
                f"  {prefix} ({len(ids)} lake col(s)): "
                f"{n_with_vol} window(s) with ≥1 lake vol > 0  |  "
                f"{n_all_zero} window(s) all-zero  |  "
                f"{n_blank} window(s) blank (< 99 % coverage or no raster)"
            )
        print("\nDone.")


if __name__ == "__main__":
    main()
