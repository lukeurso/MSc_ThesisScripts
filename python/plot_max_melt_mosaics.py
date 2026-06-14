# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 2026

@author: luur4790

plot_max_melt_mosaics.py

For each active AOI (PTM, OST), finds the annual max-melt medoid GeoTIFF
exported by gee_js/export_max_melt_medoids.js and produces a single 12-panel
true-colour figure covering 2014–2025.

TIFs are matched by AOI prefix and year; the mosaic date window from the
filename is printed in the lower-left corner of each panel.

Output filenames (saved to OUTPUT_DIR):
  {AOI}_max_melt_mosaics.png

Prerequisites:
  GeoTIFFs exported from GEE must exist in MOSAIC_DIR with the naming
  convention:
      {AOI}_{YYYY-MM-DD}_{YYYY-MM-DD}_max_melt.tif
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio


# ============================================================
# 1) USER SETTINGS
# ============================================================

YEARS = list(range(2014, 2026))  # 2014–2025 inclusive

# Set to ["PTM"], ["OST"], or ["PTM", "OST"] to process both
ACTIVE_AOIS = ["PTM", "OST"]

AOI_CONFIGS = {
    "PTM": {
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp"),
    },
    "OST": {
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp"),
    },
}

MOSAIC_DIR = Path(r"Q:\ThesisData\data\raster_data\annual_max_melt_medoids")
OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_max_melt_mosaics")

DPI = 150

# Per-band percentile stretch applied to TOA reflectance values before display.
STRETCH_PLOW  = 2
STRETCH_PHIGH = 98


# ============================================================
# 2) STYLING
# ============================================================

AOI_EDGE_COLOR = "black"
AOI_LINEWIDTH  = 0.8

N_COLS = 4
N_ROWS = 3   # 3 × 4 = 12 panels for years 2014–2025

FIG_SIZE             = (10, 12)
PANEL_WSPACE         = 0.05   # horizontal gap between columns (fraction of avg column width)
PANEL_TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE    = 13
DATE_LABEL_FONTSIZE  = 7
NO_DATA_FONTSIZE     = 9
NO_DATA_COLOR        = "#888888"


# ============================================================
# 3) CONSTANTS
# ============================================================

_MOSAIC_RE = re.compile(
    r"^([A-Z]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_max_melt\.tif$",
    re.IGNORECASE,
)


# ============================================================
# 4) HELPERS
# ============================================================

def discover_mosaics(mosaic_dir: Path, prefix: str) -> dict:
    """
    Scan mosaic_dir for TIF files matching {prefix}_{YYYY-*}_max_melt.tif.

    Returns:
        {year: {"path": Path, "start_date": str, "end_date": str}}
    """
    mosaics: dict = {}
    for f in sorted(mosaic_dir.glob(f"{prefix}_*.tif")):
        m = _MOSAIC_RE.match(f.name)
        if m is None or m.group(1).upper() != prefix.upper():
            continue
        year = int(m.group(2)[:4])
        mosaics[year] = {
            "path":       f,
            "start_date": m.group(2),
            "end_date":   m.group(3),
        }
    return mosaics


def get_raster_crs(mosaics: dict):
    """Return the CRS from the first available raster, or None."""
    for yr in sorted(mosaics):
        tif = mosaics[yr]["path"]
        if tif.exists():
            with rasterio.open(tif) as src:
                return src.crs
    return None


def read_rgb(tif_path: Path) -> tuple:
    """
    Read a 3-band GeoTIFF (band order: B4 red, B3 green, B2 blue as exported
    by GEE) and return a display-ready RGB array with a percentile stretch.

    Uses rasterio's masked read so the file's nodata value is respected
    regardless of what it is.  Nodata pixels are filled with white (1.0) so
    they blend with the figure background.  Falls back to masking all-zero
    pixels when the file carries no nodata metadata.

    Returns:
        rgb    : (H, W, 3) float32 array, values in [0, 1]
        extent : (left, right, bottom, top) in the raster's native CRS
        crs    : rasterio CRS object
    """
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            warnings.warn(f"Expected 3 bands, got {src.count} in {tif_path.name}")
        data = src.read(masked=True, out_dtype="float32")  # MaskedArray (bands, H, W)
        t    = src.transform
        h, w = src.height, src.width
        crs  = src.crs

    left   = t.c
    top    = t.f
    right  = left + t.a * w
    bottom = top  + t.e * h   # t.e is negative
    extent = (left, right, bottom, top)

    # Build invalid mask: True where any band is masked (nodata).
    # If the file has no nodata metadata, fall back to all-zero detection.
    if np.ma.is_masked(data) and data.mask.any():
        invalid = np.any(data.mask, axis=0)          # (H, W)
    else:
        invalid = np.all(data.data == 0.0, axis=0)   # fallback

    # Per-band percentile stretch on valid pixels only → [0, 1]
    raw = data.data                                  # plain ndarray, same shape
    rgb = np.ones((3, h, w), dtype=np.float32)       # initialise to white (nodata fill)
    for i in range(min(3, raw.shape[0])):
        band  = raw[i].copy()
        valid = band[~invalid]
        if valid.size == 0:
            continue
        p_low, p_high = np.percentile(valid, [STRETCH_PLOW, STRETCH_PHIGH])
        if p_high > p_low:
            stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            stretched = np.clip(band, 0.0, 1.0)
        stretched[invalid] = 1.0                     # nodata → white
        rgb[i] = stretched

    return np.moveaxis(rgb, 0, -1), extent, crs      # (H, W, 3)


def load_aoi(aoi_shp: Path) -> gpd.GeoDataFrame:
    """Load the AOI boundary shapefile."""
    return gpd.read_file(aoi_shp)


# ============================================================
# 5) PANEL HELPERS
# ============================================================

def _setup_panel(ax, year: int) -> None:
    """Apply common panel formatting: turn off axes, add year title."""
    ax.axis("off")
    ax.set_title(str(year), fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=3)


def _no_data_label(ax) -> None:
    ax.text(
        0.5, 0.5, "No data",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=NO_DATA_FONTSIZE, color=NO_DATA_COLOR, style="italic",
    )


def _date_label(ax, start_date: str, end_date: str) -> None:
    ax.text(
        0.02, 0.03, f"{start_date} – {end_date}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=DATE_LABEL_FONTSIZE, color="#333333",
    )


# ============================================================
# 6) FIGURE FUNCTION
# ============================================================

def plot_mosaic_figure(
    title: str,
    mosaics: dict,
    aoi_gdf: gpd.GeoDataFrame,
    raster_crs,
    out_path: Path,
) -> None:
    """
    12-panel true-colour figure; one panel per year (2014–2025).

    Parameters
    ----------
    title      : figure super-title
    mosaics    : {year: {"path": Path, "start_date": str, "end_date": str}}
    aoi_gdf    : AOI boundary GeoDataFrame (any CRS — reprojected internally)
    raster_crs : CRS of the rasters, used to reproject the AOI for overlay
    out_path   : output PNG path
    """
    aoi_proj = aoi_gdf.to_crs(raster_crs)
    xmin, ymin, xmax, ymax = aoi_proj.total_bounds

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE, squeeze=False)
    flat = axes.flatten()

    for idx, yr in enumerate(YEARS):
        ax = flat[idx]
        _setup_panel(ax, yr)

        entry = mosaics.get(yr)
        if entry is None or not entry["path"].exists():
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            aoi_proj.boundary.plot(ax=ax, color=AOI_EDGE_COLOR, linewidth=AOI_LINEWIDTH, zorder=2)
            _no_data_label(ax)
            continue

        rgb, extent, _ = read_rgb(entry["path"])
        ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear", zorder=1)
        # Set limits and aspect after imshow so matplotlib doesn't override them.
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        aoi_proj.boundary.plot(ax=ax, color=AOI_EDGE_COLOR, linewidth=AOI_LINEWIDTH, zorder=2)
        _date_label(ax, entry["start_date"], entry["end_date"])

    for idx in range(len(YEARS), N_ROWS * N_COLS):
        flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(wspace=PANEL_WSPACE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


# ============================================================
# 7) MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for prefix in ACTIVE_AOIS:
        if prefix not in AOI_CONFIGS:
            warnings.warn(f"Unknown AOI '{prefix}' in ACTIVE_AOIS — skipping.")
            continue

        cfg = AOI_CONFIGS[prefix]

        print(f"\n{'=' * 60}")
        print(f"[{prefix}] Discovering mosaics in {MOSAIC_DIR} …")
        mosaics = discover_mosaics(MOSAIC_DIR, prefix)
        print(f"  Found {len(mosaics)} mosaic(s): {sorted(mosaics)}")

        raster_crs = get_raster_crs(mosaics)
        if raster_crs is None:
            warnings.warn(f"[{prefix}] No readable rasters found — skipping.")
            continue
        print(f"  Raster CRS: {raster_crs}")

        print(f"[{prefix}] Loading AOI boundary …")
        aoi_gdf = load_aoi(cfg["aoi_shp"])

        print(f"[{prefix}] Plotting 12-panel mosaic figure …")
        plot_mosaic_figure(
            title      = f"{prefix}  –  Annual Max-Melt Medoid Mosaic  (2014–2025)",
            mosaics    = mosaics,
            aoi_gdf    = aoi_gdf,
            raster_crs = raster_crs,
            out_path   = OUTPUT_DIR / f"{prefix}_max_melt_mosaics.png",
        )

    print("\nAll AOIs complete.")


if __name__ == "__main__":
    main()
