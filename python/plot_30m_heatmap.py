#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_30m_heatmap.py

Plots glacier-wide persistency heatmaps from the 30 m persistency rasters
produced by create_30m_persistency_rasters.py.

For each AOI (PTM, OST), loads three persistency rasters:
    {prefix}_lake_persistency.tif
    {prefix}_slush_persistency.tif
    {prefix}_combined_persistency.tif

and produces:
  1. A three-panel per-AOI figure (a=slush, b=lake, c=combined) with AOI
     outline, scale bar, north arrow, and shared horizontal colorbar, saved
     as {prefix}_heatmap.png.
  2. Optionally (PLOT_SIDE_BY_SIDE=True), a combined figure comparing all
     AOIs side-by-side, saved as AOI_comparison_heatmap.png.

The AOI boundary shapefiles are reprojected to the raster CRS (EPSG:3995)
for the plot overlay.

Prerequisite: run create_30m_persistency_rasters.py first.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from rasterio.transform import Affine


# ============================================================
# USER SETTINGS
# ============================================================

PERSISTENCY_RASTER_DIR = Path(
    r"Q:\ThesisData\data\raster_data\30_persistency_rasters"
)

OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_30m_heatmap")

AOI_CONFIGS = [
    {
        "prefix":  "PTM",
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp"),
    },
    {
        "prefix":  "OST",
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp"),
    },
]

# When True – also produce a single side-by-side figure comparing all AOIs
# with a shared colorbar, saved as AOI_comparison_heatmap.png.
PLOT_SIDE_BY_SIDE = True

# Horizontal spacing between AOI columns in the side-by-side figure.
SIDE_BY_SIDE_WSPACE = 0.01

# Cap the colorbar upper limit. Set to None to use the observed maximum.
MAX_SCORE_CAP = 40

# Persistency scores at or below this threshold are rendered white.
MIN_DISPLAY_SCORE = 1

# Matplotlib colormap name for the persistency heatmap.
COLORMAP = "viridis_r"


# ============================================================
# DATA LOADING
# ============================================================

def load_persistency_arrays(prefix: str) -> tuple:
    """
    Load the three persistency GeoTIFFs for one AOI.

    Returns (slush_pers, lake_pers, combined_pers, transform, shape, crs).
    """
    layers = ['slush_persistency', 'lake_persistency', 'combined_persistency']
    arrays    = []
    transform = shape = crs = None

    for layer in layers:
        path = PERSISTENCY_RASTER_DIR / f"{prefix}_{layer}.tif"
        if not path.exists():
            raise FileNotFoundError(
                f"[{prefix}] Persistency raster not found: {path}\n"
                "Run create_30m_persistency_rasters.py first."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.int32)
            if transform is None:
                transform = src.transform
                shape     = (src.height, src.width)
                crs       = src.crs
        arrays.append(arr)
        print(f"  [{prefix}] Loaded {path.name}")

    return (*arrays, transform, shape, crs)


# ============================================================
# FIGURE HELPERS
# ============================================================

def _add_north_arrow(ax, x=0.96, y=0.84, arrow_len=0.08, fontsize=10):
    ax.annotate(
        "",
        xy=(x, y + arrow_len), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        annotation_clip=False,
    )
    ax.text(
        x, y + arrow_len + 0.025, "N",
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold",
        transform=ax.transAxes,
    )


def _add_scale_bar(ax, scale_km=10, n_segs=2, x_frac=0.05, y_frac=0.06):
    """Draw an alternating black/white scale bar in data coordinates (metres)."""
    xlim  = ax.get_xlim()
    ylim  = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    x0      = xlim[0] + x_frac * xspan
    y0      = ylim[0] + y_frac * yspan
    bar_len = scale_km * 1000
    seg_len = bar_len / n_segs
    bar_h   = yspan * 0.012

    for i in range(n_segs):
        rect = mpatches.FancyBboxPatch(
            (x0 + i * seg_len, y0), seg_len, bar_h,
            boxstyle="square,pad=0",
            facecolor="black" if i % 2 == 0 else "white",
            edgecolor="black", linewidth=0.8,
            transform=ax.transData, zorder=5, clip_on=False,
        )
        ax.add_patch(rect)

    km_per_seg = scale_km // n_segs
    for i in range(n_segs + 1):
        ax.text(
            x0 + i * seg_len, y0 + bar_h + yspan * 0.012,
            str(i * km_per_seg),
            ha="center", va="bottom",
            fontsize=7.5, transform=ax.transData, zorder=5,
        )
    ax.text(
        x0 + bar_len / 2, y0 + bar_h + yspan * 0.040,
        "Km",
        ha="center", va="bottom",
        fontsize=7.5, transform=ax.transData, zorder=5,
    )


def _raster_extent(transform: Affine, shape: tuple) -> list:
    """Return [xmin, xmax, ymin, ymax] in data coordinates."""
    height, width = shape
    res  = transform.a
    xmin = transform.c
    ymax = transform.f
    return [xmin, xmin + width * res, ymax - height * res, ymax]


def _build_colormap_norm(arrays: list) -> tuple:
    """Return (cmap, norm, vmax, capped) from a list of persistency arrays."""
    obs_max = max(int(a.max()) for a in arrays)
    vmax    = MAX_SCORE_CAP if MAX_SCORE_CAP is not None else obs_max
    vmax    = max(vmax, 1)
    capped  = obs_max > vmax

    cmap = plt.colormaps[COLORMAP].copy()
    cmap.set_under("white")
    cmap.set_bad("white")
    norm = Normalize(vmin=MIN_DISPLAY_SCORE - 0.5, vmax=vmax + 0.5)
    return cmap, norm, vmax, capped


def _colorbar_ticks(lo: int, vmax: int, capped: bool) -> tuple:
    """Return (tick_values, tick_labels) for a persistency colorbar."""
    if vmax - lo >= 10:
        mid         = round((lo + vmax) / 2)
        tick_vals   = [lo, mid, vmax]
        tick_labels = [str(lo), str(mid), f"≥ {vmax}" if capped else str(vmax)]
    elif vmax >= lo:
        tick_vals   = list(range(lo, vmax + 1))
        tick_labels = [f"≥ {v}" if (capped and v == vmax) else str(v)
                       for v in tick_vals]
    else:
        tick_vals   = [lo]
        tick_labels = [f"≥ {lo}" if capped else str(lo)]
    return tick_vals, tick_labels


# ============================================================
# PER-AOI FIGURE
# ============================================================

def plot_heatmap(prefix: str,
                 slush_pers: np.ndarray,
                 lake_pers: np.ndarray,
                 combined_pers: np.ndarray,
                 transform: Affine,
                 shape: tuple,
                 raster_crs,
                 aoi_shp: Path,
                 output_dir: Path) -> None:
    """
    Produce a three-panel persistency heatmap and save as {prefix}_heatmap.png.

    Panels: (a) Slush  (b) Lake  (c) Combined
    """
    aoi_gdf = gpd.read_file(aoi_shp)
    if aoi_gdf.crs is None:
        aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
    aoi_gdf = aoi_gdf.to_crs(raster_crs)

    extent = _raster_extent(transform, shape)
    xmin, xmax, ymin, ymax = extent

    cmap, norm, vmax, capped = _build_colormap_norm(
        [slush_pers, lake_pers, combined_pers]
    )

    panels = [
        (slush_pers,    "(a)"),
        (lake_pers,     "(b)"),
        (combined_pers, "(c)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 14))
    fig.subplots_adjust(hspace=0.06, left=0.04, right=0.96, top=0.97, bottom=0.08)

    for i, (ax, (arr, label)) in enumerate(zip(axes, panels)):
        masked = np.ma.masked_where(arr < MIN_DISPLAY_SCORE, arr.astype(float))

        ax.imshow(
            masked,
            cmap=cmap, norm=norm,
            extent=extent,
            origin="upper",
            interpolation="none",
            aspect="auto",
            zorder=2,
        )
        aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

        ax.text(
            0.01, 0.97, label,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left",
        )
        if i == 0:
            _add_north_arrow(ax)
        if i == len(panels) - 1:
            _add_scale_bar(ax, scale_km=10, n_segs=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.030, 0.70, 0.016])
    cbar    = fig.colorbar(
        sm, cax=cbar_ax, orientation="horizontal",
        extend="max" if capped else "neither",
    )
    cbar.set_label("Persistency Scores", fontsize=11, labelpad=4)
    cbar.ax.tick_params(labelsize=9)

    tick_vals, tick_labels = _colorbar_ticks(MIN_DISPLAY_SCORE, vmax, capped)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)

    output_path = output_dir / f"{prefix}_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    print(f"[{prefix}] Figure saved → {output_path}")


# ============================================================
# SIDE-BY-SIDE COMPARISON FIGURE
# ============================================================

def plot_side_by_side_heatmap(aoi_data_list: list, output_dir: Path) -> None:
    """
    Produce a 3-row × N-column figure comparing all AOIs with a shared colorbar.

    Rows:    slush / lake / combined persistency.
    Columns: one per AOI.

    Each dict in aoi_data_list must contain:
        prefix, slush_pers, lake_pers, combined_pers,
        transform, shape, raster_crs, aoi_shp
    """
    all_arrays = []
    for d in aoi_data_list:
        all_arrays += [d['slush_pers'], d['lake_pers'], d['combined_pers']]
    cmap, norm, vmax, capped = _build_colormap_norm(all_arrays)

    n_aois     = len(aoi_data_list)
    layer_keys = ['slush_pers', 'lake_pers', 'combined_pers']
    n_rows     = len(layer_keys)

    extent_list  = [_raster_extent(d['transform'], d['shape']) for d in aoi_data_list]
    width_ratios = [(x1 - x0) / (y1 - y0) for (x0, x1, y0, y1) in extent_list]

    row_h = 5
    fig_h = row_h * n_rows + 1.0
    fig_w = sum(width_ratios) * row_h * (0.87 / 0.94)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(
        n_rows, n_aois,
        width_ratios=width_ratios,
        hspace=0.04, wspace=SIDE_BY_SIDE_WSPACE,
        left=0.03, right=0.97, top=0.92, bottom=0.08,
    )
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(n_aois)]
                     for r in range(n_rows)])

    for col, (d, (xmin, xmax, ymin, ymax)) in enumerate(
            zip(aoi_data_list, extent_list)):
        prefix  = d['prefix']
        extent  = [xmin, xmax, ymin, ymax]

        aoi_gdf = gpd.read_file(d['aoi_shp'])
        if aoi_gdf.crs is None:
            aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
        aoi_gdf = aoi_gdf.to_crs(d['raster_crs'])

        for row, key in enumerate(layer_keys):
            ax     = axes[row, col]
            arr    = d[key]
            masked = np.ma.masked_where(arr < MIN_DISPLAY_SCORE, arr.astype(float))

            ax.imshow(
                masked,
                cmap=cmap, norm=norm,
                extent=extent,
                origin="upper",
                interpolation="none",
                aspect="equal",
                zorder=2,
            )
            aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.axis("off")

            label = f"({chr(ord('a') + col * n_rows + row)})"
            ax.text(
                0.01, 0.97, label,
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left",
            )
            if row == 0:
                ax.set_title(prefix, fontsize=13, fontweight="bold", pad=6)
                _add_north_arrow(ax)
            if row == n_rows - 1:
                _add_scale_bar(ax, scale_km=10, n_segs=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.030, 0.70, 0.014])
    cbar    = fig.colorbar(
        sm, cax=cbar_ax, orientation="horizontal",
        extend="max" if capped else "neither",
    )
    cbar.set_label("Persistency Scores", fontsize=11, labelpad=4)
    cbar.ax.tick_params(labelsize=9)

    tick_vals, tick_labels = _colorbar_ticks(MIN_DISPLAY_SCORE, vmax, capped)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)

    output_path = output_dir / "AOI_comparison_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    print(f"[side-by-side] Figure saved → {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aoi_data_list = []

    for cfg in AOI_CONFIGS:
        prefix  = cfg['prefix']
        aoi_shp = cfg['aoi_shp']

        print(f"\n[{prefix}] Loading persistency rasters...")

        slush_pers, lake_pers, combined_pers, transform, shape, raster_crs = (
            load_persistency_arrays(prefix)
        )

        plot_heatmap(
            prefix, slush_pers, lake_pers, combined_pers,
            transform, shape, raster_crs, aoi_shp, OUTPUT_DIR,
        )

        aoi_data_list.append({
            "prefix":        prefix,
            "slush_pers":    slush_pers,
            "lake_pers":     lake_pers,
            "combined_pers": combined_pers,
            "transform":     transform,
            "shape":         shape,
            "raster_crs":    raster_crs,
            "aoi_shp":       aoi_shp,
        })

    if PLOT_SIDE_BY_SIDE and len(aoi_data_list) >= 2:
        print("\nGenerating side-by-side comparison figure...")
        plot_side_by_side_heatmap(aoi_data_list, OUTPUT_DIR)

    print("\nAll AOIs complete.")


if __name__ == "__main__":
    main()
