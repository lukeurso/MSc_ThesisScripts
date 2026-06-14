# -*- coding: utf-8 -*-
"""
plot_lakes_by_drainage_count.py

For each AOI, produces two spatial maps of the persistent lake polygons:

  1. Full drainages only  (drain_type = "full", > 80 % volume loss)
  2. Full + partial combined  (produced when PLOT_COMBINED = True)

In both maps lakes are filled with graduated colours representing the
total JJA (June–August) drainage count across all study years combined.
Lakes with no qualifying drainages are rendered in NO_DRAINAGE_COLOR.

Outputs (saved to OUTPUT_DIR):
  {prefix}_lakes_by_drainage_count_full.png      – full drainages only
  {prefix}_lakes_by_drainage_count_combined.png  – full + partial (if PLOT_COMBINED)
"""

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


# ============================================================
# PARAMETERS
# ============================================================

# -- Input paths ----------------------------------------------------------
DRAINAGE_EVENTS_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\drainage_events.csv"
)

PERSISTENT_LAKES_DIR = Path(
    r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles"
)

# -- Output ---------------------------------------------------------------
OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_lakes_drainage_count")

# -- AOI configurations ---------------------------------------------------
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

# -- Lake IDs to exclude from all plots (e.g. unreliable or edge-case lakes) --
EXCLUDED_LAKE_IDS = [
    "P016",
    "P019",
]

# -- Column in the polygon shapefile that matches lake_id in the CSV ------
LAKE_ID_FIELD = "lake_id"

# -- First character of lake_id → AOI prefix mapping ---------------------
LAKE_PREFIX_AOI = {"P": "PTM", "C": "OST"}

# -- Season filter: JJA = June–August -------------------------------------
JJA_DOY_START = 152   # 1 Jun
JJA_DOY_END   = 243   # 31 Aug

# -- Set True to also produce a full + partial combined plot per AOI ------
PLOT_COMBINED = True

# -- Coordinate reference system ------------------------------------------
TARGET_CRS = "EPSG:3413"   # WGS 84 / NSIDC Sea Ice Polar Stereographic North

# -- Colormap for graduated polygon fill (lakes with ≥ 1 drainage) --------
COLORMAP = "YlOrRd"

# Cap the colorbar upper limit; set to None to use the observed maximum.
MAX_COUNT_CAP = None

# Colour for lakes with zero qualifying drainages during JJA.
NO_DRAINAGE_COLOR = "#d9d9d9"   # light grey

# AOI background fill and outline style
AOI_FACE_COLOR = "#f0f0f0"   # very light grey fill for the AOI extent
AOI_EDGE_COLOR = "black"
AOI_EDGE_WIDTH = 0.8

# Lake polygon outline style
LAKE_EDGE_COLOR = "black"
LAKE_EDGE_WIDTH = 0.4

# Scale bar length (km)
SCALE_KM = 10

# Figure dimensions and resolution
FIG_SIZE   = (7, 8)
DPI_SCREEN = 150
DPI_SAVE   = 300

# ============================================================


# ============================================================
# DATA LOADING
# ============================================================

def load_drainage_counts(csv_path: Path, prefix: str,
                         drain_types: list[str]) -> pd.Series:
    """
    Return a Series of JJA drainage counts indexed by lake_id for *prefix*.

    Only events whose drain_type is in *drain_types* are counted.
    Counts sum all qualifying events across all years in the dataset.
    """
    df = pd.read_csv(csv_path)
    df["post_win_dt"] = pd.to_datetime(df["post_win_start"], errors="coerce")
    df["doy"]         = df["post_win_dt"].dt.dayofyear
    df["aoi"]         = df["lake_id"].str[0].map(LAKE_PREFIX_AOI)

    mask = (
        df["doy"].between(JJA_DOY_START, JJA_DOY_END)
        & (df["drain_type"].isin(drain_types))
        & (df["aoi"] == prefix)
        & (~df["lake_id"].isin(EXCLUDED_LAKE_IDS))
    )
    return df[mask].groupby("lake_id").size()


def load_lake_polygons(prefix: str) -> gpd.GeoDataFrame:
    """Load and reproject the persistent lake polygon shapefile for *prefix*."""
    poly_path = PERSISTENT_LAKES_DIR / f"{prefix}_persistent_lake_polygons.shp"
    if not poly_path.exists():
        warnings.warn(f"[{prefix}] Polygon shapefile not found: {poly_path}")
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(poly_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(TARGET_CRS)
    elif str(gdf.crs).upper() != TARGET_CRS.upper():
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def load_aoi(aoi_shp: Path) -> gpd.GeoDataFrame:
    """Load and reproject the AOI boundary shapefile."""
    gdf = gpd.read_file(aoi_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() != TARGET_CRS.upper():
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


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
    xlim    = ax.get_xlim()
    ylim    = ax.get_ylim()
    xspan   = xlim[1] - xlim[0]
    yspan   = ylim[1] - ylim[0]
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
            transform=ax.transData, zorder=6, clip_on=False,
        )
        ax.add_patch(rect)

    km_per_seg = scale_km // n_segs
    for i in range(n_segs + 1):
        ax.text(
            x0 + i * seg_len, y0 + bar_h + yspan * 0.012,
            str(i * km_per_seg),
            ha="center", va="bottom",
            fontsize=7.5, transform=ax.transData, zorder=6,
        )
    ax.text(
        x0 + bar_len / 2, y0 + bar_h + yspan * 0.040,
        "Km",
        ha="center", va="bottom",
        fontsize=7.5, transform=ax.transData, zorder=6,
    )


def _build_cmap_norm(count_max: int):
    vmax   = MAX_COUNT_CAP if MAX_COUNT_CAP is not None else count_max
    vmax   = max(vmax, 1)
    capped = count_max > vmax
    cmap   = plt.colormaps[COLORMAP].copy()
    # vmin=0.5 so count=1 maps near the bottom of the colormap
    norm   = Normalize(vmin=0.5, vmax=vmax + 0.5)
    return cmap, norm, vmax, capped


def _add_colorbar(fig, cmap, norm, vmax, capped, rect, drain_label: str):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes(rect)
    cbar    = fig.colorbar(
        sm, cax=cbar_ax, orientation="horizontal",
        extend="max" if capped else "neither",
    )
    cbar.set_label(
        f"JJA {drain_label} rapid drainage count (all years)",
        fontsize=10, labelpad=4,
    )
    cbar.ax.tick_params(labelsize=9)

    if vmax <= 10:
        tick_vals   = list(range(1, vmax + 1))
        tick_labels = [
            f"≥ {v}" if (capped and v == vmax) else str(v)
            for v in tick_vals
        ]
    else:
        mid         = round((1 + vmax) / 2)
        tick_vals   = [1, mid, vmax]
        tick_labels = ["1", str(mid), f"≥ {vmax}" if capped else str(vmax)]

    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)


# ============================================================
# PER-AOI PLOT
# ============================================================

def plot_aoi(prefix: str, aoi_shp: Path,
             polys: gpd.GeoDataFrame,
             counts: pd.Series,
             output_dir: Path,
             drain_label: str,
             filename_suffix: str) -> None:
    """Produce and save a drainage-count map for one AOI."""

    polys = polys.copy()
    if LAKE_ID_FIELD in polys.columns and EXCLUDED_LAKE_IDS:
        n_before = len(polys)
        polys = polys[~polys[LAKE_ID_FIELD].isin(EXCLUDED_LAKE_IDS)].copy()
        n_excluded = n_before - len(polys)
        if n_excluded:
            print(f"[{prefix}] Excluded {n_excluded} lake(s): {EXCLUDED_LAKE_IDS}")

    if LAKE_ID_FIELD in polys.columns:
        polys["drain_count"] = (
            polys[LAKE_ID_FIELD].map(counts).fillna(0).astype(int)
        )
    else:
        warnings.warn(
            f"[{prefix}] Column '{LAKE_ID_FIELD}' not found in polygon shapefile; "
            "all lakes shown with zero count."
        )
        polys["drain_count"] = 0

    count_max = int(polys["drain_count"].max())
    n_zero    = int((polys["drain_count"] == 0).sum())
    n_nonzero = int((polys["drain_count"] > 0).sum())
    print(
        f"[{prefix}] [{drain_label}] Lakes total: {len(polys)}  |  "
        f"with JJA drainages: {n_nonzero}  |  none: {n_zero}  |  max count: {count_max}"
    )

    aoi_gdf  = load_aoi(aoi_shp)
    cmap, norm, vmax, capped = _build_cmap_norm(max(count_max, 1))

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.11)

    # Layer 1: AOI fill (spatial context)
    aoi_gdf.plot(ax=ax, facecolor=AOI_FACE_COLOR, edgecolor="none", zorder=1)

    # Layer 2: Lakes with zero qualifying JJA drainages
    zero_polys = polys[polys["drain_count"] == 0]
    if not zero_polys.empty:
        zero_polys.plot(
            ax=ax,
            facecolor=NO_DRAINAGE_COLOR,
            edgecolor=LAKE_EDGE_COLOR,
            linewidth=LAKE_EDGE_WIDTH,
            zorder=3,
        )

    # Layer 3: Lakes with ≥ 1 qualifying JJA drainage (graduated colour)
    nonzero_polys = polys[polys["drain_count"] > 0]
    if not nonzero_polys.empty:
        nonzero_polys.plot(
            ax=ax,
            column="drain_count",
            cmap=cmap,
            norm=norm,
            edgecolor=LAKE_EDGE_COLOR,
            linewidth=LAKE_EDGE_WIDTH,
            zorder=4,
        )

    # Layer 4: AOI outline
    aoi_gdf.boundary.plot(
        ax=ax, color=AOI_EDGE_COLOR, linewidth=AOI_EDGE_WIDTH, zorder=5,
    )

    # Fit axes to AOI extent
    xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    ax.set_title(
        f"{prefix} – Persistent lakes by JJA {drain_label} drainage count",
        fontsize=11, fontweight="bold", pad=6,
    )

    _add_north_arrow(ax)
    _add_scale_bar(ax, scale_km=SCALE_KM, n_segs=2)

    legend_handles = [
        mpatches.Patch(
            facecolor=NO_DRAINAGE_COLOR, edgecolor=LAKE_EDGE_COLOR,
            linewidth=LAKE_EDGE_WIDTH,
            label=f"No JJA {drain_label} drainages  (n = {n_zero})",
        ),
        mpatches.Patch(
            facecolor=plt.colormaps[COLORMAP](0.55),
            edgecolor=LAKE_EDGE_COLOR,
            linewidth=LAKE_EDGE_WIDTH,
            label=f"≥ 1 JJA {drain_label} drainage  (n = {n_nonzero})",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right",
        fontsize=8, framealpha=0.85, edgecolor="grey",
    )

    _add_colorbar(fig, cmap, norm, vmax, capped,
                  rect=[0.15, 0.038, 0.70, 0.018],
                  drain_label=drain_label)

    out_path = output_dir / f"{prefix}_lakes_by_drainage_count{filename_suffix}.png"
    fig.savefig(out_path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    print(f"[{prefix}] Figure saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DRAINAGE_EVENTS_CSV.exists():
        raise FileNotFoundError(
            f"Drainage events CSV not found:\n  {DRAINAGE_EVENTS_CSV}"
        )

    # Define the plots to produce: (drain_types, display label, filename suffix)
    plot_configs = [
        (["full"],             "full",             "_full"),
        (["full", "partial"],  "full + partial",   "_combined"),
    ]
    if not PLOT_COMBINED:
        plot_configs = plot_configs[:1]

    print("Loading drainage events …")

    for cfg in AOI_CONFIGS:
        prefix  = cfg["prefix"]
        aoi_shp = cfg["aoi_shp"]

        polys = load_lake_polygons(prefix)
        if polys.empty:
            warnings.warn(f"[{prefix}] No polygon data; skipping.")
            continue

        for drain_types, drain_label, suffix in plot_configs:
            counts = load_drainage_counts(DRAINAGE_EVENTS_CSV, prefix, drain_types)
            print(
                f"[{prefix}] JJA {drain_label} drainages: "
                f"{int(counts.sum())} events across {len(counts)} lakes"
            )
            plot_aoi(prefix, aoi_shp, polys, counts, OUTPUT_DIR,
                     drain_label=drain_label, filename_suffix=suffix)

    print("\nAll AOIs complete.")


if __name__ == "__main__":
    main()
