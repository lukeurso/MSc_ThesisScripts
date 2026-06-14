# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 2026

@author: luur4790

plot_max_melt_blocks.py

For each AOI (PTM, OST), identifies the 5-window temporal block per year
(2014–2025) with the maximum melt area for each layer type, then produces
three 12-panel figures per AOI:

  (a) max lake area     – block with highest total lake feature area per year
  (b) max slush area    – block with highest total slush feature area per year
  (c) max combined melt – block with highest total lake + slush area per year

Slush features are rendered in green; lake features in blue.
On the combined melt figure, lake features render above slush (higher zorder).
Each panel's sub-title is the year; the selected block start date is printed
in the lower-left corner of each panel.

Output filenames (saved to OUTPUT_DIR):
  {AOI}_max_lake_blocks.png
  {AOI}_max_slush_blocks.png
  {AOI}_max_melt_blocks.png

Prerequisites:
  Block shapefiles produced by create_n_day_blocks.py must exist in
  BLOCK_DIR_PTM / BLOCK_DIR_OST with the naming convention:
      {YYYY-MM-DD}_{YYYY-MM-DD}_{layer_type}.shp
"""

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1) USER SETTINGS
# ============================================================

YEARS = list(range(2014, 2026))  # 2014–2025 inclusive

AOI_CONFIGS = [
    {
        "prefix":    "PTM",
        "block_dir": Path(r"Q:\ThesisData\data\blocks\5_window\all_PTM_files_anchored"),
        "aoi_shp":   Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp"),
    },
    {
        "prefix":    "OST",
        "block_dir": Path(r"Q:\ThesisData\data\blocks\5_window\all_OST_files_anchored"),
        "aoi_shp":   Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp"),
    },
]

OUTPUT_DIR = Path(r"Q:\ThesisData\data\figures\plot_max_melt_blocks")

# Pre-computed block-level area totals (same CSV used by plot_block_melt_and_aoi_coverage.py).
# Used to rank blocks by area; shapefiles are only loaded for the 12 selected blocks at render time.
TOTAL_SUMMARY_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\5_window_block"
    r"\5_win_block_total_summary_raw.csv"
)

# CRS used for all area calculations and plotting (must match block shapefiles).
PROJECTED_CRS = "EPSG:3413"

DPI = 150


# ============================================================
# 2) STYLING
# ============================================================

SLUSH_COLOR      = "#4CAF50"   # green
LAKE_COLOR       = "#2196F3"   # blue
FEATURE_ALPHA    = 0.85
FEATURE_LINEWIDTH = 0.2        # edge width for slush and lake features (all figures)
AOI_EDGE_COLOR   = "black"
AOI_LINEWIDTH    = 0.8

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

_BLOCK_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(footprint|slush|lake)\.shp$",
    re.IGNORECASE,
)

JJA_MONTHS = {6, 7, 8}  # June, July, August


# ============================================================
# 4) HELPERS
# ============================================================

def discover_blocks(block_dir: Path) -> dict:
    """
    Scan block_dir for shapefiles matching the block naming convention and
    return a nested dict:
        {block_start: {"slush": Path | None, "lake": Path | None}}

    Only blocks whose start month falls in JJA (June, July, August) are
    included.  Footprint shapefiles are ignored; only slush and lake are
    indexed.
    """
    blocks: dict = {}
    for f in sorted(block_dir.glob("*.shp")):
        m = _BLOCK_RE.match(f.name)
        if m is None:
            continue
        blk_start = m.group(1)
        start_month = int(blk_start[5:7])
        if start_month not in JJA_MONTHS:
            continue
        layer = m.group(3).lower()
        if layer not in ("slush", "lake"):
            continue
        if blk_start not in blocks:
            blocks[blk_start] = {"slush": None, "lake": None}
        blocks[blk_start][layer] = f
    return blocks


def load_summary_csv(path: Path) -> pd.DataFrame:
    """Load the block total summary CSV and parse block_start as datetime."""
    df = pd.read_csv(path)
    df["block_start"] = pd.to_datetime(df["block_start"], errors="coerce")
    return df


def select_max_blocks_from_csv(summary_df: pd.DataFrame, aoi: str, years: list) -> dict:
    """
    Use pre-computed areas from the summary CSV to identify the max block per year.

    Metrics:
      "lake"     – block_start with highest {aoi}_lake_area_m2
      "slush"    – block_start with highest {aoi}_slush_area_m2
      "combined" – block_start with highest lake + slush area

    Returns:
        {year: {"lake": str | None, "slush": str | None, "combined": str | None}}
    """
    slush_col = f"{aoi}_slush_area_m2"
    lake_col  = f"{aoi}_lake_area_m2"

    jja = summary_df[summary_df["block_start"].dt.month.isin(JJA_MONTHS)].copy()

    def _best(yr_df: pd.DataFrame, col: str):
        if col not in yr_df.columns:
            return None
        candidates = yr_df[yr_df[col].gt(0)][["block_start", col]]
        if candidates.empty:
            return None
        return candidates.loc[candidates[col].idxmax(), "block_start"].strftime("%Y-%m-%d")

    result: dict = {}
    for yr in years:
        yr_df = jja[jja["block_start"].dt.year == yr]

        lake_best  = _best(yr_df, lake_col)
        slush_best = _best(yr_df, slush_col)

        if slush_col in yr_df.columns and lake_col in yr_df.columns:
            combined = yr_df.copy()
            combined["_combined"] = (
                combined[slush_col].fillna(0.0) + combined[lake_col].fillna(0.0)
            )
            combined = combined[combined["_combined"].gt(0)]
            combined_best = (
                combined.loc[combined["_combined"].idxmax(), "block_start"].strftime("%Y-%m-%d")
                if not combined.empty else None
            )
        else:
            combined_best = None

        result[yr] = {"lake": lake_best, "slush": slush_best, "combined": combined_best}

    return result


def load_gdf(shp_path) -> gpd.GeoDataFrame:
    """Load a shapefile and reproject to PROJECTED_CRS."""
    if shp_path is None or not Path(shp_path).exists():
        return gpd.GeoDataFrame(geometry=[], crs=PROJECTED_CRS)
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() != PROJECTED_CRS.upper():
        gdf = gdf.to_crs(PROJECTED_CRS)
    return gdf


def load_aoi(aoi_shp: Path) -> gpd.GeoDataFrame:
    """Load the AOI boundary shapefile and reproject to PROJECTED_CRS."""
    gdf = gpd.read_file(aoi_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs).upper() != PROJECTED_CRS.upper():
        gdf = gdf.to_crs(PROJECTED_CRS)
    return gdf


# ============================================================
# 5) PANEL SETUP HELPERS
# ============================================================

def _setup_panel(ax, aoi_gdf: gpd.GeoDataFrame, bounds: tuple, year: int) -> None:
    """Apply common panel formatting: limits, aspect, AOI boundary, title."""
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(str(year), fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold", pad=3)
    aoi_gdf.boundary.plot(ax=ax, color=AOI_EDGE_COLOR, linewidth=AOI_LINEWIDTH, zorder=4)


def _no_data_label(ax) -> None:
    ax.text(
        0.5, 0.5, "No data",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=NO_DATA_FONTSIZE, color=NO_DATA_COLOR, style="italic",
    )


def _date_label(ax, blk_start: str) -> None:
    ax.text(
        0.02, 0.03, blk_start,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=DATE_LABEL_FONTSIZE, color="#333333",
    )


# ============================================================
# 6) FIGURE FUNCTIONS
# ============================================================

def plot_single_layer_figure(
    title: str,
    layer: str,
    selected: dict,
    blocks: dict,
    aoi_gdf: gpd.GeoDataFrame,
    bounds: tuple,
    color: str,
    out_path: Path,
) -> None:
    """
    12-panel figure showing the max-area block for each year for a single
    layer type (either "lake" or "slush").

    Parameters
    ----------
    title    : figure super-title
    layer    : "lake" or "slush"
    selected : {year: block_start | None}
    blocks   : {block_start: {"slush": Path|None, "lake": Path|None}}
    aoi_gdf  : AOI boundary GeoDataFrame (already projected)
    bounds   : (xmin, ymin, xmax, ymax) in PROJECTED_CRS
    color    : fill colour for the features
    out_path : output PNG path
    """
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE, squeeze=False)
    flat = axes.flatten()

    for idx, yr in enumerate(YEARS):
        ax = flat[idx]
        _setup_panel(ax, aoi_gdf, bounds, yr)

        blk_start = selected.get(yr)
        if blk_start is None:
            _no_data_label(ax)
            continue

        shp_path = blocks[blk_start].get(layer)
        gdf = load_gdf(shp_path)
        if gdf.empty:
            _no_data_label(ax)
            continue

        gdf.plot(ax=ax, color=color, edgecolor=color, linewidth=FEATURE_LINEWIDTH, alpha=FEATURE_ALPHA, zorder=2)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        _date_label(ax, blk_start)

    for idx in range(len(YEARS), N_ROWS * N_COLS):
        flat[idx].set_visible(False)

    legend_label = "Lake" if layer == "lake" else "Slush"
    patch = mpatches.Patch(facecolor=color, edgecolor="none", label=legend_label)
    fig.legend(
        handles=[patch],
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#aaaaaa",
    )

    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(wspace=PANEL_WSPACE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


def plot_combined_figure(
    title: str,
    selected: dict,
    blocks: dict,
    aoi_gdf: gpd.GeoDataFrame,
    bounds: tuple,
    out_path: Path,
) -> None:
    """
    12-panel combined melt figure.  For each year, shows both slush (green)
    and lake (blue) features from the block with the highest combined
    lake + slush area.  Lake features render above slush (zorder=3 > 2).

    Parameters
    ----------
    title    : figure super-title
    selected : {year: block_start | None}  (combined-metric selection)
    blocks   : {block_start: {"slush": Path|None, "lake": Path|None}}
    aoi_gdf  : AOI boundary GeoDataFrame (already projected)
    bounds   : (xmin, ymin, xmax, ymax) in PROJECTED_CRS
    out_path : output PNG path
    """
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE, squeeze=False)
    flat = axes.flatten()

    for idx, yr in enumerate(YEARS):
        ax = flat[idx]
        _setup_panel(ax, aoi_gdf, bounds, yr)

        blk_start = selected.get(yr)
        if blk_start is None:
            _no_data_label(ax)
            continue

        slush_path = blocks[blk_start].get("slush")
        lake_path  = blocks[blk_start].get("lake")
        slush_gdf  = load_gdf(slush_path)
        lake_gdf   = load_gdf(lake_path)

        if slush_gdf.empty and lake_gdf.empty:
            _no_data_label(ax)
            continue

        # Slush rendered first (lower zorder); lake rendered on top.
        if not slush_gdf.empty:
            slush_gdf.plot(
                ax=ax, color=SLUSH_COLOR, edgecolor=SLUSH_COLOR,
                linewidth=FEATURE_LINEWIDTH, alpha=FEATURE_ALPHA, zorder=2,
            )
        if not lake_gdf.empty:
            lake_gdf.plot(
                ax=ax, color=LAKE_COLOR, edgecolor=LAKE_COLOR,
                linewidth=FEATURE_LINEWIDTH, alpha=FEATURE_ALPHA, zorder=3,
            )

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        _date_label(ax, blk_start)

    for idx in range(len(YEARS), N_ROWS * N_COLS):
        flat[idx].set_visible(False)

    slush_patch = mpatches.Patch(facecolor=SLUSH_COLOR, edgecolor="none", label="Slush")
    lake_patch  = mpatches.Patch(facecolor=LAKE_COLOR,  edgecolor="none", label="Lake")
    fig.legend(
        handles=[slush_patch, lake_patch],
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#aaaaaa",
    )

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

    print("Loading block area summary CSV …")
    summary_df = load_summary_csv(TOTAL_SUMMARY_CSV)

    for cfg in AOI_CONFIGS:
        prefix   = cfg["prefix"]
        blk_dir  = cfg["block_dir"]
        aoi_shp  = cfg["aoi_shp"]

        if not blk_dir.exists():
            warnings.warn(f"[{prefix}] Block directory not found: {blk_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{prefix}] Discovering block shapefiles …")
        blocks = discover_blocks(blk_dir)
        print(f"  Found {len(blocks)} unique block start dates")

        print(f"[{prefix}] Selecting max blocks from CSV …")
        max_blocks = select_max_blocks_from_csv(summary_df, prefix, YEARS)

        print(f"\n  {'Year':<6}  {'Max lake block':<14}  {'Max slush block':<14}  {'Max combined block'}")
        for yr in YEARS:
            mb = max_blocks[yr]
            print(
                f"  {yr:<6}  "
                f"{str(mb['lake']):<14}  "
                f"{str(mb['slush']):<14}  "
                f"{str(mb['combined'])}"
            )

        print(f"\n[{prefix}] Loading AOI boundary …")
        aoi_gdf = load_aoi(aoi_shp)
        xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
        bounds = (xmin, ymin, xmax, ymax)

        # ── (a) Max lake area ───────────────────────────────────────
        print(f"\n[{prefix}] Plotting (a) max lake area figure …")
        plot_single_layer_figure(
            title=f"{prefix}  –  Max Lake Area Block per Year  (2014–2025)",
            layer="lake",
            selected={yr: max_blocks[yr]["lake"] for yr in YEARS},
            blocks=blocks,
            aoi_gdf=aoi_gdf,
            bounds=bounds,
            color=LAKE_COLOR,
            out_path=OUTPUT_DIR / f"{prefix}_max_lake_blocks.png",
        )

        # ── (b) Max slush area ──────────────────────────────────────
        print(f"[{prefix}] Plotting (b) max slush area figure …")
        plot_single_layer_figure(
            title=f"{prefix}  –  Max Slush Area Block per Year  (2014–2025)",
            layer="slush",
            selected={yr: max_blocks[yr]["slush"] for yr in YEARS},
            blocks=blocks,
            aoi_gdf=aoi_gdf,
            bounds=bounds,
            color=SLUSH_COLOR,
            out_path=OUTPUT_DIR / f"{prefix}_max_slush_blocks.png",
        )

        # ── (c) Max combined melt area ──────────────────────────────
        print(f"[{prefix}] Plotting (c) max combined melt figure …")
        plot_combined_figure(
            title=f"{prefix}  –  Max Combined Melt Block per Year  (2014–2025)",
            selected={yr: max_blocks[yr]["combined"] for yr in YEARS},
            blocks=blocks,
            aoi_gdf=aoi_gdf,
            bounds=bounds,
            out_path=OUTPUT_DIR / f"{prefix}_max_melt_blocks.png",
        )

    print("\nAll AOIs complete.")


if __name__ == "__main__":
    main()
