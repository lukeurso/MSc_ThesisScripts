# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:48:55 2026

@author: Lukeu

create_persistent_lake_vectors.py

Extracts persistent lake pixels (score > SCORE_THRESHOLD) from the lake
persistency rasters produced by plot_heatmap.py, groups connected pixels
into polygon features, and writes them to a single shapefile with two ID fields:

  lake_id   - alphanumeric prefix + zero-padded number
                PTM lakes → P001, P002, …
                OST lakes → C001, C002, …
  name - name read from LAKE_NAMES_CSV (per-AOI column),
                assigned in descending area order

Prerequisite: run plot_heatmap.py with SAVE_PERSISTENCY_RASTERS = True so that
  {prefix}_lake_persistency_converted.tif files exist in PERSISTENCY_RASTER_DIR.

Output: PERSISTENT_LAKES_SHP
"""

import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import scipy.ndimage as ndimage
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


# ============================================================
# 1) USER SETTINGS
# ============================================================

# Pixels with lake persistency score strictly above this value are extracted.
SCORE_THRESHOLD = 25

PERSISTENCY_RASTER_DIR = Path(  #input path
    r"Q:\ThesisData\data\raster_data\converted_30m_persistency_rasters"
)

PERSISTENT_LAKES_DIR = Path(   #output path
    r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles" #updated 04-21 1806 for buffed
)

TARGET_CRS = "EPSG:3413"

# Connected components smaller than this pixel count are discarded as artefacts.
MIN_PIXEL_COUNT = 100

# CSV with one column per AOI listing Song of Ice and Fire lake names.
# Expected columns: ptm_sgl_names, ost_sgl_names  (50 rows each)
LAKE_NAMES_CSV = Path(r"Q:\ThesisData\data\drainage\persistent_lake_names_200.csv")

# Buffer distance in metres applied to every lake polygon after vectorisation.
# Set to 0 to disable buffering entirely.
BUFF_DIST = 300  # metres

# Minimum original lake area (m²) below which two overlapping buffered polygons
# are allowed to merge into a single feature.  A pair of lakes both strictly
# below this threshold will be dissolved together when their buffered extents
# intersect.  Lakes at or above this threshold are kept separate; any residual
# overlap is resolved by clipping the smaller polygon (larger lake takes
# priority).  Only active when BUFF_DIST > 0.
MERGE_THRESHOLDS = {
    "PTM": 100_800,  # m²
    "OST": 72_500,  # m²
}

# Fraction (0–1) of a polygon's perimeter that must be shared with a single
# larger adjacent polygon for the smaller to be absorbed into it.
# Evaluated after buffering and overlap resolution on the final clean geometries.
# Set to 0 to disable.  Example: 0.15 → merge when ≥15 % of perimeter is shared.
PERIMETER_SHARE_THRESHOLD = 0.15

AOI_CONFIGS = [
    {
        "prefix": "PTM",
        "id_letter": "P",
        "name_col": "ptm_sgl_names",
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\PTM\PTM_AOI_1500m.shp"),
        # Any PTM lake polygon that intersects this shapefile is deleted entirely.
        "intersector_shp": Path(
            r"Q:\ThesisData\data\data_correction_steps\1_visual"
            r"\PTM_intersect_polygons\PTM_intersector.shp"
        ),
    },
    {
        "prefix": "OST",
        "id_letter": "C",
        "name_col": "ost_sgl_names",
        "aoi_shp": Path(r"Q:\ThesisData\data\study_areas\OST\OST_AOI_1500m.shp"),
        "intersector_shp": None,
    },
]


# ============================================================
# 2) GEOMETRY HELPERS
# ============================================================

def fill_holes(geom):
    """Remove interior rings (holes) from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return geom
        return fill_holes(unary_union(polys))
    return geom


def _merge_by_perimeter_share(gdf, threshold):
    """
    Absorb a polygon into its largest touching neighbour when the length of
    their shared boundary is >= threshold fraction of the smaller polygon's
    total perimeter.  Union-find captures transitive chains (A→B, B→C → A+B+C).
    px_count is summed; area_m2 reflects the merged geometry.
    Only considers merging a polygon into a strictly larger one (by area).
    """
    if threshold <= 0 or len(gdf) < 2:
        return gdf

    gdf = gdf.copy().reset_index(drop=True)
    n = len(gdf)
    areas = gdf.geometry.area

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    sindex = gdf.sindex
    for i in range(n):
        geom_i = gdf.geometry.iloc[i]
        perim_i = geom_i.length
        if perim_i == 0:
            continue
        for j in list(sindex.intersection(geom_i.bounds)):
            if j == i or areas.iloc[j] <= areas.iloc[i]:
                continue
            geom_j = gdf.geometry.iloc[j]
            if not geom_i.intersects(geom_j):
                continue
            shared = geom_i.boundary.intersection(geom_j.boundary).length
            if shared / perim_i >= threshold:
                uf_union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    new_rows = []
    for members in groups.values():
        if len(members) == 1:
            new_rows.append(gdf.iloc[members[0]].to_dict())
        else:
            merged_geom = fill_holes(
                unary_union([gdf.geometry.iloc[m] for m in members])
            )
            largest = max(members, key=lambda m: areas.iloc[m])
            row = gdf.iloc[largest].to_dict()
            row["geometry"] = merged_geom
            row["area_m2"] = float(merged_geom.area)
            row["px_count"] = float(sum(gdf["px_count"].iloc[m] for m in members))
            new_rows.append(row)

    gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
    return gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)


def apply_buffer_and_clean(gdf, buff_dist, merge_threshold, perimeter_share_threshold=0.0):
    """
    Post-process lake polygons in five steps:

      1. Fill interior holes (gaps) in every polygon.
      2. If buff_dist > 0: buffer outward by buff_dist metres.
      3. Merge overlapping small lakes — both lakes must have a pre-buffer
         area strictly below merge_threshold.  Connectivity is resolved via
         union-find so transitive groups (A↔B, B↔C → A+B+C) are captured.
         px_count is summed across merged members.
      4. Resolve any remaining overlaps using a greedy largest-first clip:
         polygons are iterated in descending area order; each polygon has the
         already-claimed area subtracted before it is added to the output.
         Lakes clipped to empty are dropped.
      5. If perimeter_share_threshold > 0: absorb any polygon that shares
         >= that fraction of its perimeter with a single larger neighbour
         (see _merge_by_perimeter_share).

    The input GeoDataFrame must already be in a metric CRS and must contain
    columns 'area_m2' and 'px_count'.  Returns a new GeoDataFrame with updated
    geometry and area_m2; all other columns are preserved.
    """
    gdf = gdf.copy().reset_index(drop=True)

    # Step 1 – fill holes
    gdf["geometry"] = gdf.geometry.apply(fill_holes)

    if buff_dist <= 0:
        gdf["area_m2"] = gdf.geometry.area
        return _merge_by_perimeter_share(gdf, perimeter_share_threshold)

    # Preserve pre-buffer areas for merge-threshold decisions
    orig_area = gdf["area_m2"].copy()

    # Step 2 – buffer
    gdf["geometry"] = gdf.geometry.buffer(buff_dist)
    gdf["area_m2"] = gdf.geometry.area

    # Step 3 – union-find: merge overlapping small lakes
    n = len(gdf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    sindex = gdf.sindex
    for i in range(n):
        if orig_area.iloc[i] >= merge_threshold:
            continue
        for j in list(sindex.intersection(gdf.geometry.iloc[i].bounds)):
            if j <= i or orig_area.iloc[j] >= merge_threshold:
                continue
            if gdf.geometry.iloc[i].intersects(gdf.geometry.iloc[j]):
                uf_union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged_rows = []
    for members in groups.values():
        if len(members) == 1:
            merged_rows.append(gdf.iloc[members[0]].to_dict())
        else:
            merged_geom = fill_holes(
                unary_union([gdf.geometry.iloc[m] for m in members])
            )
            largest = max(members, key=lambda m: orig_area.iloc[m])
            row = gdf.iloc[largest].to_dict()
            row["geometry"] = merged_geom
            row["area_m2"] = float(merged_geom.area)
            row["px_count"] = float(sum(gdf["px_count"].iloc[m] for m in members))
            merged_rows.append(row)

    gdf = gpd.GeoDataFrame(merged_rows, crs=gdf.crs)
    gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)

    # Step 4 – greedy largest-first overlap resolution
    claimed_union = None
    resolved_rows = []
    for _, row in gdf.iterrows():
        geom = row["geometry"]
        if claimed_union is not None:
            geom = geom.difference(claimed_union)
        if geom is None or geom.is_empty:
            continue
        geom = fill_holes(geom)
        claimed_union = geom if claimed_union is None else claimed_union.union(geom)
        d = row.to_dict()
        d["geometry"] = geom
        d["area_m2"] = float(geom.area)
        resolved_rows.append(d)

    gdf = gpd.GeoDataFrame(resolved_rows, crs=gdf.crs).reset_index(drop=True)

    # Step 5 – perimeter-share merge
    return _merge_by_perimeter_share(gdf, perimeter_share_threshold)


# ============================================================
# 3) EXTRACTION AND VECTORIZATION
# ============================================================

def extract_lake_polygons(prefix, id_letter, names, aoi_shp=None, intersector_shp=None):
    """
    Load the lake persistency raster for *prefix*, threshold at SCORE_THRESHOLD,
    label connected pixel groups with 8-connectivity, vectorize to polygons,
    then assign lake_id and name fields.

    names           - ordered list of name strings for this AOI.
    aoi_shp         - optional Path to an AOI boundary shapefile; lake polygons
                      are clipped to this extent after buffering.
    intersector_shp - optional Path to a shapefile whose polygons mark invalid
                      lake locations; any lake that intersects these is dropped.
    Returns GeoDataFrame.
    """
    raster_path = PERSISTENCY_RASTER_DIR / f"{prefix}_lake_persistency_converted.tif"
    if not raster_path.exists():
        raise FileNotFoundError(
            f"[{prefix}] Lake persistency raster not found: {raster_path}\n"
            "Run plot_heatmap.py with SAVE_PERSISTENCY_RASTERS = True first."
        )

    with rasterio.open(raster_path) as src:
        lake_arr  = src.read(1).astype(np.int32)
        transform = src.transform
        raster_crs = src.crs

    # Binary mask: pixels with persistency score strictly above threshold
    mask = (lake_arr > SCORE_THRESHOLD).astype(np.uint8)
    n_above = int(mask.sum())
    print(f"[{prefix}] Pixels with score > {SCORE_THRESHOLD}: {n_above}")

    if n_above == 0:
        warnings.warn(f"[{prefix}] No pixels exceed the score threshold; skipping.")
        return gpd.GeoDataFrame()

    # Label connected components using 8-connectivity
    struct = ndimage.generate_binary_structure(2, 2)
    labeled, n_components = ndimage.label(mask, structure=struct)
    print(f"[{prefix}] Connected components found: {n_components}")

    # Vectorize: one polygon per unique label value (= one per lake component)
    polygons = []
    for geom_dict, value in rasterio.features.shapes(
        labeled.astype(np.int32),
        mask=(labeled > 0).astype(np.uint8),
        transform=transform,
    ):
        polygons.append({"component": int(value), "geometry": shape(geom_dict)})

    if not polygons:
        warnings.warn(f"[{prefix}] No polygons generated; skipping.")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(polygons, crs=raster_crs if raster_crs else TARGET_CRS)

    # Dissolve per component label in case shapes() returned multiple rings
    gdf = gdf.dissolve(by="component").reset_index()

    # Compute pixel count from polygon area and pixel resolution
    pixel_res_m = abs(transform.a)
    gdf["px_count"] = (gdf.geometry.area / (pixel_res_m ** 2)).round(1)

    gdf = gdf[gdf["px_count"] >= MIN_PIXEL_COUNT].copy()
    print(f"[{prefix}] Lakes after min-pixel filter ({MIN_PIXEL_COUNT} px): {len(gdf)}")

    if gdf.empty:
        warnings.warn(f"[{prefix}] All polygons filtered as artefacts; skipping.")
        return gpd.GeoDataFrame()

    # Reproject to target CRS if needed
    if gdf.crs and str(gdf.crs).upper() != TARGET_CRS.upper():
        gdf = gdf.to_crs(TARGET_CRS)

    gdf["area_m2"] = gdf.geometry.area

    # Fill holes, apply optional buffer, merge small overlapping lakes,
    # and resolve any remaining overlaps.  area_m2 reflects the final geometry.
    merge_thr = MERGE_THRESHOLDS.get(prefix, 0)
    gdf = apply_buffer_and_clean(gdf, BUFF_DIST, merge_thr, PERIMETER_SHARE_THRESHOLD)
    if BUFF_DIST > 0 or PERIMETER_SHARE_THRESHOLD > 0:
        print(
            f"[{prefix}] Lakes after buffer / merge / clip / perimeter-merge: {len(gdf)}"
        )

    # Clip to AOI boundary so no buffered edge extends beyond the study area
    if aoi_shp is not None:
        aoi_gdf = gpd.read_file(aoi_shp)
        if str(aoi_gdf.crs).upper() != TARGET_CRS.upper():
            aoi_gdf = aoi_gdf.to_crs(TARGET_CRS)
        aoi_union = aoi_gdf.geometry.union_all()
        gdf = gdf.clip(aoi_union).reset_index(drop=True)
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf["geometry"] = gdf.geometry.apply(fill_holes)
        gdf["area_m2"] = gdf.geometry.area
        print(f"[{prefix}] Lakes after AOI clip: {len(gdf)}")

    # Remove any lake that intersects the intersector shapefile
    if intersector_shp is not None:
        inter_gdf = gpd.read_file(intersector_shp)
        if str(inter_gdf.crs).upper() != TARGET_CRS.upper():
            inter_gdf = inter_gdf.to_crs(TARGET_CRS)
        inter_union = inter_gdf.geometry.union_all()
        bad = gdf.geometry.intersects(inter_union)
        n_removed = int(bad.sum())
        gdf = gdf[~bad].reset_index(drop=True)
        print(f"[{prefix}] {n_removed} lake(s) removed by intersector filter")

    if gdf.empty:
        warnings.warn(f"[{prefix}] No lakes remain after spatial filtering; skipping.")
        return gpd.GeoDataFrame()

    # Sort largest-first so the most persistent lake gets the lowest ID number
    gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)

    n_lakes = len(gdf)
    gdf["lake_id"] = [f"{id_letter}{i + 1:03d}" for i in range(n_lakes)]
    gdf["name"] = [
        names[i] if i < len(names) else f"{id_letter}Lake{i + 1:03d}"
        for i in range(n_lakes)
    ]

    gdf = gdf[["lake_id", "name", "area_m2", "px_count", "geometry"]].copy()

    print(
        f"[{prefix}] {n_lakes} lake(s) → "
        f"{gdf['lake_id'].iloc[0]} … {gdf['lake_id'].iloc[-1]}"
    )
    return gdf


# ============================================================
# 4) POLYGON → POINT CONVERSION
# ============================================================

def polygons_to_points(gdf):
    """
    Derive a point GeoDataFrame from a polygon GeoDataFrame.
    Uses representative_point() so every point is guaranteed to lie
    inside its polygon (unlike centroid, which can fall outside for
    concave or irregular shapes).  All attribute fields are preserved.
    """
    pt_gdf = gdf.copy()
    pt_gdf["geometry"] = gdf.geometry.apply(lambda g: g.representative_point())
    return pt_gdf


# ============================================================
# 5) MAIN
# ============================================================

def main():
    if not LAKE_NAMES_CSV.exists():
        raise FileNotFoundError(f"Lake names CSV not found: {LAKE_NAMES_CSV}")
    names_df = pd.read_csv(LAKE_NAMES_CSV)

    all_gdfs = []

    for cfg in AOI_CONFIGS:
        col = cfg["name_col"]
        if col not in names_df.columns:
            raise KeyError(f"Column '{col}' not found in {LAKE_NAMES_CSV.name}")
        names = names_df[col].dropna().tolist()

        gdf = extract_lake_polygons(
            cfg["prefix"], cfg["id_letter"], names,
            aoi_shp=cfg.get("aoi_shp"),
            intersector_shp=cfg.get("intersector_shp"),
        )
        if not gdf.empty:
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("No lake polygons extracted from any AOI. Exiting.")
        return

    PERSISTENT_LAKES_DIR.mkdir(parents=True, exist_ok=True)

    for cfg, gdf in zip(AOI_CONFIGS, all_gdfs):
        out_path = PERSISTENT_LAKES_DIR / f"{cfg['prefix']}_persistent_lake_polygons.shp"
        gdf.to_file(out_path)
        print(f"\n[{cfg['prefix']}] Polygon shapefile written → {out_path}")

        pt_path = PERSISTENT_LAKES_DIR / f"{cfg['prefix']}_persistent_lake_points.shp"
        polygons_to_points(gdf).to_file(pt_path)
        print(f"[{cfg['prefix']}] Point shapefile written   → {pt_path}")

        print(f"[{cfg['prefix']}] Total lakes: {len(gdf)}")
        print(gdf[["lake_id", "name", "area_m2", "px_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
