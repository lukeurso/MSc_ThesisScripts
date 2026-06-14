# -*- coding: utf-8 -*-
"""
analyze_drainage_events.py

Detects rapid lake drainage events from per-lake volume time series produced by
create_per_lake_volume_csv_data.py, then writes event-level and per-window data
products as both CSVs and point shapefiles.

Drainage detection
------------------
For each lake independently, consecutive *valid* (non-NaN) observations are
compared. A NaN cell means that window had insufficient footprint coverage for
that lake — it is skipped, not treated as zero volume. The gap between two
valid observations is the raw row-index distance in the volume CSV.

  Full drainage   : volume loss ≥ FULL_DRAIN_THRESHOLD  (default 80 %)
  Partial drainage: volume loss ≥ PARTIAL_DRAIN_THRESHOLD (default 40 %)
  Max window gap  : pairs separated by > MAX_WINDOW_GAP rows are ignored.
  Confidence score: equals window_gap (1 = consecutive windows, most confident).

Chain events
------------
Two or more drainage events from *different* lakes whose post_win_index values
are within ±CHAIN_WINDOW_TOLERANCE of each other are grouped into a chain.
Transitively linked events share the same chain_group_id.

Outputs
-------
  drainage_events.csv              — one row per detected drainage event
  per_lake_per_window.csv          — long format: one row per (window × lake)
  drainage_events_points.shp       — event-level, lake point geometries attached
  per_lake_per_window_points.shp   — window-level long format with geometry

Shapefile column names are limited to 10 characters; the SHP_COL_RENAME dict
maps full CSV names to their abbreviated shapefile equivalents.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 1) USER SETTINGS
# ============================================================

VOLUME_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\per_lake_volume.csv"
)
DEPTH_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\per_lake_mean_depth.csv"
)
FRAC_CSV = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events\per_lake_obs_fraction.csv"
)

OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\csv_files\csv_outputs\buffered_drainage_events"
)

SHP_OUTPUT_DIR = Path(
    r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles"
)

LAKE_POINTS = {
    "PTM": Path(r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles\PTM_persistent_lake_points.shp"),
    "OST": Path(r"Q:\ThesisData\data\drainage\persistent_lakes_shapefiles\OST_persistent_lake_points.shp"),
    }

# Drainage thresholds (fraction of volume lost between two valid observations)
FULL_DRAIN_THRESHOLD    = 0.80
PARTIAL_DRAIN_THRESHOLD = 0.40

# Maximum row-index gap between two valid observations to consider as a drainage event.
# Gap of 1 = consecutive windows; gap of 5 = up to 4 NaN windows in between.
MAX_WINDOW_GAP = 5

# Two events from different lakes are considered a chain if their post_win_index
# values are within this many windows of each other.
CHAIN_WINDOW_TOLERANCE = 0

TARGET_CRS = "EPSG:3413"

# Shapefile column name map (full name → ≤10-char abbreviation).
# Applied only when writing .shp; CSV outputs keep full names.
SHP_COL_RENAME = {
    "pre_win_start":    "pre_start",
    "pre_win_end":      "pre_end",
    "post_win_start":   "post_start",
    "post_win_end":     "post_end",
    "pre_volume_m3":    "pre_vol_m3",
    "post_volume_m3":   "pst_vol_m3",
    "volume_loss_frac": "vol_loss",
    "window_gap":       "window_gap",
    "confidence_score": "conf_score",
    "chain_event":      "chain_evt",
    "chain_group_id":   "chain_grp",
    "window_index":     "win_index",
    "mean_depth_m":     "mean_dpt_m",
    "obs_fraction":     "obs_frac",
    "is_valid_obs":     "is_valid",
}


# ============================================================
# 2) HELPERS
# ============================================================

def load_and_normalize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("win_start", "win_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def write_shp(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a GeoDataFrame to shapefile, applying SHP_COL_RENAME first."""
    out = gdf.rename(columns=SHP_COL_RENAME)
    out.to_file(path)
    print(f"Written: {path}")


# ============================================================
# 3) BAD LAKE FILTER
# ============================================================

def filter_bad_lakes(
    vol_df:   pd.DataFrame,
    depth_df: pd.DataFrame,
    frac_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove lake columns where depth is zero while lake pixels are present.

    A lake is considered bad if any window has obs_fraction > 0 AND
    mean_depth == 0 simultaneously — this indicates an upstream error in the
    depth computation (pixels exist but no positive depth was produced).

    All three DataFrames share the same lake columns; bad columns are dropped
    from all three together.
    """
    lake_cols = [c for c in vol_df.columns if c not in ("win_start", "win_end")]

    bad_lakes = [
        lake for lake in lake_cols
        if np.any((frac_df[lake].values > 0) & (depth_df[lake].values == 0))
    ]

    if bad_lakes:
        print(f"  Removing {len(bad_lakes)} bad lake(s) with depth=0 "
              f"when pixels present: {', '.join(bad_lakes)}")
        vol_df   = vol_df.drop(columns=bad_lakes)
        depth_df = depth_df.drop(columns=bad_lakes)
        frac_df  = frac_df.drop(columns=bad_lakes)
    else:
        print("  No bad lakes found.")

    return vol_df, depth_df, frac_df


# ============================================================
# 4) DRAINAGE EVENT DETECTION  (per-lake)
# ============================================================

def detect_drainage_events(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan each lake's volume series for drainage events.

    Valid observations are identified per lake independently — NaN means no
    usable observation for that lake in that window, not a zero-volume reading.

    Returns a DataFrame with one row per detected event.
    """
    windows  = vol_df[["win_start", "win_end"]].reset_index(drop=True)
    lake_cols = [c for c in vol_df.columns if c not in ("win_start", "win_end")]

    events = []

    for lake_id in lake_cols:
        series    = vol_df[lake_id].values
        valid_idx = np.where(~np.isnan(series))[0]

        if len(valid_idx) < 2:
            continue

        for k in range(len(valid_idx) - 1):
            i   = valid_idx[k]
            j   = valid_idx[k + 1]
            gap = int(j - i)   # 1 = consecutive windows

            if gap > MAX_WINDOW_GAP:
                continue

            pre_vol  = float(series[i])
            post_vol = float(series[j])

            if pre_vol <= 0:
                continue  # cannot compute a meaningful loss fraction

            loss_frac = (pre_vol - post_vol) / pre_vol

            if loss_frac >= FULL_DRAIN_THRESHOLD:
                drain_type = "full"
            elif loss_frac >= PARTIAL_DRAIN_THRESHOLD:
                drain_type = "partial"
            else:
                continue

            events.append({
                "lake_id":           lake_id,
                "drain_type":        drain_type,
                "pre_win_start":     windows.loc[i, "win_start"],
                "pre_win_end":       windows.loc[i, "win_end"],
                "post_win_start":    windows.loc[j, "win_start"],
                "post_win_end":      windows.loc[j, "win_end"],
                "pre_win_index":     i,
                "post_win_index":    j,
                "pre_volume_m3":     round(pre_vol,  2),
                "post_volume_m3":    round(post_vol, 2),
                "volume_loss_frac":  round(loss_frac, 4),
                "window_gap":        gap,
                "confidence_score":  gap,
            })

    return pd.DataFrame(events) if events else pd.DataFrame(columns=[
        "lake_id", "drain_type",
        "pre_win_start", "pre_win_end", "post_win_start", "post_win_end",
        "pre_win_index", "post_win_index",
        "pre_volume_m3", "post_volume_m3", "volume_loss_frac",
        "window_gap", "confidence_score",
    ])


# ============================================================
# 5) CHAIN EVENT DETECTION
# ============================================================

def assign_chain_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag events from different lakes whose post_win_index values fall within
    ±CHAIN_WINDOW_TOLERANCE of each other as chain events, and assign a shared
    chain_group_id (integer, starting at 1).

    CHAIN_WINDOW_TOLERANCE == 0  →  O(n) groupby: events on the exact same window.
    CHAIN_WINDOW_TOLERANCE  > 0  →  O(n × 2T) sorted sliding window + union-find.
    """
    df = events_df.copy().reset_index(drop=True)
    df["chain_event"]    = False
    df["chain_group_id"] = pd.NA

    if df.empty:
        return df

    if CHAIN_WINDOW_TOLERANCE == 0:
        n_lakes = df.groupby("post_win_index")["lake_id"].transform("nunique")
        df["chain_event"] = n_lakes >= 2
        chain_wins   = sorted(df.loc[df["chain_event"], "post_win_index"].unique())
        win_to_group = {w: gid for gid, w in enumerate(chain_wins, start=1)}
        df.loc[df["chain_event"], "chain_group_id"] = (
            df.loc[df["chain_event"], "post_win_index"].map(win_to_group)
        )
        return df

    # Tolerance > 0: sorted sliding window with inline path-compressed union-find
    df_s  = df.sort_values("post_win_index").reset_index()   # "index" = original row
    wins  = df_s["post_win_index"].to_numpy()
    orig  = df_s["index"].to_numpy(dtype=int)
    lakes = df_s["lake_id"].to_numpy()

    parent: dict[int, int] = {}
    linked: set[int] = set()

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    l = 0
    for r in range(len(wins)):
        while wins[r] - wins[l] > CHAIN_WINDOW_TOLERANCE:
            l += 1
        for j in range(l, r):
            if lakes[r] != lakes[j]:
                a, b = int(orig[r]), int(orig[j])
                linked.update((a, b))
                union(a, b)

    if not linked:
        return df

    components: dict[int, set] = {}
    for node in linked:
        components.setdefault(find(node), set()).add(node)

    chain_map = {
        idx: gid
        for gid, members in enumerate(
            (m for m in components.values() if len(m) >= 2), start=1
        )
        for idx in members
    }

    if chain_map:
        df["chain_event"]    = df.index.isin(chain_map)
        df["chain_group_id"] = df.index.map(lambda x: chain_map.get(x, pd.NA))

    return df


# ============================================================
# 6) PER-LAKE-PER-WINDOW LONG TABLE
# ============================================================

def build_per_lake_per_window(
    vol_df:    pd.DataFrame,
    depth_df:  pd.DataFrame,
    frac_df:   pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Melt volume, mean depth, and obs_fraction to long format, then join drainage
    flags from events_df (keyed on lake_id + post_win_index).

    One row per (window × lake). Rows with NaN volume are retained — they
    represent windows where that lake had insufficient footprint coverage.
    """
    lake_cols = [c for c in vol_df.columns if c not in ("win_start", "win_end")]

    # Tag each row with its integer window index before melting
    def add_win_idx(df):
        out = df.copy()
        out["window_index"] = range(len(df))
        return out

    vol_df   = add_win_idx(vol_df)
    depth_df = add_win_idx(depth_df)
    frac_df  = add_win_idx(frac_df)

    id_vars_full  = ["win_start", "win_end", "window_index"]
    id_vars_index = ["window_index"]

    long = vol_df.melt(
        id_vars=id_vars_full, value_vars=lake_cols,
        var_name="lake_id", value_name="volume_m3",
    )
    long = long.merge(
        depth_df.melt(id_vars=id_vars_index, value_vars=lake_cols,
                      var_name="lake_id", value_name="mean_depth_m"),
        on=["window_index", "lake_id"], how="left",
    )
    long = long.merge(
        frac_df.melt(id_vars=id_vars_index, value_vars=lake_cols,
                     var_name="lake_id", value_name="obs_fraction"),
        on=["window_index", "lake_id"], how="left",
    )

    long["is_valid_obs"] = long["volume_m3"].notna()

    # Join drainage flags: event is attached to the post_win_index row
    if not events_df.empty:
        flags = (
            events_df[["lake_id", "post_win_index", "drain_type",
                        "window_gap", "confidence_score",
                        "chain_event", "chain_group_id"]]
            .copy()
            .rename(columns={"post_win_index": "window_index"})
        )
        # If a window somehow has both full and partial events for one lake,
        # keep the more severe (full < partial in severity ordering).
        flags["_sev"] = flags["drain_type"].map({"full": 0, "partial": 1})
        flags = (
            flags.sort_values("_sev")
            .drop_duplicates(["lake_id", "window_index"])
            .drop(columns="_sev")
        )
        long = long.merge(flags, on=["lake_id", "window_index"], how="left")
        long["drain_type"]  = long["drain_type"].fillna("none")
        long["chain_event"] = long["chain_event"].fillna(False)
    else:
        long["drain_type"]       = "none"
        long["window_gap"]       = pd.NA
        long["confidence_score"] = pd.NA
        long["chain_event"]      = False
        long["chain_group_id"]   = pd.NA

    col_order = [
        "win_start", "win_end", "window_index", "lake_id",
        "volume_m3", "mean_depth_m", "obs_fraction", "is_valid_obs",
        "drain_type", "window_gap", "confidence_score",
        "chain_event", "chain_group_id",
    ]
    return long[col_order].sort_values(["window_index", "lake_id"]).reset_index(drop=True)


# ============================================================
# 7) LAKE POINT GEOMETRY
# ============================================================

def load_lake_points() -> gpd.GeoDataFrame:
    """Load and combine PTM and OST lake point shapefiles into one GeoDataFrame."""
    gdfs = []
    for prefix, shp_path in LAKE_POINTS.items():
        if not shp_path.exists():
            warnings.warn(f"[{prefix}] Lake points file not found: {shp_path}")
            continue
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(TARGET_CRS)
        elif str(gdf.crs).upper() != TARGET_CRS.upper():
            gdf = gdf.to_crs(TARGET_CRS)
        gdfs.append(gdf)

    if not gdfs:
        raise FileNotFoundError("No lake point shapefiles could be loaded.")

    combined = pd.concat(gdfs, ignore_index=True)
    if "lake_id" not in combined.columns:
        raise KeyError(
            "Lake points shapefiles must contain a 'lake_id' column to join "
            "against drainage data."
        )
    return gpd.GeoDataFrame(combined, crs=TARGET_CRS)


# ============================================================
# 8) MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ─────────────────────────────────────────
    print("Loading input CSVs…")
    vol_df   = load_and_normalize(VOLUME_CSV)
    depth_df = load_and_normalize(DEPTH_CSV)
    frac_df  = load_and_normalize(FRAC_CSV)

    n_windows = len(vol_df)
    lake_cols = [c for c in vol_df.columns if c not in ("win_start", "win_end")]
    print(f"  {n_windows} windows, {len(lake_cols)} lake columns")

    # ── Filter bad lakes ─────────────────────────────────────
    print("Filtering bad lakes (depth=0 when pixels present)…")
    vol_df, depth_df, frac_df = filter_bad_lakes(vol_df, depth_df, frac_df)
    lake_cols = [c for c in vol_df.columns if c not in ("win_start", "win_end")]
    print(f"  {len(lake_cols)} lake columns remaining")

    # ── Detect drainage events ───────────────────────────────
    print("Detecting drainage events…")
    events_df = detect_drainage_events(vol_df)
    n_full    = (events_df["drain_type"] == "full").sum()
    n_partial = (events_df["drain_type"] == "partial").sum()
    print(f"  {len(events_df)} events — {n_full} full, {n_partial} partial")

    # ── Chain detection ──────────────────────────────────────
    print("Assigning chain events…")
    events_df = assign_chain_events(events_df)
    n_chain  = int(events_df["chain_event"].sum())
    n_groups = int(events_df["chain_group_id"].nunique()) if n_chain else 0
    print(f"  {n_chain} events in {n_groups} chain group(s)")

    # ── Per-lake-per-window table ────────────────────────────
    print("Building per-lake-per-window table…")
    long_df = build_per_lake_per_window(vol_df, depth_df, frac_df, events_df)
    print(f"  {len(long_df)} rows ({n_windows} windows × {len(lake_cols)} lakes)")

    # ── Write CSVs ───────────────────────────────────────────
    events_csv_out = events_df.drop(columns=["pre_win_index", "post_win_index"])
    events_csv_out.to_csv(OUTPUT_DIR / "drainage_events.csv", index=False)
    print(f"Written: {OUTPUT_DIR / 'drainage_events.csv'}")

    long_df.to_csv(OUTPUT_DIR / "per_lake_per_window.csv", index=False)
    print(f"Written: {OUTPUT_DIR / 'per_lake_per_window.csv'}")

    # ── Load lake point geometries ───────────────────────────
    print("Loading lake point geometries…")
    points_gdf = load_lake_points()
    geom_join  = points_gdf[["lake_id", "geometry"]]

    # ── Write SHPs ───────────────────────────────────────────
    SHP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not events_df.empty:
        events_geo = events_csv_out.merge(geom_join, on="lake_id", how="left")
        write_shp(
            gpd.GeoDataFrame(events_geo, crs=TARGET_CRS),
            SHP_OUTPUT_DIR / "drainage_events_points.shp",
        )
    else:
        print("No drainage events — skipping drainage_events_points.shp")

    long_geo = long_df.merge(geom_join, on="lake_id", how="left")
    write_shp(
        gpd.GeoDataFrame(long_geo, crs=TARGET_CRS),
        SHP_OUTPUT_DIR / "per_lake_per_window_points.shp",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
