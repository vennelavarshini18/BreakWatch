"""
ml/train/sample_points.py — Sample ~80 cropland training points across Telangana.

Algorithm
---------
1. Query the PC STAC for ESA WorldCover 2021 (v200) tiles covering Telangana.
2. For each tile, read the GeoTIFF at coarse resolution (~0.01 deg ≈ 1000 m).
   We use rasterio's windowed overview reads; the native 10 m raster has
   overviews at 2x, 4x, 8x … We pick the overview level closest to 0.01 deg.
3. Resample to a 0.01-degree grid by counting class-40 (Cropland) pixels vs
   total pixels in each 0.01-deg cell.  Keep cells with cropland_frac >= 0.6.
4. Load Telangana district polygons from data/boundaries/telangana_districts.json
   (from https://github.com/gggodhwani/telangana_boundaries).
5. Assign each candidate cell to a district via point-in-polygon (shapely).
6. Stratified sample ~80 points proportional to each district's cropland area,
   seed 42.
7. Write data/train_points.json.

Boundary source used
--------------------
  https://github.com/gggodhwani/telangana_boundaries  (file: districts.json)
  Pre-2016 bifurcation boundaries → 10 districts.  Downloaded to
  data/boundaries/telangana_districts.json.

Usage
-----
    python -m ml.train.sample_points          # normal run
    python -m ml.train.sample_points --dry-run  # print stats only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
import rasterio.windows
import requests
from shapely.geometry import Point, shape

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
OUTPUT_JSON = DATA_DIR / "train_points.json"
DISTRICT_FILE = BOUNDARIES_DIR / "telangana_districts.json"

# Telangana bounding box (lon_min, lat_min, lon_max, lat_max)
TELANGANA_BBOX = (77.2, 15.8, 81.8, 19.9)
CROPLAND_CLASS = 40         # ESA WorldCover class for cropland
CROPLAND_MIN_FRAC = 0.6     # keep grid cells with >= 60 % cropland
TARGET_N_POINTS = 80
COARSE_DEG = 0.01           # target grid resolution in degrees
RANDOM_SEED = 42

# PC STAC endpoint — no auth required for ESA WorldCover public tiles
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WC_COLLECTION = "esa-worldcover"
WC_YEAR = "2021"            # use v200 (2021) as the more recent product

HTTP_TIMEOUT = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stac_search(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Return signed STAC items for ESA WorldCover 2021 tiles intersecting *bbox*.

    Items are signed via planetary_computer so that asset HREFs carry SAS tokens
    that rasterio/GDAL can open without 409 errors from Azure Blob Storage.
    """
    url = f"{PC_STAC_URL}/search"
    params = {
        "collections": WC_COLLECTION,
        "bbox": ",".join(str(x) for x in bbox),
        "limit": 100,
        "datetime": f"{WC_YEAR}-01-01/{WC_YEAR}-12-31",
    }
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    items = resp.json().get("features", [])
    # Filter to v200 (2021) only
    items = [it for it in items if WC_YEAR in it["id"] and "v200" in it["id"]]
    # Sign each item so its asset URLs include SAS tokens (avoids HTTP 409 from Azure)
    items = [pc.sign(it) for it in items]
    return items


def _overview_level(src: rasterio.DatasetReader, target_m: float = 1000.0) -> int:
    """Choose the overview level whose resolution is closest to *target_m* metres."""
    native_res = abs(src.transform.a)  # metres per pixel (native ≈ 10 m)
    overviews = src.overviews(1)       # list of overview factors e.g. [2,4,8,16,32]
    if not overviews:
        return 0  # no overviews — use native
    factors = [o * native_res for o in overviews]
    best_idx = int(np.argmin([abs(f - target_m) for f in factors]))
    return best_idx + 1  # rasterio uses 1-based overview index


def _read_tile_coarse(href: str, bbox: tuple) -> tuple[np.ndarray, rasterio.Affine]:
    """Read a WorldCover GeoTIFF tile at coarse resolution clipped to *bbox*.

    Returns (data_array, transform) where data_array is a 2-D uint8 array of
    WorldCover class values at the chosen overview resolution.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    with rasterio.open(href) as src:
        ovr = _overview_level(src, target_m=1100.0)  # aim for ~1 km
        # Build the window for the Telangana bbox in pixel coords
        window = rasterio.windows.from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            transform=src.transform,
        )
        # Read at the chosen overview level
        out_shape_rows = max(1, round(window.height / (2 ** ovr)))
        out_shape_cols = max(1, round(window.width / (2 ** ovr)))
        data = src.read(
            1,
            window=window,
            out_shape=(out_shape_rows, out_shape_cols),
            resampling=rasterio.enums.Resampling.nearest,
        )
        # Compute the actual transform for this coarse window
        transform = src.window_transform(window)
        # Adjust transform for the resampling factor
        scale_row = window.height / out_shape_rows
        scale_col = window.width / out_shape_cols
        coarse_transform = rasterio.transform.Affine(
            transform.a * scale_col,
            transform.b,
            transform.c,
            transform.d,
            transform.e * scale_row,
            transform.f,
        )
        return data, coarse_transform


def _load_districts() -> list[dict]:
    """Load district polygons as list of {name, geometry_shape}."""
    if not DISTRICT_FILE.exists():
        raise FileNotFoundError(
            f"District file not found: {DISTRICT_FILE}\n"
            f"Download from https://github.com/gggodhwani/telangana_boundaries "
            f"and save as data/boundaries/telangana_districts.json"
        )
    with open(DISTRICT_FILE) as f:
        fc = json.load(f)
    districts = []
    for feature in fc["features"]:
        districts.append(
            {
                "name": feature["properties"]["D_N"].title(),
                "geom": shape(feature["geometry"]),
            }
        )
    return districts


def _assign_district(lat: float, lon: float, districts: list[dict]) -> str:
    """Return district name for (lat, lon), or 'Unknown' if outside all polygons."""
    pt = Point(lon, lat)
    for d in districts:
        if d["geom"].contains(pt):
            return d["name"]
    # Fallback: nearest district centroid
    best = min(districts, key=lambda d: d["geom"].centroid.distance(pt))
    return best["name"]


def _grid_cropland(
    tiles: list[tuple[np.ndarray, rasterio.Affine]],
    bbox: tuple,
    cell_deg: float = COARSE_DEG,
) -> list[dict]:
    """Build a list of {lat, lon, cropland_frac} for cells in *bbox*.

    Uses a 0.01-degree grid.  For each cell, collects pixel class values from
    all tiles and computes the fraction that are CROPLAND_CLASS (40).
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Build grid cell centres
    lons = np.arange(lon_min + cell_deg / 2, lon_max, cell_deg)
    lats = np.arange(lat_min + cell_deg / 2, lat_max, cell_deg)

    cells: list[dict] = []
    for lat in lats:
        for lon in lons:
            counts = {"total": 0, "crop": 0}
            for data, transform in tiles:
                # Convert grid cell corners to pixel indices for this tile
                half = cell_deg / 2.0
                try:
                    row_min, col_min = rasterio.transform.rowcol(
                        transform, lon - half, lat + half
                    )
                    row_max, col_max = rasterio.transform.rowcol(
                        transform, lon + half, lat - half
                    )
                except Exception:
                    continue
                # Clip to valid pixel range
                nrows, ncols = data.shape
                r0 = max(0, min(row_min, row_max))
                r1 = min(nrows, max(row_min, row_max) + 1)
                c0 = max(0, min(col_min, col_max))
                c1 = min(ncols, max(col_min, col_max) + 1)
                if r1 <= r0 or c1 <= c0:
                    continue
                patch = data[r0:r1, c0:c1]
                counts["total"] += patch.size
                counts["crop"] += int(np.sum(patch == CROPLAND_CLASS))

            if counts["total"] == 0:
                continue
            frac = counts["crop"] / counts["total"]
            if frac >= CROPLAND_MIN_FRAC:
                cells.append({"lat": float(lat), "lon": float(lon), "cropland_frac": float(frac)})
    return cells


def _stratified_sample(
    cells: list[dict],
    districts: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Stratify by district, sample proportionally, seed fixed.

    Returns a list of point dicts with id, lat, lon, district, cropland_frac.
    """
    rng = np.random.default_rng(seed)

    # Assign districts
    for c in cells:
        c["district"] = _assign_district(c["lat"], c["lon"], districts)

    # Group by district
    by_district: dict[str, list[dict]] = {}
    for c in cells:
        by_district.setdefault(c["district"], []).append(c)

    dist_names = sorted(by_district.keys())
    sizes = np.array([len(by_district[d]) for d in dist_names], dtype=float)
    total = sizes.sum()
    if total == 0:
        raise ValueError("No cropland cells found — check bbox and threshold.")

    # Proportional allocation with at least 1 per district (if district has cells)
    alloc = np.maximum(1, np.round(sizes / total * n).astype(int))
    # Trim to target total
    while alloc.sum() > n:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < n:
        alloc[np.argmax(sizes)] += 1

    sampled: list[dict] = []
    idx = 1
    for dname, k in zip(dist_names, alloc):
        pool = by_district[dname]
        k = min(k, len(pool))
        chosen_idx = rng.choice(len(pool), size=k, replace=False)
        for ci in sorted(chosen_idx):
            c = pool[ci]
            sampled.append(
                {
                    "id": f"pt_{idx:04d}",
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "district": c["district"],
                    "cropland_frac": c["cropland_frac"],
                }
            )
            idx += 1

    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dry_run: bool = False) -> None:
    """Run the full sampling pipeline."""
    print(f"[sample_points] Searching STAC for ESA WorldCover {WC_YEAR} tiles …")
    items = _stac_search(TELANGANA_BBOX)
    if not items:
        raise RuntimeError(
            f"No WorldCover {WC_YEAR} items found for bbox {TELANGANA_BBOX}. "
            "Check STAC connectivity."
        )
    print(f"[sample_points] Found {len(items)} tile(s): {[it['id'] for it in items]}")

    print("[sample_points] Reading tiles at coarse resolution (~1 km) …")
    tiles: list[tuple[np.ndarray, rasterio.Affine]] = []
    for item in items:
        href = item["assets"]["map"]["href"]
        # Extract bare filename from URL before the '?' (SAS token query string)
        bare_name = href.split("?")[0].split("/")[-1]
        print(f"  reading {bare_name} …", end=" ", flush=True)
        data, transform = _read_tile_coarse(href, TELANGANA_BBOX)
        tiles.append((data, transform))
        print(f"shape={data.shape}")

    print(f"[sample_points] Gridding at {COARSE_DEG}° resolution …")
    cells = _grid_cropland(tiles, TELANGANA_BBOX, cell_deg=COARSE_DEG)
    print(f"[sample_points] Cropland cells (>= {CROPLAND_MIN_FRAC:.0%}): {len(cells)}")

    if dry_run:
        print("[sample_points] --dry-run: stopping before sampling.")
        return

    print("[sample_points] Loading district boundaries …")
    districts = _load_districts()
    print(f"[sample_points] Districts: {[d['name'] for d in districts]}")

    print(f"[sample_points] Stratified sampling {TARGET_N_POINTS} points …")
    points = _stratified_sample(cells, districts, TARGET_N_POINTS, seed=RANDOM_SEED)
    print(f"[sample_points] Sampled {len(points)} points.")

    # District breakdown
    counts: dict[str, int] = {}
    for p in points:
        counts[p["district"]] = counts.get(p["district"], 0) + 1
    for dist, cnt in sorted(counts.items()):
        print(f"  {dist}: {cnt}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(points, f, indent=2)
    print(f"[sample_points] Wrote {len(points)} points → {OUTPUT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample training points from ESA WorldCover.")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing output.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
