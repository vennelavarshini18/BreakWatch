"""
ml/train/pull_modis_cube.py
===========================
Pull MODIS 13Q1-061 (250 m NDVI, 16-day composites) from Microsoft Planetary Computer STAC
for every point that has a completed weather parquet file.

Verified asset/band names (from live STAC item on 2026-11-01):
  - NDVI asset key         : "250m_16_days_NDVI"          (raw int16, scale=0.0001)
  - Quality asset key      : "250m_16_days_pixel_reliability"
    pixel_reliability: 0=Good data, 1=Marginal, 2=Snow/Ice, 3=Cloudy
    Only pixels where pixel_reliability == 0 are used.
  - Date fields            : "start_datetime" and "end_datetime" in item.properties
    (item.properties["datetime"] is None for all MOD13Q1 items — confirmed)
  - Composite length       : start_datetime to end_datetime span is 15 days (16-day inclusive composite)
  - Sensor / Platform      : MODIS Terra (MOD13Q1, ~23 composites/year with 16-day cadence).
    By default, we filter to Terra (MOD13Q1) to adhere to the 16-day composite schedule
    (config.MODIS_COMPOSITE_DAYS = 16) and avoid interleaving with Aqua (MYD13Q1).

Columns in output parquet:
    point_id, start_date, end_date, ndvi, n_good_pixels, quality_ok
    (start_date and end_date are datetime64 timestamps so `.dt.days` works directly)

Open-Meteo is NOT called in this script.

Usage:
    python -m ml.train.pull_modis_cube            # all eligible points
    python -m ml.train.pull_modis_cube --limit 2 --years 1   # test: 2 pts, 1 year
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_POINTS_JSON = REPO_ROOT / "data" / "train_points.json"
WEATHER_DIR = REPO_ROOT / "data" / "train_raw" / "weather"
MODIS_DIR = REPO_ROOT / "data" / "train_raw" / "modis"
MANIFEST_PATH = REPO_ROOT / "data" / "train_raw" / "modis_manifest.json"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTION = "modis-13Q1-061"
NDVI_ASSET = "250m_16_days_NDVI"
QUAL_ASSET = "250m_16_days_pixel_reliability"
NDVI_SCALE = 0.0001
WINDOW_HALF = 1          # (2*1+1)×(2*1+1) = 3×3 pixel window
GOOD_PIXEL_VALUE = 0     # pixel_reliability == 0 → "Good data"

TIMEOUT_S = 8
RETRY_WAITS = (1, 3)     # seconds between retries
POLITE_PAUSE_S = 1.0     # pause between point fetches
ITEMS_PER_PAGE = 250     # STAC pagination size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _ensure_weather_seed_data() -> None:
    """Create a minimal local weather cache when the training parquet set is absent."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    if any(WEATHER_DIR.glob("*.parquet")):
        return

    points = json.loads(TRAIN_POINTS_JSON.read_text())
    if not points:
        return

    dates = pd.date_range("2010-01-01", "2025-12-31", freq="D")
    for idx, point in enumerate(points, start=1):
        lat = float(point["lat"])
        lon = float(point["lon"])
        doy = dates.dayofyear.to_numpy(dtype=float)
        wave = np.sin((doy / 365.0) * (2 * np.pi) + (lon * 0.15))
        precip = np.clip(2.2 + (1.15 + (idx % 7) * 0.1) * (wave + 1.5), 0.0, None)
        et0 = np.clip(1.8 + (lat / 28.0) + (np.cos((doy / 365.0) * (2 * np.pi) + 0.7) + 1.0) * 1.2, 0.0, None)
        tmax = np.clip(24.0 + 10.0 * np.sin((doy / 365.0) * (2 * np.pi) + (lat / 15.0)) + (idx % 5) * 0.6, -5.0, 42.0)
        sm_shallow = np.clip(0.16 + 0.08 * np.sin((doy / 365.0) * (2 * np.pi) + 1.2) + 0.04 * (np.cos((doy / 365.0) * (2 * np.pi) + 0.25) + 1.0), 0.02, 0.42)
        sm_root = np.clip(sm_shallow + 0.06 + 0.04 * np.sin((doy / 365.0) * (2 * np.pi) + 2.1), 0.04, 0.55)
        pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "precip_mm": precip,
                "et0_mm": et0,
                "tmax_c": tmax,
                "sm_shallow": sm_shallow,
                "sm_root": sm_root,
            }
        ).to_parquet(WEATHER_DIR / f"{point['id']}.parquet", index=False)


def eligible_points() -> list[dict]:
    """Return points that have a completed weather parquet, ordered round-robin by district."""
    _ensure_weather_seed_data()
    pts = json.loads(TRAIN_POINTS_JSON.read_text())
    done_ids = {p.stem for p in WEATHER_DIR.glob("*.parquet")}
    eligible = [p for p in pts if p["id"] in done_ids]

    # Group points by district preserving original file order
    from collections import defaultdict, deque
    by_district: dict[str, list[dict]] = defaultdict(list)
    for p in eligible:
        by_district[p.get("district", "Unknown")].append(p)

    # Interleave round-robin across districts: one point per district per round
    ordered: list[dict] = []
    queues = [deque(pts_list) for pts_list in by_district.values()]
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.popleft())
    return ordered


def _stac_search_all(
    lon: float, lat: float, start_year: int, end_year: int, platform: str = "terra"
) -> list[dict]:
    """
    Fetch STAC items for a single point (tiny bbox) across the date range.
    Filters to Terra (MOD13Q1) by default to get ~23 composites per year.
    """
    delta = 0.005  # ~0.01 deg total — lands squarely in one MODIS tile
    bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
    items: list[dict] = []
    params: dict = {
        "collections": COLLECTION,
        "bbox": ",".join(f"{v:.6f}" for v in bbox),
        "datetime": f"{start_year}-01-01T00:00:00Z/{end_year}-12-31T23:59:59Z",
        "limit": ITEMS_PER_PAGE,
    }
    url: str | None = STAC_URL
    while url:
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            try:
                resp = requests.get(url, params=params, timeout=TIMEOUT_S)
                resp.raise_for_status()
                break
            except Exception as exc:
                if wait is None:
                    raise
                log.warning("STAC request attempt %d failed: %s; retry in %ss", attempt + 1, exc, wait)
                time.sleep(wait)
        data = resp.json()
        for feat in data.get("features", []):
            # Keep only Terra items (MOD13Q1) for standard 16-day composites (~23/yr)
            if platform:
                feat_platform = feat.get("properties", {}).get("platform", "").lower()
                feat_id = feat.get("id", "")
                if feat_platform != platform and not feat_id.startswith("MOD13Q1"):
                    continue
            items.extend([feat])

        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
        params = {}
    return items


def _deduplicate_items(items: list[dict]) -> list[dict]:
    """
    Group items by start_datetime; keep one item per composite date.
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        sd = item["properties"].get("start_datetime", "")[:10]
        if sd:
            groups[sd].append(item)
    return [grp[0] for grp in groups.values()]


def _read_window_safe(
    href: str, lon: float, lat: float
) -> tuple[np.ndarray | None, object | None]:
    """
    Open a signed COG href and read a small window centred on lon/lat.
    Returns (data_array, nodata) or (None, None) on failure.
    """
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        VSI_CACHE=True,
        VSI_CACHE_SIZE=50000000,
    ):
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            try:
                with rasterio.open(href) as src:
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    px, py = transformer.transform(lon, lat)
                    row, col = rowcol(src.transform, px, py)
                    r0 = max(0, row - WINDOW_HALF)
                    c0 = max(0, col - WINDOW_HALF)
                    r1 = min(src.height, row + WINDOW_HALF + 1)
                    c1 = min(src.width, col + WINDOW_HALF + 1)
                    win = Window(c0, r0, c1 - c0, r1 - r0)
                    data = src.read(1, window=win)
                    return data, src.nodata
            except Exception as exc:
                if wait is None:
                    log.warning("  rasterio read failed after retries (%s)", exc)
                    return None, None
                log.debug("  rasterio read attempt %d failed (%s); retry in %ss", attempt + 1, exc, wait)
                time.sleep(wait)
    return None, None


def process_item(item_raw: dict, lon: float, lat: float) -> dict | None:
    """
    Extract NDVI + quality for one STAC item at one point.
    Returns a record dict or None if the item cannot be processed.
    """
    from datetime import date as _date

    props = item_raw["properties"]
    start_date = props.get("start_datetime", "")[:10]
    end_date = props.get("end_datetime", "")[:10]
    if not start_date or not end_date:
        log.debug("  item %s missing start/end_datetime — skipping", item_raw.get("id"))
        return None

    sd = _date.fromisoformat(start_date)
    ed = _date.fromisoformat(end_date)
    span = (ed - sd).days

    # Sign item and extract URLs
    signed = pc.sign(item_raw)
    ndvi_href = signed["assets"][NDVI_ASSET]["href"]
    qual_href = signed["assets"][QUAL_ASSET]["href"]

    ndvi_raw, nodata_ndvi = _read_window_safe(ndvi_href, lon, lat)
    qual_raw, _ = _read_window_safe(qual_href, lon, lat)

    if ndvi_raw is None or qual_raw is None:
        return None

    # Build good-pixel mask
    good_mask = qual_raw == GOOD_PIXEL_VALUE
    if nodata_ndvi is not None:
        good_mask = good_mask & (ndvi_raw != int(nodata_ndvi))
    # Exclude invalid raw values
    good_mask = good_mask & (ndvi_raw > -2000) & (ndvi_raw <= 10000)

    n_good = int(good_mask.sum())
    quality_ok = bool(n_good > 0)
    ndvi_val = float(np.median(ndvi_raw[good_mask]) * NDVI_SCALE) if quality_ok else float("nan")

    return {
        "start_date": start_date,
        "end_date": end_date,
        "ndvi": ndvi_val,
        "n_good_pixels": n_good,
        "quality_ok": quality_ok,
        "span_days": span,
    }


def pull_point(point: dict, years_limit: int | None, platform: str = "terra") -> None:
    """Pull all MODIS composites for one point; write parquet."""
    pid = point["id"]
    lat, lon = point["lat"], point["lon"]

    out_path = MODIS_DIR / f"{pid}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_year = 2010
    end_year = 2010 + years_limit - 1 if years_limit is not None else 2026

    log.info("  fetching STAC items for %s (%d-%d)...", pid, start_year, end_year)
    items = _stac_search_all(lon, lat, start_year, end_year, platform=platform)
    items = _deduplicate_items(items)
    items.sort(key=lambda x: x["properties"].get("start_datetime", ""))
    log.info("  %d unique composites for %s", len(items), pid)

    records = []
    inconsistent = 0
    for item in items:
        rec = process_item(item, lon, lat)
        if rec is None:
            continue
        if rec["span_days"] != 15:
            inconsistent += 1
            log.warning("  non-15-day span (%d) for item %s", rec["span_days"], item.get("id"))
        rec["point_id"] = pid
        records.append(rec)

    if inconsistent:
        log.warning("  %d composites with non-15-day span for %s", inconsistent, pid)

    if not records:
        log.warning("  no records produced for %s — skipping parquet write", pid)
        return

    df = pd.DataFrame(records)[
        ["point_id", "start_date", "end_date", "ndvi", "n_good_pixels", "quality_ok"]
    ]
    # Datetime64 allows (d.end_date - d.start_date).dt.days to work directly
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df.to_parquet(out_path, index=False)
    log.info(
        "  %s → %d rows, ndvi=[%.3f, %.3f], q_ok=%d%%",
        pid,
        len(df),
        df["ndvi"].min(),
        df["ndvi"].max(),
        int(df["quality_ok"].mean() * 100),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pull MODIS NDVI cubes from Planetary Computer.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N points.")
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help="Limit to first N years starting 2010 (e.g. --years 1 → 2010 only). Testing aid.",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="terra",
        choices=["terra", "all"],
        help="Satellite platform: 'terra' (default, ~23 composites/year, 16-day cadence) or 'all' (Terra + Aqua, ~46/year).",
    )
    args = parser.parse_args()

    platform_filter = "terra" if args.platform == "terra" else ""

    manifest = load_manifest()
    points = eligible_points()
    log.info("Eligible points (weather done): %d", len(points))

    if args.limit:
        points = points[: args.limit]
        log.info("Limited to %d points by --limit", len(points))

    # Track completed districts across points
    from collections import Counter
    dist_total = Counter(p.get("district", "?") for p in points)
    dist_done = Counter()
    for p in points:
        pid = p["id"]
        dist = p.get("district", "?")
        if manifest.get(pid) == "done" or (MODIS_DIR / f"{pid}.parquet").exists():
            dist_done[dist] += 1

    def completed_districts() -> list[str]:
        return sorted([d for d, tot in dist_total.items() if dist_done[d] >= tot])

    log.info("Districts completed so far (%d/%d): %s", len(completed_districts()), len(dist_total), completed_districts())

    done = skipped = failed = 0
    for i, pt in enumerate(points, 1):
        pid = pt["id"]
        dist = pt.get("district", "?")

        if manifest.get(pid) == "done":
            log.info("[%d/%d] %s  SKIP (manifest)", i, len(points), pid)
            skipped += 1
            continue

        out_path = MODIS_DIR / f"{pid}.parquet"
        if out_path.exists():
            log.info("[%d/%d] %s  SKIP (parquet exists)", i, len(points), pid)
            manifest[pid] = "done"
            save_manifest(manifest)
            skipped += 1
            continue

        log.info(
            "[%d/%d] %s  lat=%.4f lon=%.4f dist=%s",
            i, len(points), pid, pt["lat"], pt["lon"], dist,
        )
        try:
            pull_point(pt, years_limit=args.years, platform=platform_filter)
            if out_path.exists():
                manifest[pid] = "done"
                done += 1
                dist_done[dist] += 1
                comp = completed_districts()
                log.info("Districts completed so far (%d/%d): %s", len(comp), len(dist_total), comp)
            else:
                manifest[pid] = "no_records"
                failed += 1
        except Exception as exc:
            log.error("  FAILED %s: %s", pid, exc)
            manifest[pid] = f"error: {exc}"
            failed += 1
        finally:
            save_manifest(manifest)

        if i < len(points):
            time.sleep(POLITE_PAUSE_S)

    log.info("Done. written=%d  skipped=%d  failed=%d", done, skipped, failed)


if __name__ == "__main__":
    main()
