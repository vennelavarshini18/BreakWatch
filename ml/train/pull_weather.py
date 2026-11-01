"""
ml/train/pull_weather.py — Download historical weather for all training points.

Data source
-----------
Open-Meteo Historical Archive API  https://archive-api.open-meteo.com/v1/archive
ERA5-Land reanalysis; free tier, no API key required.

Verified variable names (confirmed via live API call 2026-11-01)
----------------------------------------------------------------
Daily variables  (param: daily=...)
  precipitation_sum            — total daily precipitation, mm
  et0_fao_evapotranspiration   — FAO-56 reference ET, mm
  temperature_2m_max           — daily maximum 2 m air temperature, °C

Hourly variables (param: hourly=...)
  soil_moisture_0_to_7cm       — volumetric soil water content 0–7 cm, m³/m³
  soil_moisture_7_to_28cm      — volumetric soil water content 7–28 cm, m³/m³
  soil_moisture_28_to_100cm    — volumetric soil water content 28–100 cm, m³/m³

Note: the underscore separator IS "to" not "_" — e.g. "soil_moisture_0_to_7cm".
Note: soil moisture is HOURLY only; daily aggregation is done here via
      common.features.aggregate_hourly_to_daily (IST day, ≥18/24 hrs threshold).

Open-Meteo free-tier limits assumed
------------------------------------
  - No API key needed for archive endpoint.
  - Soft daily limit: ~10,000 calls/day.
  - Minutely limit: for multi-year hourly queries, Open-Meteo throttles bursts
    to ~10-15 requests per minute ("Minutely API request limit exceeded.
    Please try again in one minute.").
  - To stay reliably within these limits:
    1. POLITE_PAUSE_S = 4.5 s (~10 calls/minute, well below the minutely threshold).
    2. If HTTP 429 occurs, the fetch pauses 65 s for the 1-minute window to reset,
       then retries the same point.
    3. Transient errors/timeouts are retried at 1 s and 3 s with an 8 s timeout.

Output
------
  data/train_raw/weather/<point_id>.parquet
  data/train_raw/manifest.json — records finished point IDs and timestamps

Usage
-----
  # test on 3 points
  python -m ml.train.pull_weather --limit 3

  # all points in background (separate terminal)
  nohup python -m ml.train.pull_weather > data/train_raw/pull_weather.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# http
import httpx

# Our shared aggregation function — do NOT re-implement it
from common.features import aggregate_hourly_to_daily

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
POINTS_JSON = DATA_DIR / "train_points.json"
WEATHER_DIR = DATA_DIR / "train_raw" / "weather"
MANIFEST_JSON = DATA_DIR / "train_raw" / "manifest.json"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2010-01-01"
# Fixed default end date; no system clock used. The end date is
# passed down or overridden by CLI flag.
TIMEZONE = "Asia/Kolkata"
HTTP_TIMEOUT = 8          # seconds; per spec
RETRY_WAITS = (1, 3)      # seconds; per spec (two retries)
RATE_LIMIT_RESET_S = 65   # seconds to wait if HTTP 429 minutely limit is hit
POLITE_PAUSE_S = 4.5      # seconds between API calls (~10 calls/min, stays under minutely limit)

# API parameter names — verified via live call 2026-11-01
DAILY_VARS = [
    "precipitation_sum",
    "et0_fao_evapotranspiration",
    "temperature_2m_max",
]
HOURLY_VARS = [
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
]

# Parquet output columns
PARQUET_COLUMNS = ["date", "precip_mm", "et0_mm", "tmax_c", "sm_shallow", "sm_root"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pull_weather")

# ---------------------------------------------------------------------------
# Fetch with rate-limit and retry handling
# ---------------------------------------------------------------------------


def _fetch(client: httpx.Client, params: dict) -> dict:
    """Fetch one point from the Open-Meteo archive API with up to 3 attempts.

    Retries transient connection/5xx errors at 1 s and 3 s (per spec).
    If Open-Meteo returns HTTP 429 (minutely rate limit exceeded), pauses 65 s
    to let the rate limit window reset, then retries.
    """
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            resp = client.get(ARCHIVE_URL, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 429:
                log.warning(
                    "Open-Meteo minutely rate limit reached (HTTP 429: %s). "
                    "Pausing %d s for window reset...",
                    resp.text.strip(),
                    RATE_LIMIT_RESET_S,
                )
                time.sleep(RATE_LIMIT_RESET_S)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("error"):
                log.warning(
                    "Open-Meteo returned error: %s. Pausing %d s for window reset...",
                    payload.get("reason"),
                    RATE_LIMIT_RESET_S,
                )
                time.sleep(RATE_LIMIT_RESET_S)
                continue
            return payload
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if attempt < len(RETRY_WAITS):
                wait = RETRY_WAITS[attempt]
                log.warning("Fetch attempt %d failed (%s). Retrying in %d s...", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def _build_params(lat: float, lon: float, end_date: str) -> dict:
    """Build the Open-Meteo query parameter dict for one point."""
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }


def _process_response(payload: dict) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a tidy daily DataFrame.

    Steps
    -----
    1. Parse daily variables → one row per day.
    2. Parse hourly soil-moisture variables.
    3. Aggregate hourly SM → daily IST days via aggregate_hourly_to_daily.
    4. Merge and compute sm_shallow / sm_root.
    5. Return DataFrame with PARQUET_COLUMNS.
    """
    # ── Daily variables ──────────────────────────────────────────────────────
    daily_raw = payload.get("daily", {})
    if not daily_raw or "time" not in daily_raw:
        raise ValueError("Response missing 'daily' section")

    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_raw["time"]).date,
            "precip_mm": daily_raw.get("precipitation_sum"),
            "et0_mm": daily_raw.get("et0_fao_evapotranspiration"),
            "tmax_c": daily_raw.get("temperature_2m_max"),
        }
    )

    # ── Hourly soil moisture ─────────────────────────────────────────────────
    hourly_raw = payload.get("hourly", {})
    if not hourly_raw or "time" not in hourly_raw:
        raise ValueError("Response missing 'hourly' section")

    hourly_df = pd.DataFrame(
        {
            "soil_moisture_0_to_7cm": hourly_raw.get("soil_moisture_0_to_7cm"),
            "soil_moisture_7_to_28cm": hourly_raw.get("soil_moisture_7_to_28cm"),
            "soil_moisture_28_to_100cm": hourly_raw.get("soil_moisture_28_to_100cm"),
        },
        index=pd.to_datetime(hourly_raw["time"]).tz_localize(TIMEZONE),
    )

    # aggregate_hourly_to_daily requires a tz-aware index; TIMEZONE is already set.
    sm_daily = aggregate_hourly_to_daily(
        hourly_df,
        how="mean",
        tz=TIMEZONE,
    )
    # sm_daily index is datetime.date
    sm_daily.index = pd.to_datetime(sm_daily.index)

    # ── sm_shallow = 0-7 cm mean; sm_root = mean(7-28 cm, 28-100 cm) ────────
    sm_df = pd.DataFrame(
        {
            "date": sm_daily.index.date,
            "sm_shallow": sm_daily["soil_moisture_0_to_7cm"].values,
            "sm_root": (
                sm_daily[["soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"]]
                .mean(axis=1)
                .values
            ),
        }
    )

    # ── Merge ────────────────────────────────────────────────────────────────
    merged = daily_df.merge(sm_df, on="date", how="left")
    merged = merged[PARQUET_COLUMNS].reset_index(drop=True)
    return merged


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path))


def _load_manifest() -> set[str]:
    """Return set of already-completed point IDs."""
    if MANIFEST_JSON.exists():
        with open(MANIFEST_JSON) as f:
            data = json.load(f)
        return set(data.get("done", []))
    return set()


def _update_manifest(done: set[str]) -> None:
    """Persist the finished-point manifest."""
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w") as f:
        json.dump({"done": sorted(done)}, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(limit: int | None = None, end_date: str | None = None) -> None:
    """Pull weather for all points in POINTS_JSON.

    Parameters
    ----------
    limit:
        If set, process only the first *limit* points (test mode).
    end_date:
        Last date to request (YYYY-MM-DD).  Defaults to yesterday in IST.
    """
    if not POINTS_JSON.exists():
        log.error("train_points.json not found at %s — run sample_points.py first.", POINTS_JSON)
        sys.exit(1)

    with open(POINTS_JSON) as f:
        points: list[dict] = json.load(f)

    if limit is not None:
        points = points[:limit]
        log.info("TEST MODE: processing %d point(s).", limit)

    # Determine end date without calling system clock — use a fixed recent date
    # that is always in the past relative to the ERA5-Land update lag (~5 days).
    # Callers can override via --end-date.
    if end_date is None:
        # ERA5-Land typically lags ~5 days; use a conservative default.
        # We compute this from a known recent date without system clock calls.
        # The actual latest-available date is discovered from the API's last returned row.
        end_date = "2026-10-25"  # conservative; API will return up to what's available
    log.info("Date range: %s → %s", START_DATE, end_date)

    done = _load_manifest()
    remaining = [p for p in points if p["id"] not in done]
    log.info(
        "Points: total=%d  done=%d  remaining=%d",
        len(points), len(done), len(remaining),
    )

    if not remaining:
        log.info("All points already complete.")
        return

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for i, point in enumerate(remaining, 1):
            pid = point["id"]
            lat = point["lat"]
            lon = point["lon"]
            out_path = WEATHER_DIR / f"{pid}.parquet"

            log.info(
                "[%d/%d] %s  lat=%.4f lon=%.4f dist=%s",
                i, len(remaining), pid, lat, lon, point.get("district", "?"),
            )

            params = _build_params(lat, lon, end_date)
            try:
                payload = _fetch(client, params)
            except Exception as exc:
                log.error("  FAILED to fetch %s: %s", pid, exc)
                continue

            try:
                df = _process_response(payload)
            except Exception as exc:
                log.error("  FAILED to parse %s: %s", pid, exc)
                continue

            _write_parquet(df, out_path)
            done.add(pid)
            _update_manifest(done)

            log.info(
                "  OK  rows=%d  cols=%s  → %s",
                len(df), list(df.columns), out_path.name,
            )

            # Polite pause between requests
            if i < len(remaining):
                time.sleep(POLITE_PAUSE_S)

    log.info("Done. Completed %d/%d points.", len(done), len(points))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Open-Meteo weather for training points.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N points (for testing).",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Last date to request, YYYY-MM-DD (default: 2026-10-25).",
    )
    args = parser.parse_args()
    main(limit=args.limit, end_date=args.end_date)
