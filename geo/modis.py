"""geo/modis.py - MODIS 13Q1 NDVI observations for the ObsBundle (P1).

* Sampling is IDENTICAL to training: it reuses ml.train.pull_modis_cube (Terra only, 3x3 px window at the field
  centroid, pixel_reliability == 0, median * 0.0001), so serving z-scores match the training distribution.
* SensorObs.day = composite END date (start + 15). Only composites with end_date <= as_of are used.
* z is computed with common.baseline.compute_z (sensor 'modis_ndvi': start year 2010, +-10 d pooled window,
  strictly earlier years, min 5 years / 6 obs, sigma floor). Nothing is recomputed here.
* The last N_RECENT usable composites are returned (P2 needs 3 for ndvi_z_slope_3comp).
* A composite with no good pixels is skipped with an event - never a fake value.
* Per-item extractions are cached under data/cache/modis/ (composites are immutable once published).
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common.baseline import circ_doy_dist, compute_z
from config import BASELINE_HALF_WINDOW_DAYS, BASELINE_START_YEAR
from contracts import SensorObs
from geo.http import CACHE_DIR
from geo.weather import EvidenceFactory, centroid_of

SOURCE = "planetary-computer:modis-13Q1-061"
N_RECENT = 3
STALE_EVENT_DAYS = 20
_WINDOW_PIXELS = 9          # 3x3


def _search_items(lon: float, lat: float, as_of: dt.date) -> list[dict]:
    """STAC items (Terra, one per composite) 2010..as_of.year. Patched in tests."""
    from ml.train.pull_modis_cube import _deduplicate_items, _stac_search_all
    start_year = BASELINE_START_YEAR["modis_ndvi"]
    return _deduplicate_items(_stac_search_all(lon, lat, start_year, as_of.year, platform="terra"))


def _cache_path(lat: float, lon: float, item_id: str):
    return CACHE_DIR / "modis" / f"{lat:.4f}_{lon:.4f}" / f"{item_id}.json"


def _extract(item: dict, lon: float, lat: float):
    """-> (record | None, from_cache). Patched in tests."""
    path = _cache_path(lat, lon, item["id"])
    if path.exists():
        try:
            return json.loads(path.read_text()), True
        except Exception:
            pass
    from ml.train.pull_modis_cube import process_item
    rec = process_item(item, lon, lat)
    if rec is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec))
        except Exception:
            pass                                  # cache failure must never break a live answer
    return rec, False


def _dates(item: dict):
    p = item["properties"]
    return dt.date.fromisoformat(p["start_datetime"][:10]), dt.date.fromisoformat(p["end_datetime"][:10])


def get_modis_obs(field, factory: EvidenceFactory | None = None):
    """field: contracts.FieldInput -> (list[SensorObs], events). Evidence rows go into *factory*."""
    factory = factory or EvidenceFactory()
    as_of = field.as_of
    events: list[str] = []
    lat, lon = centroid_of(field.polygon)

    dated: list[tuple[dt.date, dt.date, dict]] = []
    for it in _search_items(lon, lat, as_of):
        try:
            sd, ed = _dates(it)
        except (KeyError, ValueError):
            continue
        if ed <= as_of:                           # D5: no look-ahead
            dated.append((sd, ed, it))
    dated.sort(key=lambda x: x[1])
    if not dated:
        events.append("modis_unavailable:no_composites")
        return [], events

    recent = dated[-N_RECENT:]
    recent_ids = {it["id"] for _, _, it in recent}
    targets = [ed for _, ed, _ in recent]
    need = [(sd, ed, it) for sd, ed, it in dated
            if it["id"] in recent_ids
            or (ed.year < as_of.year and any(circ_doy_dist(ed, t) <= BASELINE_HALF_WINDOW_DAYS for t in targets))]

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda x: _extract(x[2], lon, lat), need))

    recs: dict[str, tuple[dt.date, dict]] = {}
    failed, any_live = 0, False
    for (sd, ed, it), (rec, cached) in zip(need, results):
        if rec is None:
            failed += 1
            continue
        any_live = any_live or not cached
        recs[it["id"]] = (ed, rec)
    if failed:
        events.append(f"modis_read_failed:{failed}")

    series = pd.Series({pd.Timestamp(ed): float(rec.get("ndvi", float("nan"))) for ed, rec in recs.values()},
                       dtype=float).sort_index()
    quality = "live" if any_live else "cached"
    fetched_at = dt.datetime.fromtimestamp(time.time(), tz=dt.timezone.utc)

    obs: list[SensorObs] = []
    for sd, ed, it in recent:
        got = recs.get(it["id"])
        if got is None:
            continue
        rec = got[1]
        ndvi = float(rec.get("ndvi", float("nan")))
        if not rec.get("quality_ok") or ndvi != ndvi:
            events.append(f"modis_composite_unusable:{ed}")
            continue
        z, n_years, n_obs = compute_z(ndvi, series, ed, "modis_ndvi")
        ev = factory.make("modis_ndvi", round(ndvi, 4), SOURCE, fetched_at, quality, "ndvi", ed,
                          f"composite {sd}..{ed}; 3x3 px @250 m; reliability==0")
        if z is not None:
            factory.make("modis_ndvi_z", round(z, 2), SOURCE, fetched_at, quality, "z", ed,
                         f"compute_z: n_years={n_years}, n_obs={n_obs}")
        else:
            events.append(f"modis_z_unavailable:{ed}:n_years={n_years}")
        obs.append(SensorObs(sensor="modis_ndvi", day=ed, value=ndvi, z=z,
                             valid_frac=min(1.0, int(rec.get("n_good_pixels", 0)) / _WINDOW_PIXELS),
                             n_baseline=n_years, evidence_id=ev.id))

    if obs and (as_of - obs[-1].day).days > STALE_EVENT_DAYS:
        events.append(f"modis_stale:{(as_of - obs[-1].day).days}_days_old")
    return obs, events