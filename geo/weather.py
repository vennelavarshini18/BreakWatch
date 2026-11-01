"""geo/weather.py - Open-Meteo weather, ensemble and climatology (P1).

Replay : archive endpoint only, days <= as_of, no forecast rows, no ensemble.
Live   : archive (settled past) + forecast endpoint (recent gap + 16 forecast days) + ensemble (14 days).
          In live mode the as_of day itself is marked is_forecast=True (that day is not complete yet),
          which gives 16 forecast rows (as_of .. as_of+15).

Climatology is NOT computed here. The archive history (2010-01-01 .. Dec 31 of as_of.year-1) is fetched once
(immutable cache) and handed to common.features.climatology_features, which uses common.baseline.
Hourly soil moisture is reduced to daily with common.features.aggregate_hourly_to_daily (>= 18 h/day, IST).

Never reads the system date. The CLI value "--as-of live" goes through api.clock.

Evidence: one row per (source, is_forecast) group of daily rows (WeatherDay.evidence_id points to it), plus one
row per derived climatology number. ids are "geo_001"... (P3's Ledger re-stamps/merges them).
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Optional

import pandas as pd
from shapely.geometry import shape

from common.features import aggregate_hourly_to_daily, climatology_features
from config import MIN_BASELINE_YEARS
from contracts import Climatology, Evidence, WeatherDay
from geo.http import IMMUTABLE_TTL_S, ApiError, NetworkUnavailable, get_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ENSEMBLE_MODEL = "ecmwf_ifs025"
TZ = "Asia/Kolkata"

DAILY_VARS = "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max"
HOURLY_SM = "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm"
INDIA_BBOX = (68.0, 6.0, 98.0, 38.0)          # lon_min, lat_min, lon_max, lat_max
ARCHIVE_LAG_DAYS = 11                          # live: archive up to as_of-11, forecast endpoint (past_days=10) after
FORECAST_PAST_DAYS = 10
FORECAST_DAYS = 16
ENSEMBLE_DAYS = 14
CLIM_FIRST_YEAR = 2010
REPLAY_TTL_S = 24 * 3600
LIVE_FORECAST_CACHE_TTL_S = 15 * 60
_D = dt.timedelta


@dataclass
class WeatherBundle:
    days: list[WeatherDay]
    evidence: list[Evidence]
    events: list[str]
    rain_ensemble: list[list[float]] = dc_field(default_factory=list)
    climatology: Climatology = dc_field(default_factory=Climatology)
    lat: float = 0.0
    lon: float = 0.0


class EvidenceFactory:
    def __init__(self, prefix: str = "geo"):
        self.prefix, self.n, self.rows = prefix, 0, []

    def make(self, key, value, source, fetched_at, quality, unit=None, obs_date=None, note=None) -> Evidence:
        self.n += 1
        ev = Evidence(id=f"{self.prefix}_{self.n:03d}", key=key, value=value, unit=unit, source=source,
                      fetched_at=fetched_at, obs_date=obs_date, quality=quality, note=note)
        self.rows.append(ev)
        return ev


# ---------------------------------------------------------------- geometry
def check_india(lat: float, lon: float) -> None:
    lo, la, hi, ha = INDIA_BBOX
    if not (lo <= lon <= hi and la <= lat <= ha):
        raise ValueError(f"location ({lat:.4f}, {lon:.4f}) is outside India")


def centroid_of(polygon: dict) -> tuple[float, float]:
    try:
        geom = shape(polygon)
    except Exception as e:
        raise ValueError(f"invalid polygon: {e}")
    if geom.is_empty or geom.geom_type != "Polygon":
        raise ValueError("expected a non-empty GeoJSON Polygon")
    c = geom.centroid
    check_india(c.y, c.x)
    return c.y, c.x


# ---------------------------------------------------------------- parsing
def _f(x) -> Optional[float]:
    return None if x is None else float(x)


def _hourly_sm_to_daily(h: dict) -> dict[dt.date, dict]:
    """Hourly soil moisture -> {day: {sm0, sm7, sm28}} using the shared 18-hour IST rule."""
    times = h.get("time") or []
    cols: dict[str, list] = {}
    for src, dst in (("soil_moisture_0_to_7cm", "sm0"), ("soil_moisture_7_to_28cm", "sm7"),
                     ("soil_moisture_28_to_100cm", "sm28")):
        vals = h.get(src)
        if vals and len(vals) == len(times):
            cols[dst] = [None if v is None else float(v) for v in vals]
    if not times or not cols:
        return {}
    df = pd.DataFrame(cols, index=pd.to_datetime(times).tz_localize(TZ), dtype=float)
    daily = aggregate_hourly_to_daily(df, how="mean", tz=TZ)
    out: dict[dt.date, dict] = {}
    for day, row in daily.iterrows():
        for col, v in row.items():
            if pd.notna(v):
                out.setdefault(day, {})[col] = float(v)
    return out


def parse_openmeteo(js: dict) -> dict[dt.date, dict]:
    """-> {day: {precip, et0, tmax, sm0, sm7, sm28}}; soil moisture via common.features (>= 18 valid hours/day)."""
    out: dict[dt.date, dict] = {}
    d = js.get("daily") or {}
    for i, t in enumerate(d.get("time", [])):
        rec = out.setdefault(dt.date.fromisoformat(t), {})
        for src, dst in (("precipitation_sum", "precip"), ("et0_fao_evapotranspiration", "et0"),
                         ("temperature_2m_max", "tmax")):
            vals = d.get(src)
            rec[dst] = _f(vals[i]) if vals and i < len(vals) else None
    for day, vals in _hourly_sm_to_daily(js.get("hourly") or {}).items():
        out.setdefault(day, {}).update(vals)
    return out


def _sm_root(rec: dict) -> Optional[float]:
    a, b = rec.get("sm7"), rec.get("sm28")
    return None if a is None or b is None else (a + b) / 2.0


def _has_data(rec: dict) -> bool:
    return any(rec.get(k) is not None for k in ("precip", "et0", "tmax", "sm0", "sm7", "sm28"))


def parse_ensemble(js: dict, as_of: dt.date, n_days: int) -> list[list[float]]:
    """Hourly precipitation per member -> [member][day] daily totals for as_of .. as_of+n_days-1.
    Members with missing hours are dropped (never filled)."""
    h = js.get("hourly") or {}
    times = h.get("time", [])
    wanted = [(as_of + _D(days=i)).isoformat() for i in range(n_days)]
    members: list[list[float]] = []
    for key, vals in h.items():
        if key == "time" or not key.startswith("precipitation"):
            continue
        per_day: dict[str, list] = {}
        for t, v in zip(times, vals):
            per_day.setdefault(t[:10], []).append(v)
        series, ok = [], True
        for d in wanted:
            lst = per_day.get(d)
            if not lst or len(lst) < 24 or any(v is None for v in lst):
                ok = False
                break
            series.append(round(sum(lst), 3))
        if ok:
            members.append(series)
    return members


# ---------------------------------------------------------------- fetching
def _fetch_segment(
    url: str,
    lat: float,
    lon: float,
    extra: dict,
    ttl_s: int,
    stale_if_error_ttl_s: int | None = None,
):
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "daily": DAILY_VARS,
              "hourly": HOURLY_SM, "timezone": TZ, **extra}
    if stale_if_error_ttl_s is None:
        js, quality, fetched_at, events = get_json(url, params, ttl_s)
    else:
        js, quality, fetched_at, events = get_json(
            url,
            params,
            ttl_s,
            stale_if_error_ttl_s=stale_if_error_ttl_s,
        )
    return parse_openmeteo(js), quality, fetched_at, events


def _collect_days(lat, lon, as_of, mode, lookback_days, events):
    start = as_of - _D(days=lookback_days - 1)
    merged: dict[dt.date, tuple] = {}              # day -> (rec, source, quality, fetched_at)

    arch_end = as_of if mode == "replay" else as_of - _D(days=ARCHIVE_LAG_DAYS)
    immutable = arch_end <= as_of - _D(days=10)
    ttl = IMMUTABLE_TTL_S if immutable else (REPLAY_TTL_S if mode == "replay" else 0)
    try:
        recs, q, fa, ev = _fetch_segment(ARCHIVE_URL, lat, lon,
                                         {"start_date": start.isoformat(), "end_date": arch_end.isoformat()}, ttl)
        events += ev
        for day, rec in recs.items():
            if start <= day <= arch_end and _has_data(rec):
                merged[day] = (rec, "open-meteo:archive", q, fa)
    except (NetworkUnavailable, ApiError) as e:
        if mode == "replay":
            raise
        events.append(f"archive_failed:{type(e).__name__}")
        arch_end = start - _D(days=1)               # forecast endpoint must cover the whole lookback

    if mode == "live":
        past = min(92, max(FORECAST_PAST_DAYS, (as_of - arch_end).days))
        recs, q, fa, ev = _fetch_segment(FORECAST_URL, lat, lon,
                                         {"past_days": past, "forecast_days": FORECAST_DAYS},
                                         LIVE_FORECAST_CACHE_TTL_S,
                                         stale_if_error_ttl_s=LIVE_FORECAST_CACHE_TTL_S)
        events += ev
        last = as_of + _D(days=FORECAST_DAYS - 1)
        for day, rec in recs.items():
            if arch_end < day <= last and day >= start and _has_data(rec):
                merged[day] = (rec, "open-meteo:forecast", q, fa)

    last_obs = as_of - _D(days=1) if mode == "live" else as_of
    missing = sum(1 for i in range((last_obs - start).days + 1) if (start + _D(days=i)) not in merged)
    if missing:
        events.append(f"weather_gap:{missing}_days_missing")
    return merged


def get_rain_ensemble(lat, lon, as_of, events, factory):
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "hourly": "precipitation",
              "models": ENSEMBLE_MODEL, "forecast_days": ENSEMBLE_DAYS + 1, "timezone": TZ}
    try:
        js, quality, fa, ev = get_json(ENSEMBLE_URL, params, 0)
    except (NetworkUnavailable, ApiError) as e:
        events.append(f"ensemble_failed:{type(e).__name__}")
        return []
    events += ev
    members = parse_ensemble(js, as_of, ENSEMBLE_DAYS)
    if not members:
        events.append("ensemble_failed:no_complete_members")
        return []
    totals = sorted(sum(m) for m in members)
    src = f"open-meteo:ensemble:{ENSEMBLE_MODEL}"
    factory.make("ensemble_members", len(members), src, fa, quality, note=f"{ENSEMBLE_DAYS}-day horizon")
    factory.make("ensemble_rain_total_mean_mm", round(sum(totals) / len(totals), 1), src, fa, quality, "mm",
                 obs_date=as_of)
    factory.make("ensemble_rain_total_p10_mm", round(totals[int(0.1 * (len(totals) - 1))], 1), src, fa, quality, "mm",
                 obs_date=as_of)
    factory.make("ensemble_rain_total_p90_mm", round(totals[int(0.9 * (len(totals) - 1))], 1), src, fa, quality, "mm",
                 obs_date=as_of)
    return members


# ---------------------------------------------------------------- climatology (delegated)
def _history_days(lat, lon, as_of, events):
    """Archive history 2010-01-01 .. Dec 31 of as_of.year-1 in ONE call (stable cache key, immutable).
    Baselines only ever use years strictly before as_of.year, so nothing later is needed."""
    start = dt.date(CLIM_FIRST_YEAR, 1, 1)
    end = dt.date(as_of.year - 1, 12, 31)
    if end < start:
        return {}, "live", None
    recs, quality, fetched_at, ev = _fetch_segment(
        ARCHIVE_URL, lat, lon, {"start_date": start.isoformat(), "end_date": end.isoformat()}, IMMUTABLE_TTL_S)
    events += ev
    return recs, quality, fetched_at


def compute_climatology(lat, lon, as_of, days: list[WeatherDay], factory: EvidenceFactory, events: list[str],
                        first_year: int = CLIM_FIRST_YEAR) -> Climatology:
    """Fill Climatology by calling common.features.climatology_features (which uses common.baseline).
    No statistics are computed in this module. Reference day = last OBSERVED day (as_of in replay, as_of-1 live)."""
    observed = sorted([d for d in days if not d.is_forecast and d.day <= as_of and d.precip_mm is not None],
                      key=lambda d: d.day)
    if not observed:
        events.append("climatology_unavailable:no_observed_days")
        return Climatology()
    ref_day = observed[-1].day

    try:
        hist, quality, fetched_at = _history_days(lat, lon, as_of, events)
    except (NetworkUnavailable, ApiError) as e:
        events.append(f"climatology_unavailable:{type(e).__name__}")
        return Climatology()

    rows: dict[dt.date, dict] = {}
    for day, r in hist.items():
        rows[day] = {"day": day, "precip_mm": r.get("precip"), "et0_mm": r.get("et0"), "tmax_c": r.get("tmax"),
                     "sm_shallow": r.get("sm0"), "sm_root": _sm_root(r)}
    for d in observed:                                   # recent window overrides (same days, same source)
        rows[d.day] = {"day": d.day, "precip_mm": d.precip_mm, "et0_mm": d.et0_mm, "tmax_c": d.tmax_c,
                       "sm_shallow": d.sm_shallow, "sm_root": d.sm_root}
    df = pd.DataFrame(list(rows.values()))
    for col in ("precip_mm", "et0_mm", "tmax_c", "sm_shallow", "sm_root"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    res = climatology_features(df, ref_day)
    n_years = int(res["n_years"])
    if n_years < MIN_BASELINE_YEARS["weather"] or fetched_at is None:
        events.append("climatology_insufficient_years")
        return Climatology(n_years=n_years)

    src = f"open-meteo:archive:era5-climatology:{first_year}-{as_of.year - 1}"
    ids: list[str] = []

    def add(key, value, unit, note=None):
        ids.append(factory.make(key, value, src, fetched_at, quality, unit, ref_day, note).id)

    clim = Climatology(n_years=n_years, baseline_years=res["baseline_years"])
    if res["sm_root_percentile"] is not None:
        clim.sm_root_percentile = round(float(res["sm_root_percentile"]), 1)
        add("sm_root_percentile", clim.sm_root_percentile, "percentile", f"n_years={n_years}, common.baseline")
    if res["sm_shallow_percentile"] is not None:
        clim.sm_shallow_percentile = round(float(res["sm_shallow_percentile"]), 1)
        add("sm_shallow_percentile", clim.sm_shallow_percentile, "percentile", f"n_years={n_years}, common.baseline")
    if res["rain_deficit_14d_mm"] is not None:
        clim.rain_deficit_14d_mm = round(float(res["rain_deficit_14d_mm"]), 1)
        add("rain_deficit_14d_mm", clim.rain_deficit_14d_mm, "mm", "normal - observed; positive = deficit")
    if res["days_since_rain_gt_2p5mm"] is not None:
        clim.days_since_rain_gt_2p5mm = int(res["days_since_rain_gt_2p5mm"])
        add("days_since_rain_gt_2p5mm", clim.days_since_rain_gt_2p5mm, "days", "capped at 60-day lookback")
    clim.evidence_ids = ids
    return clim


# ---------------------------------------------------------------- public API
def get_weather(lat: float, lon: float, as_of: dt.date, mode: str, lookback_days: int = 60,
                include_climatology: bool = True, factory: Optional[EvidenceFactory] = None) -> WeatherBundle:
    if mode not in ("live", "replay"):
        raise ValueError("mode must be 'live' or 'replay'")
    check_india(lat, lon)
    factory = factory or EvidenceFactory()
    events: list[str] = []
    merged = _collect_days(lat, lon, as_of, mode, lookback_days, events)

    groups: dict[tuple, list] = {}
    for day, (rec, source, q, fa) in merged.items():
        is_fc = mode == "live" and day >= as_of
        groups.setdefault((source, is_fc, q, fa), []).append(day)
    ev_by_group: dict[tuple, str] = {}
    for (source, is_fc, q, fa), ds in groups.items():
        ds.sort()
        note = ("forecast days (model forecast)" if is_fc else
                "observed/reanalysis days: precip, ET0, Tmax, soil moisture (hourly -> daily mean, >=18 h)")
        ev = factory.make("weather_daily_series", f"{len(ds)} days {ds[0]}..{ds[-1]}", source, fa, q,
                          obs_date=ds[-1], note=note)
        ev_by_group[(source, is_fc, q, fa)] = ev.id

    days: list[WeatherDay] = []
    for day in sorted(merged):
        rec, source, q, fa = merged[day]
        is_fc = mode == "live" and day >= as_of
        days.append(WeatherDay(day=day, precip_mm=rec.get("precip"), et0_mm=rec.get("et0"), tmax_c=rec.get("tmax"),
                               sm_shallow=rec.get("sm0"), sm_root=_sm_root(rec), is_forecast=is_fc,
                               evidence_id=ev_by_group[(source, is_fc, q, fa)]))

    if mode == "replay":
        events.append("no_ensemble_in_replay")
        ensemble: list[list[float]] = []
    else:
        ensemble = get_rain_ensemble(lat, lon, as_of, events, factory)

    clim = compute_climatology(lat, lon, as_of, days, factory, events) if include_climatology else Climatology()
    return WeatherBundle(days=days, evidence=list(factory.rows), events=events, rain_ensemble=ensemble,
                         climatology=clim, lat=lat, lon=lon)


# ---------------------------------------------------------------- CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description="BreakWatch P1 weather check")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--as-of", required=True, help="YYYY-MM-DD or 'live' (current IST date via api.clock)")
    ap.add_argument("--mode", choices=["live", "replay"], required=True)
    ap.add_argument("--lookback", type=int, default=60)
    a = ap.parse_args()
    if a.as_of == "live":
        from api.clock import today_utc
        as_of = today_utc()
    else:
        as_of = dt.date.fromisoformat(a.as_of)
    wb = get_weather(a.lat, a.lon, as_of, a.mode, a.lookback)
    fc = [d for d in wb.days if d.is_forecast]
    print(f"as_of={as_of} mode={a.mode} lat={a.lat} lon={a.lon}")
    print(f"daily rows: {len(wb.days)}  (forecast rows: {len(fc)})")
    print(f"{'day':<12}{'precip':>8}{'et0':>7}{'tmax':>7}{'sm_sh':>8}{'sm_root':>9}  fc")
    shown = wb.days[:3] + wb.days[-3:] if len(wb.days) > 6 else wb.days
    fmt = lambda v, n: f"{v:{n}.2f}" if v is not None else " " * (n - 4) + "None"
    for d in shown:
        print(f"{d.day!s:<12}{fmt(d.precip_mm, 8)}{fmt(d.et0_mm, 7)}{fmt(d.tmax_c, 7)}"
              f"{fmt(d.sm_shallow, 8)}{fmt(d.sm_root, 9)}  {d.is_forecast}")
    print("climatology:", wb.climatology.model_dump())
    print(f"ensemble members: {len(wb.rain_ensemble)}")
    print("evidence rows:")
    for e in wb.evidence:
        print(f"  {e.id} {e.key}={e.value} {e.unit or ''} src={e.source} q={e.quality} obs={e.obs_date}")
    print("events:", wb.events)


if __name__ == "__main__":
    _main()