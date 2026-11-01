"""Offline tests for geo/weather.py using synthetic Open-Meteo-shaped payloads (tests/ only)."""
import datetime as dt

import pytest

import geo.modis as M
import geo.weather as W
from contracts import FieldInput

D = dt.timedelta
POLY = {"type": "Polygon", "coordinates": [[[79.60, 18.00], [79.61, 18.00], [79.61, 18.01], [79.60, 18.01], [79.60, 18.00]]]}


@pytest.fixture(autouse=True)
def _no_modis_network(monkeypatch):
    monkeypatch.setattr(M, "get_modis_obs", lambda field, factory=None: ([], ["modis_stubbed_in_test"]))


def payload(s, e):
    days = [s + D(days=i) for i in range((e - s).days + 1)]
    return {"daily": {"time": [d.isoformat() for d in days],
                      "precipitation_sum": [6.0 if d.toordinal() % 5 == 0 else 0.0 for d in days],
                      "et0_fao_evapotranspiration": [4.0] * len(days),
                      "temperature_2m_max": [32.0] * len(days)},
            "hourly": {"time": [f"{d}T{h:02d}:00" for d in days for h in range(24)],
                       "soil_moisture_0_to_7cm": [0.2 + 0.001 * (d.toordinal() % 30) for d in days for _ in range(24)],
                       "soil_moisture_7_to_28cm": [0.25] * (24 * len(days)),
                       "soil_moisture_28_to_100cm": [0.30] * (24 * len(days))}}


def make_fake(ref):
    def fake(url, params, ttl_s=0, **kwargs):
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        if "ensemble" in url:
            days = [ref + D(days=i) for i in range(15)]
            h = {"time": [f"{d}T{x:02d}:00" for d in days for x in range(24)]}
            for k in ("precipitation", "precipitation_member01"):
                h[k] = [0.1] * (24 * 15)
            return {"hourly": h}, "live", now, []
        if "start_date" in params:
            s, e = dt.date.fromisoformat(params["start_date"]), dt.date.fromisoformat(params["end_date"])
        else:
            s, e = ref - D(days=params["past_days"]), ref + D(days=params["forecast_days"] - 1)
        return payload(s, e), "live", now, []
    return fake


def test_replay_no_forecast_and_valid_bundle(monkeypatch):
    as_of = dt.date(2025, 8, 20)
    monkeypatch.setattr(W, "get_json", make_fake(as_of))
    from geo import build_observations
    f = FieldInput(polygon=POLY, as_of=as_of, mode="replay")
    b = build_observations(f)
    assert len(b.weather_daily) >= 60 and not any(w.is_forecast for w in b.weather_daily)
    assert max(w.day for w in b.weather_daily) == as_of
    assert b.rain_ensemble == [] and "no_ensemble_in_replay" in b.events
    assert b.climatology.n_years == as_of.year - W.CLIM_FIRST_YEAR
    assert b.climatology.sm_root_percentile is not None and b.climatology.rain_deficit_14d_mm is not None
    ids = {e.id for e in b.evidence}
    assert all(w.evidence_id in ids for w in b.weather_daily)
    assert all(i in ids for i in b.climatology.evidence_ids)


def test_live_has_16_forecast_rows_and_ensemble(monkeypatch):
    as_of = dt.date(2026, 11, 1)
    monkeypatch.setattr(W, "get_json", make_fake(as_of))
    from geo import build_observations
    b = build_observations(FieldInput(polygon=POLY, as_of=as_of, mode="live"))
    assert sum(w.is_forecast for w in b.weather_daily) == 16
    assert len(b.rain_ensemble) == 2 and len(b.rain_ensemble[0]) == 14
    # climatology reference day is the last OBSERVED day (as_of - 1 in live)
    assert max(w.day for w in b.weather_daily if not w.is_forecast) == as_of - D(days=1)


def test_live_forecast_segment_uses_bounded_recent_cache(monkeypatch):
    as_of = dt.date(2026, 11, 1)
    calls = []

    def fake_segment(url, lat, lon, extra, ttl_s, stale_if_error_ttl_s=None):
        calls.append((url, ttl_s, stale_if_error_ttl_s))
        return {}, "cached", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), []

    monkeypatch.setattr(W, "_fetch_segment", fake_segment)
    W._collect_days(17.98, 79.58, as_of, "live", 60, [])

    forecast_call = next(call for call in calls if call[0] == W.FORECAST_URL)
    assert forecast_call == (
        W.FORECAST_URL,
        W.LIVE_FORECAST_CACHE_TTL_S,
        W.LIVE_FORECAST_CACHE_TTL_S,
    )


def test_climatology_comes_from_common_features(monkeypatch):
    """geo must not compute its own statistics: patch the shared function and see its numbers pass through."""
    as_of = dt.date(2025, 8, 20)
    monkeypatch.setattr(W, "get_json", make_fake(as_of))
    seen = {}

    def fake_clim(df, ref_day):
        seen["ref"] = ref_day
        return {"sm_root_percentile": 12.34, "sm_shallow_percentile": 56.78, "rain_deficit_14d_mm": 9.99,
                "days_since_rain_gt_2p5mm": 7, "n_years": 15, "baseline_years": list(range(2010, 2025)),
                "evidence_ids": []}

    monkeypatch.setattr(W, "climatology_features", fake_clim)
    from geo import build_observations
    b = build_observations(FieldInput(polygon=POLY, as_of=as_of, mode="replay"))
    assert seen["ref"] == as_of
    assert b.climatology.sm_root_percentile == 12.3
    assert b.climatology.rain_deficit_14d_mm == 10.0
    assert b.climatology.days_since_rain_gt_2p5mm == 7


def test_outside_india_rejected():
    with pytest.raises(ValueError):
        W.check_india(51.5, -0.1)