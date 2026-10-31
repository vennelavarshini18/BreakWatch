"""Offline tests for geo/weather.py using synthetic Open-Meteo-shaped payloads (tests/ only)."""
import datetime as dt

import pytest

import geo.weather as W
from contracts import FieldInput

D = dt.timedelta
POLY = {"type": "Polygon", "coordinates": [[[79.60, 18.00], [79.61, 18.00], [79.61, 18.01], [79.60, 18.01], [79.60, 18.00]]]}


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


def make_fake(today):
    def fake(url, params, ttl_s=0):
        now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        if "ensemble" in url:
            days = [today + D(days=i) for i in range(15)]
            h = {"time": [f"{d}T{x:02d}:00" for d in days for x in range(24)]}
            for k in ("precipitation", "precipitation_member01"):
                h[k] = [0.1] * (24 * 15)
            return {"hourly": h}, "live", now, []
        if "start_date" in params:
            s, e = dt.date.fromisoformat(params["start_date"]), dt.date.fromisoformat(params["end_date"])
        else:
            s, e = today - D(days=params["past_days"]), today + D(days=params["forecast_days"] - 1)
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
    today = dt.date(2026, 11, 1)
    monkeypatch.setattr(W, "get_json", make_fake(today))
    from geo import build_observations
    b = build_observations(FieldInput(polygon=POLY, as_of=today, mode="live"))
    assert sum(w.is_forecast for w in b.weather_daily) == 16
    assert len(b.rain_ensemble) == 2 and len(b.rain_ensemble[0]) == 14


def test_outside_india_rejected():
    with pytest.raises(ValueError):
        W.check_india(51.5, -0.1)


def test_percentile():
    assert W._pct(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 45.0