"""
tests/test_triggers.py
======================
Unit tests for ICAR dry spell triggers in ml/triggers.py.
Synthetic arrays are used and explicitly marked as test-only.
"""

from __future__ import annotations

from datetime import date, timedelta
import pytest

from contracts import FieldInput, ObsBundle, WeatherDay
from ml.triggers import check_triggers


def _make_weather_days(
    end_date: date,
    precip_values: list[float | None],
    forecast_flags: list[bool] | None = None,
) -> list[WeatherDay]:
    """Test-only helper: create synthetic WeatherDay sequence ending on end_date."""
    n = len(precip_values)
    start_date = end_date - timedelta(days=n - 1)
    if forecast_flags is None:
        forecast_flags = [False] * n

    days = []
    for i, (p, is_fc) in enumerate(zip(precip_values, forecast_flags)):
        d = start_date + timedelta(days=i)
        days.append(
            WeatherDay(
                day=d,
                precip_mm=p,
                et0_mm=4.0,
                tmax_c=32.0,
                sm_shallow=0.25,
                sm_root=0.30,
                is_forecast=is_fc,
                evidence_id=f"ev_test_{i}",
            )
        )
    return days


def _make_bundle(
    as_of: date,
    weather_days: list[WeatherDay],
    district: str = "Adilabad",
    sowing_date: date | None = None,
) -> ObsBundle:
    """Test-only helper: build an ObsBundle."""
    field = FieldInput(
        polygon={"type": "Polygon", "coordinates": [[[78.0, 19.0], [78.1, 19.0], [78.1, 19.1], [78.0, 19.1], [78.0, 19.0]]]},
        as_of=as_of,
        district=district,
        sowing_date=sowing_date,
    )
    return ObsBundle(field=field, weather_daily=weather_days)


def test_15_dry_days_fires():
    """15 consecutive dry days (< 2.5 mm) fires both mid-season and early-season rules."""
    as_of = date(2026, 8, 15)
    sowing = as_of - timedelta(days=20)
    # 15 days of 0.0 mm rain
    weather = _make_weather_days(as_of, [0.0] * 15)
    bundle = _make_bundle(as_of, weather, district="Adilabad", sowing_date=sowing)

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    # mid_season (> 14 days)
    assert by_id["mid_season_dry_spell"].fired is True
    assert "15 consecutive dry days" in by_id["mid_season_dry_spell"].detail
    assert len(by_id["mid_season_dry_spell"].chunk_id) > 0
    assert len(by_id["mid_season_dry_spell"].evidence_ids) == 3

    # early_season (>= 15 days after sowing)
    assert by_id["early_season_dry_spell"].fired is True
    assert len(by_id["early_season_dry_spell"].chunk_id) > 0


def test_13_dry_days_does_not_fire():
    """13 consecutive dry days (< 2.5 mm) does not fire mid-season (needs > 14 days)."""
    as_of = date(2026, 8, 15)
    # 13 days of dry preceded by 1 day of 10.0 mm rain
    weather = _make_weather_days(as_of, [10.0] + [0.5] * 13)
    bundle = _make_bundle(as_of, weather, district="Mahabubnagar")

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    assert by_id["mid_season_dry_spell"].fired is False
    assert "current dry run is 13 days" in by_id["mid_season_dry_spell"].detail


def test_3mm_day_resets_run():
    """A day with 3.0 mm rain (>= 2.5 mm threshold) breaks the dry spell run."""
    as_of = date(2026, 8, 20)
    # 10 dry days, then 3.0 mm rain, then 5 dry days
    precip = [0.0] * 10 + [3.0] + [0.0] * 5
    weather = _make_weather_days(as_of, precip)
    bundle = _make_bundle(as_of, weather, district="Karimnagar")

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    # Only 5 days in the current dry run
    assert by_id["mid_season_dry_spell"].fired is False
    assert "current dry run is 5 days" in by_id["mid_season_dry_spell"].detail


def test_forecast_rows_are_ignored():
    """Weather rows marked is_forecast=True are ignored when checking historical dry run."""
    as_of = date(2026, 8, 15)
    # 14 days of dry, but 2 of them are marked is_forecast=True
    precip = [0.0] * 14
    flags = [False] * 12 + [True, True]
    weather = _make_weather_days(as_of, precip, forecast_flags=flags)
    bundle = _make_bundle(as_of, weather, district="Adilabad")

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    # Forecast rows ignored -> fewer than 14 historical non-forecast days
    assert by_id["mid_season_dry_spell"].fired is False
    assert "Fewer than 14 days" in by_id["mid_season_dry_spell"].detail or "missing" in by_id["mid_season_dry_spell"].detail


def test_missing_sowing_date():
    """Early season dry spell returns fired=False with 'sowing date needed' when missing."""
    as_of = date(2026, 8, 15)
    weather = _make_weather_days(as_of, [0.0] * 20)
    bundle = _make_bundle(as_of, weather, district="Adilabad", sowing_date=None)

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    assert by_id["early_season_dry_spell"].fired is False
    assert by_id["early_season_dry_spell"].detail == "sowing date needed"


def test_missing_weather_day_reported_as_missing_data():
    """A missing day in the lookback window is reported as missing data, not rain."""
    as_of = date(2026, 8, 15)
    # 14 days, but middle day is None (missing)
    precip = [0.0] * 7 + [None] + [0.0] * 6
    weather = _make_weather_days(as_of, precip)
    bundle = _make_bundle(as_of, weather, district="Nalgonda")

    triggers = check_triggers(bundle)
    by_id = {t.id: t for t in triggers}

    assert by_id["mid_season_dry_spell"].fired is False
    assert "Weather data is missing" in by_id["mid_season_dry_spell"].detail
    assert "not that it rained" in by_id["mid_season_dry_spell"].detail


def test_unknown_district_returns_fired_false_no_invented_chunk_id():
    """An unknown district without an ICAR plan returns fired=False and empty chunk_id."""
    as_of = date(2026, 8, 15)
    weather = _make_weather_days(as_of, [0.0] * 20)
    # Hyderabad is not in ICAR corpus (urban district)
    bundle = _make_bundle(as_of, weather, district="Hyderabad")

    triggers = check_triggers(bundle)
    for t in triggers:
        assert t.fired is False
        assert t.chunk_id == ""
        assert "No ICAR plan exists" in t.detail

