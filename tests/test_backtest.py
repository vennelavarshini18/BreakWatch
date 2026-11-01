"""
tests/test_backtest.py — Tests for ml/backtest.py.

All tests use a stubbed pipeline (test-only fakes):
  - Trigger date logic (first firing at day 15)
  - False-alarm count on control field (0 false alarms)
  - Fusion unavailable path (NotImplementedError -> skipped forecast, no fabricated FusedState)
  - Forward-compatibility path when fusion succeeds
"""

from __future__ import annotations

from tqdm import tqdm
tqdm.monitor_interval = 0

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from contracts import Climatology, Evidence, FieldInput, FusedState, ObsBundle, WeatherDay
from ml.backtest import run_single_replay_step


# ---------------------------------------------------------------------------
# Test Helpers: Synthetic ObsBundle Generator
# ---------------------------------------------------------------------------

def _make_fake_bundle(
    as_of: date,
    precip_history: dict[date, float],
    district: str = "mahabubnagar",
) -> ObsBundle:
    """Build a fake ObsBundle with controlled daily precipitation."""
    polygon = {
        "type": "Polygon",
        "coordinates": [[[78.0, 16.0], [78.01, 16.0], [78.01, 16.01], [78.0, 16.01], [78.0, 16.0]]],
    }
    field = FieldInput(
        polygon=polygon,
        as_of=as_of,
        mode="replay",
        district=district,
        crop=None,
        sowing_date=None,
    )

    days = []
    # Build 60 days of lookback
    for i in range(60):
        d = as_of - timedelta(days=59 - i)
        precip = precip_history.get(d, 0.0)
        days.append(
            WeatherDay(
                day=d,
                precip_mm=precip,
                et0_mm=3.5,
                tmax_c=31.0,
                sm_shallow=0.25,
                sm_root=0.30,
                is_forecast=False,
                evidence_id=f"ev_wx_{d}",
            )
        )

    evidence = [
        Evidence(
            id="ev_001",
            key="test_source",
            value=1.0,
            source="open-meteo:archive",
            fetched_at="2026-11-01T00:00:00Z",
            quality="cached",
        )
    ]

    return ObsBundle(
        field=field,
        obs=[],
        weather_daily=days,
        climatology=Climatology(),
        evidence=evidence,
        events=[],
    )


# ---------------------------------------------------------------------------
# 1. Trigger Date Logic Test
# ---------------------------------------------------------------------------

def test_backtest_trigger_date_logic():
    """Verify that a 25-day dry spell (starting 2026-08-24) fires on day 15 (2026-09-07) and not before."""
    field_cfg = {
        "id": "field_dry_2026",
        "role": "dry_spell",
        "district": "mahabubnagar",
        "polygon": {"type": "Polygon", "coordinates": [[[78.0, 16.0], [78.01, 16.0], [78.01, 16.01], [78.0, 16.01], [78.0, 16.0]]]},
        "crop": None,
        "sowing_date": None,
        "soil": None,
    }

    # Weather: rain up to 2026-08-23, then dry (0 mm) from 2026-08-24 to 2026-09-17
    dry_start = date(2026, 8, 24)
    dry_end = date(2026, 9, 17)

    precip_history: dict[date, float] = {}
    curr = date(2026, 6, 1)
    while curr <= date(2026, 10, 1):
        if dry_start <= curr <= dry_end:
            precip_history[curr] = 0.0
        else:
            precip_history[curr] = 10.0  # rain days
        curr += timedelta(days=1)

    # 1. as_of = 2026-08-31: dry run is 8 days -> mid-season trigger should NOT fire (threshold > 14 days)
    as_of_early = date(2026, 8, 31)
    fake_bundle_early = _make_fake_bundle(as_of_early, precip_history)

    with patch("geo.build_observations", return_value=fake_bundle_early):
        res_early = run_single_replay_step(field_cfg, as_of_early)
        assert res_early["mid_season_fired"] is False
        assert res_early["mid_season_run_length"] == 8

    # 2. as_of = 2026-09-07: dry run is 15 days -> mid-season trigger MUST fire (15 > 14)
    as_of_first_fire = date(2026, 9, 7)
    fake_bundle_fire = _make_fake_bundle(as_of_first_fire, precip_history)

    with patch("geo.build_observations", return_value=fake_bundle_fire):
        res_fire = run_single_replay_step(field_cfg, as_of_first_fire)
        assert res_fire["mid_season_fired"] is True
        assert res_fire["mid_season_run_length"] == 15
        assert res_fire["mid_season_run_start"] == str(dry_start)
        assert res_fire["mid_season_run_end"] == str(as_of_first_fire)
        assert any("Mid-Season Dry Spell" in name for name in res_fire["triggers_fired"])

    # 3. as_of = 2026-09-17: dry run is 25 days -> mid-season trigger continues firing
    as_of_late = date(2026, 9, 17)
    fake_bundle_late = _make_fake_bundle(as_of_late, precip_history)

    with patch("geo.build_observations", return_value=fake_bundle_late):
        res_late = run_single_replay_step(field_cfg, as_of_late)
        assert res_late["mid_season_fired"] is True
        assert res_late["mid_season_run_length"] == 25


# ---------------------------------------------------------------------------
# 2. False-Alarm Count on Control Field Test
# ---------------------------------------------------------------------------

def test_backtest_false_alarm_count_control():
    """Verify that a control field with runs <= 9 dry days produces 0 false alarms."""
    field_cfg = {
        "id": "field_control_mahabubnagar",
        "role": "control",
        "district": "mahabubnagar",
        "polygon": {"type": "Polygon", "coordinates": [[[78.0, 16.0], [78.01, 16.0], [78.01, 16.01], [78.0, 16.01], [78.0, 16.0]]]},
        "crop": None,
        "sowing_date": None,
        "soil": None,
    }

    # Weather: periodic rain every 6 days, so no dry spell ever reaches 10 days
    precip_history: dict[date, float] = {}
    curr = date(2026, 5, 1)
    day_count = 0
    while curr <= date(2026, 9, 1):
        precip_history[curr] = 12.0 if (day_count % 6 == 0) else 0.0
        curr += timedelta(days=1)
        day_count += 1

    eval_dates = [
        date(2026, 7, 1),
        date(2026, 7, 8),
        date(2026, 7, 15),
        date(2026, 7, 22),
        date(2026, 7, 29),
    ]

    false_alarms = 0
    for as_of in eval_dates:
        bundle = _make_fake_bundle(as_of, precip_history)
        with patch("geo.build_observations", return_value=bundle):
            res = run_single_replay_step(field_cfg, as_of)
            if res["mid_season_fired"] or res["early_season_fired"]:
                false_alarms += 1

    assert false_alarms == 0, f"Expected 0 false alarms on control field, got {false_alarms}"


# ---------------------------------------------------------------------------
# 3. Fusion Unavailable Path Test
# ---------------------------------------------------------------------------

def test_backtest_fusion_unavailable_path():
    """When geo.fusion.fuse raises NotImplementedError, record 'fusion unavailable' and skip forecast."""
    field_cfg = {
        "id": "field_dry_2026",
        "role": "dry_spell",
        "district": "mahabubnagar",
        "polygon": {"type": "Polygon", "coordinates": [[[78.0, 16.0], [78.01, 16.0], [78.01, 16.01], [78.0, 16.01], [78.0, 16.0]]]},
        "crop": None,
        "sowing_date": None,
        "soil": None,
    }

    as_of = date(2026, 9, 7)
    bundle = _make_fake_bundle(as_of, {})

    with patch("geo.build_observations", return_value=bundle):
        # geo.fusion.fuse naturally raises NotImplementedError right now
        res = run_single_replay_step(field_cfg, as_of)

        assert res["fusion_status"] == "fusion unavailable"
        assert res["p_stress"] is None
        assert res["risk_level"] is None
        assert res["forecast_status"] == "skipped: fusion unavailable"
        assert res["forecast_p50"] is None


# ---------------------------------------------------------------------------
# 4. Forward-Compatibility Path When Fusion Succeeds
# ---------------------------------------------------------------------------

def test_backtest_fusion_available_path():
    """When geo.fusion.fuse succeeds, fill p_stress, risk level, and attempt forecast."""
    field_cfg = {
        "id": "field_dry_2026",
        "role": "dry_spell",
        "district": "mahabubnagar",
        "polygon": {"type": "Polygon", "coordinates": [[[78.0, 16.0], [78.01, 16.0], [78.01, 16.01], [78.0, 16.01], [78.0, 16.0]]]},
        "crop": None,
        "sowing_date": None,
        "soil": None,
    }

    as_of = date(2026, 9, 7)
    bundle = _make_fake_bundle(as_of, {})

    mock_fused = FusedState(
        series=[],
        latest_mean=0.8,
        latest_std=0.2,
        p_stress=0.35,  # Watch level
        sources_used=["open-meteo:archive"],
    )

    mock_forecast = MagicMock(
        validated_for_season=True,
        ndvi_z_p50=0.15,
        withheld_reason=None,
    )

    with patch("geo.build_observations", return_value=bundle):
        with patch("geo.fusion.fuse", return_value=mock_fused):
            with patch("ml.forecast.forecast", return_value=mock_forecast):
                res = run_single_replay_step(field_cfg, as_of)

                assert res["fusion_status"] == "available"
                assert res["p_stress"] == 0.35
                assert res["risk_level"] == "watch"
                assert res["forecast_status"] == "validated"
                assert res["forecast_p50"] == 0.15
