"""
tests/test_backtest.py — Tests for ml/backtest.py.

All tests use a stubbed pipeline (test-only fakes):
  - Trigger date logic (first firing at day 15)
  - False-alarm count on control field (0 false alarms)
  - Fusion unavailable path (NotImplementedError -> skipped forecast, no fabricated FusedState)
  - Forward-compatibility path when fusion succeeds
"""

from __future__ import annotations

import json

from tqdm import tqdm

tqdm.monitor_interval = 0

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from contracts import (
    Climatology,
    Evidence,
    FieldInput,
    FusedState,
    ObsBundle,
    Trigger,
    WeatherDay,
)
from ml import backtest
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


def test_disk_weather_loader_reads_only_rows_through_as_of(tmp_path, monkeypatch):
    """Disk loader preserves archived values, source quality, and the as_of cutoff."""
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2024-08-01",
                "precip_mm": 1.25,
                "et0_mm": 4.0,
                "tmax_c": 31.0,
                "sm_shallow": 0.2,
                "sm_root": 0.3,
            },
            {
                "date": "2024-08-02",
                "precip_mm": 9.0,
                "et0_mm": 5.0,
                "tmax_c": 32.0,
                "sm_shallow": 0.21,
                "sm_root": 0.31,
            },
        ]
    ).to_parquet(weather_dir / "pt_test.parquet", index=False)
    monkeypatch.setattr(backtest, "WEATHER_DIR", weather_dir)

    weather_days, evidence = backtest._load_disk_weather("pt_test", date(2024, 8, 1))

    assert len(weather_days) == 1
    assert weather_days[0].day == date(2024, 8, 1)
    assert weather_days[0].precip_mm == 1.25
    assert weather_days[0].is_forecast is False
    assert weather_days[0].evidence_id == evidence[0].id
    assert evidence[0].quality == "cached"
    assert evidence[0].source == "disk archive (Open-Meteo ERA5 download)"


def test_from_disk_mode_writes_separate_trigger_only_outputs(tmp_path, monkeypatch):
    """Disk mode uses the stubbed archive loader and writes disk-specific outputs."""
    as_of = date(2024, 8, 1)
    fields = [
        {
            "id": "field_disk_test",
            "role": "dry_spell",
            "district": "medak",
            "polygon": {
                "type": "Polygon",
                "coordinates": [[[78, 17], [78.1, 17], [78.1, 17.1], [78, 17.1], [78, 17]]],
            },
            "as_of_dates": [as_of.isoformat()],
            "selection_justification": {
                "coincides_with_training_point": "pt_test",
            },
        }
    ]
    fields_path = tmp_path / "test_fields.json"
    fields_path.write_text(json.dumps(fields), encoding="utf-8")
    monkeypatch.setattr(backtest, "TEST_FIELDS_JSON", fields_path)

    evidence = Evidence(
        id="disk_test",
        key="weather_daily_series",
        value="1 day",
        source="disk archive (Open-Meteo ERA5 download)",
        fetched_at="2024-08-01T00:00:00Z",
        quality="cached",
    )
    weather_days = [
        WeatherDay(
            day=as_of,
            precip_mm=0.5,
            is_forecast=False,
            evidence_id=evidence.id,
        )
    ]
    fired_trigger = Trigger(
        id="mid_season_dry_spell",
        name="Mid-Season Dry Spell (> 2 weeks rainless)",
        fired=True,
        detail="test-only trigger stub",
        chunk_id="",
    )
    with (
        patch.object(backtest, "_load_disk_weather", return_value=(weather_days, [evidence])) as loader,
        patch("ml.triggers.check_triggers", return_value=[fired_trigger]) as check_triggers,
        patch.object(backtest, "run_single_replay_step", side_effect=AssertionError("live replay called")),
    ):
        results, summary = backtest.run_backtest(
            field_ids=[],
            output_dir=tmp_path,
            from_disk=True,
        )

    loader.assert_called_once_with("pt_test", as_of)
    check_triggers.assert_called_once()
    assert results.loc[0, "fusion_status"] == "fusion not run in --from-disk mode"
    assert results.loc[0, "sources_used"] == "disk archive (Open-Meteo ERA5 download)"
    assert "trigger-only replay on archived weather" in summary
    assert "already 15 days long" in summary
    assert "not an early forecast" in summary
    assert "No stress result or lead-time result exists" in summary
    assert (tmp_path / "backtest_results_disk.csv").is_file()
    assert (tmp_path / "backtest_summary_disk.md").is_file()
    assert (tmp_path / "backtest_plots" / "field_disk_test_disk.png").is_file()


def test_from_disk_missing_weather_file_is_recorded_and_skipped():
    """A missing archive is reported without invoking the trigger checker."""
    field_cfg = {
        "id": "field_missing_archive",
        "role": "control",
        "district": "medak",
        "polygon": {"type": "Polygon", "coordinates": [[[78, 17], [78.1, 17], [78.1, 17.1], [78, 17.1], [78, 17]]]},
        "selection_justification": {"coincides_with_training_point": "pt_missing"},
    }

    with (
        patch.object(backtest, "_load_disk_weather", side_effect=FileNotFoundError),
        patch("ml.triggers.check_triggers") as check_triggers,
    ):
        result = backtest.run_single_disk_replay_step(field_cfg, date(2024, 8, 1))

    assert result["observation_status"] == "missing_weather_file: pt_missing.parquet"
    assert result["events"] == ["weather_file_missing:pt_missing.parquet"]
    assert result["fusion_status"] == "fusion not run in --from-disk mode"
    assert result["p_stress"] is None
    assert result["forecast_p50"] is None
    check_triggers.assert_not_called()
