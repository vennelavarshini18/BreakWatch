"""
tests/test_features.py

Unit tests for common/features.py.
All data is synthetic and constructed inline — no network, no real files.
Tests are marked with comments indicating which spec requirement they cover.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from common.features import (
    FEATURE_NAMES,
    STALE_NDVI_DAYS,
    aggregate_hourly_to_daily,
    build_features,
    consecutive_dry_days,
    days_since_rain_gt_2p5mm,
    feature_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"


def _utc_ts(d: date, hour: int) -> pd.Timestamp:
    """UTC timestamp for a given date + hour."""
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour, tz="UTC")


def _weather_df(
    start: date,
    n_days: int,
    precip: float = 0.0,
    et0: float = 2.0,
    tmax: float = 32.0,
    sm_shallow: float = 0.20,
    sm_root: float = 0.22,
) -> pd.DataFrame:
    """Build a simple uniform daily weather DataFrame."""
    days = [start + timedelta(days=i) for i in range(n_days)]
    return pd.DataFrame(
        {
            "day": days,
            "precip_mm": [precip] * n_days,
            "et0_mm": [et0] * n_days,
            "tmax_c": [tmax] * n_days,
            "sm_shallow": [sm_shallow] * n_days,
            "sm_root": [sm_root] * n_days,
        }
    )


def _modis_df(end_dates: list[date], ndvi_vals: list[float]) -> pd.DataFrame:
    """Build a MODIS composite DataFrame."""
    return pd.DataFrame(
        {
            "start_date": [d - timedelta(days=15) for d in end_dates],
            "end_date": end_dates,
            "ndvi": ndvi_vals,
        }
    )


# ---------------------------------------------------------------------------
# 1. aggregate_hourly_to_daily — 18-hour threshold and IST boundary
# ---------------------------------------------------------------------------


class TestAggregateHourlyToDaily:
    """Spec item 1: 17/24 hours → NaN; 18/24 hours → value; IST boundary respected."""

    def _make_series(self, d: date, n_hours: int, value: float = 1.0) -> pd.Series:
        """UTC timestamps for the first *n_hours* hours of IST day *d*."""
        # IST midnight = UTC 18:30 of the PREVIOUS day
        ist_midnight_utc = pd.Timestamp(
            year=d.year, month=d.month, day=d.day,
            hour=0, minute=0, tz=IST
        ).tz_convert("UTC")
        timestamps = [ist_midnight_utc + pd.Timedelta(hours=h) for h in range(n_hours)]
        return pd.Series([value] * n_hours, index=timestamps)

    def test_17_hours_gives_nan(self):
        """17 valid hours → below threshold of 18 → NaN."""
        s = self._make_series(date(2024, 6, 15), 17)
        result = aggregate_hourly_to_daily(s)
        assert result.empty or all(math.isnan(v) for v in result.values), (
            f"Expected NaN for 17-hour day, got {result.values}"
        )

    def test_18_hours_gives_value(self):
        """18 valid hours → meets threshold → returns a value (not NaN)."""
        s = self._make_series(date(2024, 6, 15), 18, value=3.0)
        result = aggregate_hourly_to_daily(s)
        assert not result.empty
        assert not math.isnan(float(result.iloc[0])), "Expected a value for 18-hour day"

    def test_ist_day_boundary(self):
        """Hours crossing the IST midnight boundary land on the correct IST date.

        UTC 18:30 Jun-14 = IST 00:00 Jun-15.
        An hour at UTC 18:29 Jun-14 belongs to IST Jun-14;
        an hour at UTC 18:30 Jun-14 belongs to IST Jun-15.
        Fill both IST dates with 18+ hours so neither is NaN.
        """
        base_ist_14 = pd.Timestamp(2024, 6, 14, 0, 0, tz=IST).tz_convert("UTC")
        base_ist_15 = pd.Timestamp(2024, 6, 15, 0, 0, tz=IST).tz_convert("UTC")

        # 18 hours each, value 1.0 on Jun-14 IST and 2.0 on Jun-15 IST
        ts_14 = [base_ist_14 + pd.Timedelta(hours=h) for h in range(18)]
        ts_15 = [base_ist_15 + pd.Timedelta(hours=h) for h in range(18)]
        vals = [1.0] * 18 + [2.0] * 18
        s = pd.Series(vals, index=ts_14 + ts_15)

        result = aggregate_hourly_to_daily(s)
        result_map = {d: v for d, v in zip(result.index, result.values)}

        jun14 = date(2024, 6, 14)
        jun15 = date(2024, 6, 15)
        assert jun14 in result_map and not math.isnan(result_map[jun14])
        assert jun15 in result_map and not math.isnan(result_map[jun15])
        assert result_map[jun14] == pytest.approx(1.0)
        assert result_map[jun15] == pytest.approx(2.0)

    def test_how_sum(self):
        """how='sum' accumulates rather than averages."""
        s = self._make_series(date(2024, 6, 15), 18, value=2.0)
        mean_result = aggregate_hourly_to_daily(s, how="mean")
        sum_result = aggregate_hourly_to_daily(s, how="sum")
        assert float(sum_result.iloc[0]) == pytest.approx(float(mean_result.iloc[0]) * 18)

    def test_empty_returns_empty(self):
        s = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        result = aggregate_hourly_to_daily(s)
        assert result.empty


# ---------------------------------------------------------------------------
# 2. consecutive_dry_days
# ---------------------------------------------------------------------------


class TestConsecutiveDryDays:
    """Spec item 2: run counting, rain reset, NaN termination."""

    def _make_precip(self, days_vals: list[tuple[date, float | None]]) -> pd.Series:
        days, vals = zip(*days_vals)
        return pd.Series(list(vals), index=pd.to_datetime(list(days)))

    def test_15_dry_days(self):
        """15 consecutive dry days (0 mm) gives run_length=15."""
        as_of = date(2024, 8, 15)
        pairs = [(as_of - timedelta(days=i), 0.0) for i in range(20)]
        s = self._make_precip(pairs)
        length, start, end = consecutive_dry_days(s, as_of)
        assert length == 15 or length == 20  # stops at lookback boundary or full run
        assert length >= 15
        assert end == as_of

    def test_15_exact_dry_days_then_rain(self):
        """Exactly 15 dry days then a 3 mm rain day resets the run to 15."""
        as_of = date(2024, 8, 15)
        pairs = [(as_of - timedelta(days=i), 0.0) for i in range(15)]
        # Day 15 ago has 3 mm — ends the dry run
        pairs.append((as_of - timedelta(days=15), 3.0))
        s = self._make_precip(pairs)
        length, start, end = consecutive_dry_days(s, as_of)
        assert length == 15
        assert start == as_of - timedelta(days=14)
        assert end == as_of

    def test_rain_day_resets_run(self):
        """A 3 mm day resets: 5 dry, 1 rainy, 5 dry → run = 5, not 10."""
        as_of = date(2024, 8, 10)
        # Days 0-4 (from as_of): dry; day 5: rain; days 6-10: dry
        pairs = []
        for i in range(5):
            pairs.append((as_of - timedelta(days=i), 0.0))
        pairs.append((as_of - timedelta(days=5), 3.0))
        for i in range(6, 11):
            pairs.append((as_of - timedelta(days=i), 0.0))
        s = self._make_precip(pairs)
        length, _, end = consecutive_dry_days(s, as_of)
        assert length == 5
        assert end == as_of

    def test_nan_ends_run(self):
        """A NaN day terminates the run; the returned run_length covers days before the NaN."""
        as_of = date(2024, 8, 10)
        pairs = [
            (as_of, 0.0),
            (as_of - timedelta(days=1), 0.0),
            (as_of - timedelta(days=2), float("nan")),  # NaN
            (as_of - timedelta(days=3), 0.0),
        ]
        s = self._make_precip(pairs)
        length, start, end = consecutive_dry_days(s, as_of)
        # Run goes back 2 days (as_of and as_of-1), stopped by NaN
        assert length == 2
        assert end == as_of

    def test_zero_run_when_as_of_has_rain(self):
        """If as_of itself has rain, run_length=0 and start/end are None."""
        as_of = date(2024, 8, 10)
        pairs = [(as_of, 5.0), (as_of - timedelta(days=1), 0.0)]
        s = self._make_precip(pairs)
        length, start, end = consecutive_dry_days(s, as_of)
        assert length == 0
        assert start is None
        assert end is None

    def test_rows_after_as_of_ignored(self):
        """Rows with day > as_of must not affect the result."""
        as_of = date(2024, 8, 10)
        pairs = [(as_of - timedelta(days=i), 0.0) for i in range(5)]
        pairs.append((as_of + timedelta(days=1), 0.0))  # future row
        s = self._make_precip(pairs)
        length_with_future, _, _ = consecutive_dry_days(s, as_of)

        pairs2 = pairs[:-1]  # without future row
        s2 = self._make_precip(pairs2)
        length_without, _, _ = consecutive_dry_days(s2, as_of)

        assert length_with_future == length_without


# ---------------------------------------------------------------------------
# 3. build_features — leakage and look-ahead tests
# ---------------------------------------------------------------------------


def _rich_weather(as_of: date, n_history: int = 5000) -> pd.DataFrame:
    """Build a multi-year weather DataFrame ending on as_of with no gaps."""
    start = as_of - timedelta(days=n_history)
    return _weather_df(start, n_history + 1, precip=0.0, et0=2.5, tmax=31.0,
                       sm_shallow=0.21, sm_root=0.23)


def _rich_modis(as_of: date, n_composites: int = 40) -> pd.DataFrame:
    """Build MODIS composites ending on or before as_of (16-day spacing)."""
    end_dates = [as_of - timedelta(days=16 * i) for i in range(n_composites)]
    end_dates = [d for d in end_dates if d <= as_of]
    end_dates.sort()
    ndvi_vals = [0.5 + 0.01 * (i % 5) for i in range(len(end_dates))]
    return _modis_df(end_dates, ndvi_vals)


class TestBuildFeaturesLeakage:
    """LEAKAGE: weather rows or MODIS composites after as_of must not affect output."""

    def test_weather_row_after_as_of_does_not_change_output(self):
        as_of = date(2024, 8, 15)
        wx = _rich_weather(as_of)
        mdv = _rich_modis(as_of)

        f_base = build_features(wx, mdv, as_of)

        # Add a future weather row with extreme values
        future_row = pd.DataFrame([{
            "day": as_of + timedelta(days=1),
            "precip_mm": 999.0, "et0_mm": 999.0, "tmax_c": 999.0,
            "sm_shallow": 0.99, "sm_root": 0.99,
        }])
        wx_future = pd.concat([wx, future_row], ignore_index=True)
        f_future = build_features(wx_future, mdv, as_of)

        assert f_base == f_future, "Future weather rows changed the output"

    def test_modis_composite_end_after_as_of_not_used(self):
        """A composite with start_date <= as_of but end_date > as_of must be excluded."""
        as_of = date(2024, 8, 15)
        wx = _rich_weather(as_of)
        mdv = _rich_modis(as_of)

        f_base = build_features(wx, mdv, as_of)

        # Add a composite that straddles as_of (start before, end after)
        leaky = pd.DataFrame([{
            "start_date": as_of - timedelta(days=5),
            "end_date": as_of + timedelta(days=10),  # end after as_of!
            "ndvi": 0.99,  # would distort z-score if leaked
        }])
        mdv_leaky = pd.concat([mdv, leaky], ignore_index=True)
        f_leaky = build_features(wx, mdv_leaky, as_of)

        assert f_base == f_leaky, "Composite with end_date > as_of leaked into output"

    def test_composite_partially_in_future_excluded(self):
        """start_date <= as_of < end_date → composite is NOT usable (D5)."""
        as_of = date(2024, 8, 15)
        wx = _rich_weather(as_of)
        # One composite that ends 1 day after as_of — must be excluded
        mdv = pd.DataFrame([{
            "start_date": as_of - timedelta(days=15),
            "end_date": as_of + timedelta(days=1),  # end_date > as_of
            "ndvi": 0.5,
        }])
        f = build_features(wx, mdv, as_of)
        # ndvi_z must be None because the only composite is unusable
        assert f["ndvi_z"] is None, "Composite with end_date > as_of should not produce ndvi_z"


# ---------------------------------------------------------------------------
# 4. build_features — None on insufficient data
# ---------------------------------------------------------------------------


class TestBuildFeaturesNone:
    """Features are None when data is insufficient — never a default number."""

    def test_empty_weather_gives_none_features(self):
        as_of = date(2024, 8, 15)
        wx = pd.DataFrame(columns=["day", "precip_mm", "et0_mm", "tmax_c", "sm_shallow", "sm_root"])
        mdv = pd.DataFrame(columns=["start_date", "end_date", "ndvi"])
        f = build_features(wx, mdv, as_of)
        # doy_sin and doy_cos are always available; everything else should be None
        assert f["rain_deficit_14d_mm"] is None
        assert f["days_since_rain_gt_2p5mm"] is None
        assert f["sm_shallow_pctl"] is None
        assert f["sm_root_pctl"] is None
        assert f["ndvi_z"] is None
        assert f["ndvi_z_slope_3comp"] is None
        assert f["et0_sum_14d"] is None
        assert f["tmax_mean_7d"] is None
        # doy features are always computable
        assert f["doy_sin"] is not None
        assert f["doy_cos"] is not None

    def test_et0_below_min_days_is_none(self):
        """Fewer than 12 of 14 days present → et0_sum_14d is None."""
        as_of = date(2024, 8, 15)
        # Only 11 days of et0 in the 14-day window
        days = [as_of - timedelta(days=i) for i in range(11)]
        wx = pd.DataFrame({
            "day": days,
            "precip_mm": [0.0] * 11,
            "et0_mm": [2.0] * 11,
            "tmax_c": [32.0] * 11,
            "sm_shallow": [0.2] * 11,
            "sm_root": [0.22] * 11,
        })
        mdv = pd.DataFrame(columns=["start_date", "end_date", "ndvi"])
        f = build_features(wx, mdv, as_of)
        assert f["et0_sum_14d"] is None

    def test_tmax_below_min_days_is_none(self):
        """Fewer than 6 of 7 days present → tmax_mean_7d is None."""
        as_of = date(2024, 8, 15)
        days = [as_of - timedelta(days=i) for i in range(5)]
        wx = pd.DataFrame({
            "day": days,
            "precip_mm": [0.0] * 5,
            "et0_mm": [2.0] * 5,
            "tmax_c": [32.0] * 5,
            "sm_shallow": [0.2] * 5,
            "sm_root": [0.22] * 5,
        })
        mdv = pd.DataFrame(columns=["start_date", "end_date", "ndvi"])
        f = build_features(wx, mdv, as_of)
        assert f["tmax_mean_7d"] is None

    def test_stale_ndvi_gives_none(self):
        """NDVI composite whose end_date is > STALE_NDVI_DAYS before as_of → ndvi_z is None."""
        as_of = date(2024, 8, 15)
        wx = _rich_weather(as_of)
        stale_end = as_of - timedelta(days=STALE_NDVI_DAYS + 1)
        # Build 10 composites to satisfy baseline minimum, all stale
        end_dates = [stale_end - timedelta(days=16 * i) for i in range(9, -1, -1)]
        mdv = _modis_df(end_dates, [0.5] * 10)
        f = build_features(wx, mdv, as_of)
        assert f["ndvi_z"] is None, "Stale NDVI composite should give ndvi_z=None"

    def test_ndvi_slope_none_when_fewer_than_3_usable(self):
        """ndvi_z_slope_3comp is None when < 3 composites have computable z-scores."""
        as_of = date(2024, 8, 15)
        wx = _rich_weather(as_of)
        # Only 2 composites
        end_dates = [as_of - timedelta(days=16), as_of]
        mdv = _modis_df(end_dates, [0.5, 0.52])
        f = build_features(wx, mdv, as_of)
        assert f["ndvi_z_slope_3comp"] is None


# ---------------------------------------------------------------------------
# 5. feature_names() matches build_features keys
# ---------------------------------------------------------------------------


def test_feature_names_matches_build_features_keys():
    """feature_names() must return exactly the keys that build_features returns."""
    as_of = date(2024, 8, 15)
    wx = _rich_weather(as_of)
    mdv = _rich_modis(as_of)
    f = build_features(wx, mdv, as_of)

    names = feature_names()
    assert names == list(FEATURE_NAMES), "feature_names() diverged from FEATURE_NAMES constant"
    assert set(names) == set(f.keys()), (
        f"feature_names() keys differ from build_features keys.\n"
        f"  In names but not features: {set(names) - set(f.keys())}\n"
        f"  In features but not names: {set(f.keys()) - set(names)}"
    )
    assert names == list(f.keys()), "Key order differs between feature_names() and build_features()"


# ---------------------------------------------------------------------------
# 6. days_since_rain_gt_2p5mm — public function
# ---------------------------------------------------------------------------


def test_days_since_rain_gt_2p5mm_basic():
    """5 dry days after the last rainy day → returns 5."""
    as_of = date(2024, 8, 15)
    pairs = [(as_of - timedelta(days=i), 0.0) for i in range(5)]
    pairs.append((as_of - timedelta(days=5), 5.0))  # rain 5 days ago
    for i in range(6, 20):
        pairs.append((as_of - timedelta(days=i), 0.0))
    s = pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs]))
    result = days_since_rain_gt_2p5mm(s, as_of)
    assert result == 5


def test_days_since_rain_capped_at_lookback():
    """If no rain within lookback, returns lookback."""
    as_of = date(2024, 8, 15)
    pairs = [(as_of - timedelta(days=i), 0.0) for i in range(80)]
    s = pd.Series([v for _, v in pairs], index=pd.to_datetime([d for d, _ in pairs]))
    result = days_since_rain_gt_2p5mm(s, as_of, lookback=60)
    assert result == 60
