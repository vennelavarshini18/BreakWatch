"""
tests/test_baseline.py

Synthetic-data unit tests for common/baseline.py.
All series are built from known values so expected results can be hand-verified.
No network calls, no real data.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from common.baseline import (
    baseline_window_dates,
    compute_percentile,
    compute_z,
    forecast_stress_interval,
    rain_normal_14d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(dates_values: list[tuple[dt.date, float]]) -> pd.Series:
    """Build a pd.Series from (date, value) pairs."""
    dates, values = zip(*dates_values)
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


def _uniform_series(start_year: int, end_year: int, month: int, day: int, value: float = 0.5) -> pd.Series:
    """One observation per year on the same calendar date."""
    pairs = []
    for yr in range(start_year, end_year):
        try:
            pairs.append((dt.date(yr, month, day), value))
        except ValueError:
            pass  # skip Feb 29 in non-leap years
    return _series(pairs)


# ---------------------------------------------------------------------------
# T1 — Circular wrap: Dec 28 target captures Jan 1-7 observations from other years
# ---------------------------------------------------------------------------


def test_circular_wrap_dec_target_pulls_jan_observations():
    """
    target_date = 2024-12-28 (DoY 363).
    DoY window = [363-10, 363+10] wrapping around → includes DoY 1..7 (Jan 1-7).
    Observations on 2023-01-05 (DoY 5) are within 10 days circularly (365-363+5=7 ≤ 10).
    """
    target = dt.date(2024, 12, 28)
    # Build series: one MODIS-group observation on Jan 5 for each of several years
    pairs = [(dt.date(yr, 1, 5), 0.4) for yr in range(2018, 2024)]
    # Also add a July observation that must NOT be captured
    pairs += [(dt.date(yr, 7, 15), 0.9) for yr in range(2018, 2024)]
    s = _series(pairs)

    windowed = baseline_window_dates(target, s)
    dates_in = pd.to_datetime(windowed.index).date

    # All Jan-5 observations (years 2018–2023, strictly before 2024) must appear
    jan5_in = [d for d in dates_in if d.month == 1 and d.day == 5]
    assert len(jan5_in) == 6, f"Expected 6 Jan-5 obs, got {len(jan5_in)}"

    # July observations must NOT appear
    jul_in = [d for d in dates_in if d.month == 7]
    assert len(jul_in) == 0, f"July obs leaked into window: {jul_in}"


# ---------------------------------------------------------------------------
# T2 — Strictly-before rule: rows in or after target_date.year are excluded
# ---------------------------------------------------------------------------


def test_strictly_before_year_rule():
    """
    Observations exactly in target_date.year or later must not affect mean/std/z.
    """
    target = dt.date(2024, 6, 15)
    sensor = "modis_ndvi"

    # 6 years of legitimate history (2018–2023) → z computable
    pairs_before = [(dt.date(yr, 6, 15), 0.5) for yr in range(2018, 2024)]
    s_before = _series(pairs_before)
    z_before, n_yrs_before, _ = compute_z(0.5, s_before, target, sensor)

    # Add 2024 and 2025 rows with wildly different values — must not shift z
    pairs_with = pairs_before + [
        (dt.date(2024, 6, 15), 99.0),
        (dt.date(2025, 6, 15), -99.0),
    ]
    s_with = _series(pairs_with)
    z_with, n_yrs_with, _ = compute_z(0.5, s_with, target, sensor)

    assert n_yrs_before == n_yrs_with, "Year count changed after adding future rows"
    # z should be identical (same baseline pool)
    assert z_before == pytest.approx(z_with), "Future rows shifted the z-score"


# ---------------------------------------------------------------------------
# T3a — Fewer than min_years → z is None
# ---------------------------------------------------------------------------


def test_insufficient_years_returns_none():
    """
    MODIS needs ≥ 5 years.  3 years of data → z = None.
    """
    target = dt.date(2024, 6, 15)
    sensor = "modis_ndvi"

    # Only 3 years (2018, 2019, 2020) — each has 2 obs to exceed MIN_BASELINE_OBS
    pairs = [
        (dt.date(2018, 6, 14), 0.5),
        (dt.date(2018, 6, 16), 0.5),
        (dt.date(2019, 6, 14), 0.5),
        (dt.date(2019, 6, 16), 0.5),
        (dt.date(2020, 6, 14), 0.5),
        (dt.date(2020, 6, 16), 0.5),
    ]
    s = _series(pairs)
    z, n_yrs, n_obs = compute_z(0.5, s, target, sensor)

    assert z is None, f"Expected None z for {n_yrs} years, got {z}"
    assert n_yrs == 3


# ---------------------------------------------------------------------------
# T3b — Fewer than 6 observations → z is None (even if enough years)
# ---------------------------------------------------------------------------


def test_insufficient_obs_returns_none():
    """
    5 years but only 5 total observations → n_obs < MIN_BASELINE_OBS(6) → None.
    """
    target = dt.date(2024, 6, 15)
    sensor = "modis_ndvi"

    # 5 distinct years, 1 obs each = 5 obs total
    pairs = [(dt.date(yr, 6, 15), 0.5) for yr in range(2018, 2023)]
    s = _series(pairs)
    z, n_yrs, n_obs = compute_z(0.5, s, target, sensor)

    assert z is None, f"Expected None z for {n_obs} obs, got {z}"
    assert n_yrs == 5
    assert n_obs == 5


# ---------------------------------------------------------------------------
# T4 — Sigma floor is applied when historical std is tiny
# ---------------------------------------------------------------------------


def test_sigma_floor_applied():
    """
    If all pooled values are identical the std=0.  The sigma floor prevents
    division by zero and bounds the z.
    MODIS floor = 0.03.  value offset by 0.06 → z should be 0.06/0.03 = 2.0 exactly.
    """
    target = dt.date(2024, 6, 15)
    sensor = "modis_ndvi"  # sigma_floor = 0.03
    base_val = 0.5

    # 6 identical observations across 6 years (one per year) → std = 0
    pairs = []
    for yr in range(2018, 2024):
        pairs.append((dt.date(yr, 6, 14), base_val))
        # add a second obs per year so n_obs >= 6 with exactly 6 years
    s = _series(pairs)

    value = base_val + 0.06  # 2 sigma-floors above mean
    z, _, _ = compute_z(value, s, target, sensor)

    assert z is not None
    assert z == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T5 — Percentile of the max pooled value ≈ 100, of the min ≈ 0
# ---------------------------------------------------------------------------


def test_percentile_extremes():
    """
    With 10 distinct values [0.1, 0.2, ..., 1.0] across ≥5 years (≥6 obs):
    - querying the maximum (1.0) should return ≈ 90 (9 out of 10 are less)
    - querying the minimum (0.1) should return 0.0 (none are less)
    """
    target = dt.date(2024, 6, 15)
    sensor = "modis_ndvi"

    values_per_yr = np.linspace(0.1, 1.0, 10)
    # Spread across 5 years, 2 obs each (different days in window)
    pairs = []
    yr_list = list(range(2018, 2023))
    for i, v in enumerate(values_per_yr):
        yr = yr_list[i % 5]
        day_offset = i // 5  # 0 or 1
        d = dt.date(yr, 6, 14 + day_offset)
        pairs.append((d, v))
    s = _series(pairs)

    pct_max = compute_percentile(1.0, s, target, sensor)
    pct_min = compute_percentile(0.1, s, target, sensor)

    assert pct_max is not None
    assert pct_min is not None
    # np.mean(values < 1.0) = 9/10 = 0.9 → 90
    assert pct_max == pytest.approx(90.0, abs=1e-9)
    # np.mean(values < 0.1) = 0 → 0
    assert pct_min == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T6 — forecast_stress_interval inverts quantiles correctly
# ---------------------------------------------------------------------------


def test_forecast_stress_interval_inversion():
    """
    ndvi_z: p10=-1.5, p50=-0.5, p90=0.3  (negative = bad in ndvi_z scale)
    stress:  p10 = -ndvi_z_p90 = -0.3   (best stress scenario)
             p50 = -ndvi_z_p50 =  0.5
             p90 = -ndvi_z_p10 =  1.5   (worst stress scenario)
    Ordering must satisfy stress_p10 <= stress_p50 <= stress_p90.
    """
    from common.baseline import forecast_stress_interval

    sp10, sp50, sp90 = forecast_stress_interval(-1.5, -0.5, 0.3)

    assert sp10 == pytest.approx(-0.3)
    assert sp50 == pytest.approx(0.5)
    assert sp90 == pytest.approx(1.5)
    # Ordering check
    assert sp10 <= sp50 <= sp90


def test_forecast_stress_interval_none_propagation():
    """Any None input → corresponding output is None."""
    sp10, sp50, sp90 = forecast_stress_interval(None, -0.5, None)
    assert sp10 is None  # -ndvi_z_p90, and p90 is None
    assert sp50 == pytest.approx(0.5)
    assert sp90 is None  # -ndvi_z_p10, and p10 is None


def test_forecast_stress_interval_all_none():
    sp10, sp50, sp90 = forecast_stress_interval(None, None, None)
    assert sp10 is None
    assert sp50 is None
    assert sp90 is None


# ---------------------------------------------------------------------------
# T7 — rain_normal_14d computes mean of prior-year 14-day totals
# ---------------------------------------------------------------------------


def test_rain_normal_14d_basic():
    """
    Build 5 years of daily rain (2010-2014), exactly 10 mm/day every day.
    14-day total each year = 140 mm.  Normal should be 140 mm.
    """
    target = dt.date(2015, 6, 15)
    pairs = []
    for yr in range(2010, 2015):
        for d in range(1, 30):
            pairs.append((dt.date(yr, 6, d), 10.0))
    s = _series(pairs)

    normal = rain_normal_14d(s, target)
    assert normal is not None
    assert normal == pytest.approx(140.0, abs=1e-6)


def test_rain_normal_14d_empty_returns_none():
    target = dt.date(2024, 6, 15)
    s = pd.Series([], dtype=float)
    assert rain_normal_14d(s, target) is None

