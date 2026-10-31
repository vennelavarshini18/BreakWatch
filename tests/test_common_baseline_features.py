from datetime import date

import pandas as pd
import pytest

from common.baseline import (
    BaselineStats,
    circ_doy_dist,
    forecast_stress_interval,
    modis_end_date,
    modis_usable,
    percentile_rank,
    to_stress,
    zscore,
)
from common.features import FEATURE_NAMES, aggregate_hourly_to_daily, build_features
from config import SM_DAILY_MIN_HOURS


def make_modis_baseline(target_day: date):
    days = []
    values = []
    for year in range(2015, target_day.year):
        d = date(year, target_day.month, target_day.day)
        if d.month == 2 and d.day == 29 and year % 4 != 0:
            continue
        days.append(d)
        values.append(0.42 + (year - 2015) * 0.01)
    return days, values


def test_circ_doy_dist_wraps_across_year_boundary():
    assert circ_doy_dist(date(2024, 1, 5), date(2024, 12, 31)) == 5


def test_baseline_stats_excludes_current_year_and_rejects_low_coverage():
    target_day = date(2025, 1, 8)
    days, values = make_modis_baseline(target_day)
    stats = __import__('common.baseline', fromlist=['baseline_stats']).baseline_stats(days, values, target_day, 'modis_ndvi')
    assert stats is not None
    assert stats.n_years >= 5
    days_plus_future = days + [date(2025, 1, 7)]
    values_plus_future = values + [99.0]
    stats2 = __import__('common.baseline', fromlist=['baseline_stats']).baseline_stats(days_plus_future, values_plus_future, target_day, 'modis_ndvi')
    assert stats2 == stats


def test_zscore_uses_sigma_floor():
    stats = BaselineStats(mean=10.0, std=0.0, n_obs=10, n_years=10)
    assert zscore(10.05, stats, 'modis_ndvi') == pytest.approx(1.6666666666666667, rel=1e-12)


def test_percentile_rank_is_mid_rank_for_ties():
    vals = [1.0, 2.0, 2.0, 3.0]
    rank = percentile_rank(2.0, [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)], vals, date(2024, 1, 1), 'modis_ndvi')
    assert rank == 50.0


def test_to_stress_and_forecast_stress_interval_signs():
    assert to_stress('modis_ndvi', -1.0) == 1.0
    fc = type('X', (), {"ndvi_z_p10": -2.0, "ndvi_z_p50": -1.0, "ndvi_z_p90": -0.5})()
    assert forecast_stress_interval(fc) == (0.5, 1.0, 2.0)


def test_modis_end_date_rule():
    start = date(2025, 8, 1)
    assert modis_end_date(start) == date(2025, 8, 16)
    assert modis_usable(start, date(2025, 8, 16)) is True
    assert modis_usable(start, date(2025, 8, 15)) is False


def test_hourly_to_daily_uses_min_valid_hours():
    day = pd.Timestamp('2024-01-01 00:00:00', tz='Asia/Kolkata')
    rows = []
    for hour in range(17):
        rows.append({"timestamp": day + pd.Timedelta(hours=hour), "sm_root": 0.23})
    rows.append({"timestamp": day + pd.Timedelta(hours=17), "sm_root": None})
    df = pd.DataFrame(rows)
    out = aggregate_hourly_to_daily(df, ['sm_root'])
    assert pd.isna(out['sm_root'].iloc[0])


def test_build_features_is_stable_when_future_rows_are_appended():
    as_of = date(2024, 1, 10)
    weather = pd.DataFrame(
        {
            'day': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 7), date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)],
            'precip_mm': [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'et0_mm': [2.0, 2.2, 2.1, 2.3, 2.5, 2.0, 2.1, 2.2, 2.3, 2.4],
            'tmax_c': [30.0, 31.0, 31.5, 32.0, 31.8, 30.5, 30.9, 31.2, 30.7, 31.0],
            'sm_shallow': [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],
            'sm_root': [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        }
    )
    # MODIS ndvi with start_date / end_date columns (D5 contract)
    ndvi = pd.DataFrame(
        {
            'start_date': [date(2023, 12, 5), date(2023, 12, 21)],
            'end_date':   [date(2023, 12, 20), date(2024, 1, 5)],
            'ndvi': [0.5, 0.55],
        }
    )
    base = build_features(weather, ndvi, as_of)  # (weather_daily, modis_ndvi, as_of)
    weather_future = pd.concat([
        weather,
        pd.DataFrame([{'day': date(2024, 1, 11), 'precip_mm': 5.0, 'et0_mm': 2.1, 'tmax_c': 31.0, 'sm_shallow': 0.30, 'sm_root': 0.31}])
    ], ignore_index=True)
    future = build_features(weather_future, ndvi, as_of)
    assert base == future
    assert 'doy_sin' in FEATURE_NAMES


def test_all_feature_names_are_present():
    assert FEATURE_NAMES == (
        'rain_deficit_14d_mm',
        'days_since_rain_gt_2p5mm',
        'sm_shallow_pctl',
        'sm_root_pctl',
        'ndvi_z',
        'ndvi_z_slope_3comp',
        'et0_sum_14d',
        'tmax_mean_7d',
        'doy_sin',
        'doy_cos',
    )
