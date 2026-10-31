from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np

from config import (
    BASELINE_HALF_WINDOW_DAYS,
    BASELINE_START_YEAR,
    MIN_BASELINE_OBS,
    MIN_BASELINE_YEARS,
    MODIS_COMPOSITE_DAYS,
    SENSOR_SIGN,
    SIGMA_FLOOR,
    baseline_group,
)


@dataclass(frozen=True)
class BaselineStats:
    mean: float
    std: float
    n_obs: int
    n_years: int


def _finite_float_values(values) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        out.append(fv)
    return out


def circ_doy_dist(a: date, b: date) -> int:
    """Circular day-of-year distance with Dec/Jan wrap."""
    year_span = 366 if (a.year % 4 == 0 and (a.year % 100 != 0 or a.year % 400 == 0)) or (b.year % 4 == 0 and (b.year % 100 != 0 or b.year % 400 == 0)) else 365
    delta = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(delta, year_span - delta)


def _baseline_candidates(days, values, target_day: date, sensor: str):
    group = baseline_group(sensor)
    start_year = BASELINE_START_YEAR[group]
    selected_days: list[date] = []
    selected_values: list[float] = []
    for day, value in zip(days, values):
        if day is None:
            continue
        if not isinstance(day, date):
            try:
                day = date.fromisoformat(str(day))
            except ValueError:
                continue
        if day.year < start_year or day.year >= target_day.year:
            continue
        if circ_doy_dist(day, target_day) > BASELINE_HALF_WINDOW_DAYS:
            continue
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            continue
        selected_days.append(day)
        selected_values.append(fv)
    return selected_days, selected_values


def baseline_stats(days, values, target_day: date, sensor: str) -> BaselineStats | None:
    """Return pooled baseline stats for same-day window in prior years only."""
    selected_days, selected_values = _baseline_candidates(days, values, target_day, sensor)
    if not selected_values:
        return None
    n_years = len({d.year for d in selected_days})
    group = baseline_group(sensor)
    if n_years < MIN_BASELINE_YEARS[group] or len(selected_values) < MIN_BASELINE_OBS:
        return None
    mean = float(np.mean(selected_values))
    if len(selected_values) > 1:
        std = float(np.std(selected_values, ddof=1))
    else:
        std = 0.0
    return BaselineStats(mean=mean, std=std, n_obs=len(selected_values), n_years=n_years)


def zscore(x, stats: BaselineStats | None, sensor: str) -> float | None:
    if stats is None:
        return None
    sigma = max(stats.std, SIGMA_FLOOR[sensor])
    return float((x - stats.mean) / sigma)


def percentile(values, q: float) -> float:
    nums = sorted(_finite_float_values(values))
    if not nums:
        return float("nan")
    if len(nums) == 1:
        return float(nums[0])
    if q <= 0:
        return float(nums[0])
    if q >= 100:
        return float(nums[-1])
    position = (len(nums) - 1) * (q / 100.0)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(nums[lo])
    fraction = position - lo
    return float(nums[lo] + (nums[hi] - nums[lo]) * fraction)


def percentile_rank(x, days, values, target_day: date, sensor: str) -> float | None:
    selected_days, selected_values = _baseline_candidates(days, values, target_day, sensor)
    if not selected_values:
        return None
    x = float(x)
    count_less = sum(1 for v in selected_values if v < x)
    count_equal = sum(1 for v in selected_values if math.isclose(v, x, rel_tol=1e-12, abs_tol=1e-12))
    percentile = 100.0 * (count_less + 0.5 * count_equal) / len(selected_values)
    return min(100.0, max(0.0, percentile))


def to_stress(sensor: str, z: float) -> float:
    return float(SENSOR_SIGN[sensor] * z)


def forecast_stress_interval(fc) -> tuple[float | None, float | None, float | None]:
    if fc is None:
        return (None, None, None)
    if isinstance(fc, dict):
        p10 = fc.get("ndvi_z_p10")
        p50 = fc.get("ndvi_z_p50")
        p90 = fc.get("ndvi_z_p90")
    else:
        p10 = getattr(fc, "ndvi_z_p10", None)
        p50 = getattr(fc, "ndvi_z_p50", None)
        p90 = getattr(fc, "ndvi_z_p90", None)
    lo = None if p90 is None else -float(p90)
    mid = None if p50 is None else -float(p50)
    hi = None if p10 is None else -float(p10)
    return (lo, mid, hi)


def modis_end_date(start: date) -> date:
    return start + timedelta(days=MODIS_COMPOSITE_DAYS - 1)


def modis_usable(start: date, as_of: date) -> bool:
    return modis_end_date(start) <= as_of


__all__ = [
    "BaselineStats",
    "circ_doy_dist",
    "baseline_stats",
    "zscore",
    "percentile",
    "percentile_rank",
    "to_stress",
    "forecast_stress_interval",
    "modis_end_date",
    "modis_usable",
]
