"""
common/baseline.py — Pure statistical helpers for BreakWatch anomaly detection.

All constants (sigma floors, min years, window size, start years) are read from
config.py so there is a single source of truth.  No network, no clock, no global
state.  Every function is a pure transformation of its arguments.

Leap-year handling
------------------
Day-of-year (DoY) arithmetic is done on a 365-day calendar by mapping each date
to ``date.timetuple().tm_yday``.  Feb 29 in a leap year returns DoY 60; all
subsequent days in that year are shifted by +1 relative to the same date in a
non-leap year.  This is the standard NumPy / pandas convention and is
*intentionally preserved* here so that the ±10-day circular window consistently
captures the right phenological period — the ±1-day drift for post-Feb-28
dates is negligible for a ±10-day window.  We do NOT normalise to a 365-day
year because that would require reindexing all historical data at every call.

Two API styles coexist here (both tested):
  - Primary (P3): list-based API — circ_doy_dist, baseline_stats, zscore,
    percentile_rank, to_stress, forecast_stress_interval(Forecast|dict).
  - Compatibility (P2): pd.Series-based API — compute_z, compute_percentile,
    baseline_window_dates, rain_normal_14d used by ml/ training pipeline.
Both call the same core logic; no duplication of rules.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DOY_YEAR = 365  # fixed-length year for circular arithmetic


@dataclass(frozen=True)
class BaselineStats:
    """Pooled baseline statistics for a sensor/date window."""
    mean: float
    std: float
    n_obs: int
    n_years: int


def _finite_float_values(values) -> list[float]:
    """Filter to finite floats; skip None, NaN, inf."""
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


# ---------------------------------------------------------------------------
# Primary API — list-based (used by P3/agent and features.py)
# ---------------------------------------------------------------------------


def circ_doy_dist(a: date, b: date) -> int:
    """Circular day-of-year distance between two dates with Dec/Jan wrap.

    Uses the actual year_span of whichever year is a leap year so that the
    distance across the Dec 31 → Jan 1 boundary is always 1.
    """
    year_span = (
        366
        if (a.year % 4 == 0 and (a.year % 100 != 0 or a.year % 400 == 0))
        or (b.year % 4 == 0 and (b.year % 100 != 0 or b.year % 400 == 0))
        else 365
    )
    delta = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(delta, year_span - delta)


def _baseline_candidates(days, values, target_day: date, sensor: str):
    """Filter (days, values) to the ±10-day window in prior valid years.

    Returns two parallel lists: selected_days, selected_values (both finite).
    """
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
        # Strictly before target year and not before sensor start year
        if day.year < start_year or day.year >= target_day.year:
            continue
        if circ_doy_dist(day, target_day) > BASELINE_HALF_WINDOW_DAYS:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        selected_days.append(day)
        selected_values.append(fv)
    return selected_days, selected_values


def baseline_stats(days, values, target_day: date, sensor: str) -> Optional[BaselineStats]:
    """Return pooled BaselineStats for the ±10-day window in prior valid years.

    Returns None if n_years < min_years or n_obs < MIN_BASELINE_OBS.
    """
    selected_days, selected_values = _baseline_candidates(days, values, target_day, sensor)
    if not selected_values:
        return None
    n_years = len({d.year for d in selected_days})
    group = baseline_group(sensor)
    if n_years < MIN_BASELINE_YEARS[group] or len(selected_values) < MIN_BASELINE_OBS:
        return None
    mean = float(np.mean(selected_values))
    std = float(np.std(selected_values, ddof=1)) if len(selected_values) > 1 else 0.0
    return BaselineStats(mean=mean, std=std, n_obs=len(selected_values), n_years=n_years)


def zscore(x, stats: Optional[BaselineStats], sensor: str) -> Optional[float]:
    """z = (x - mean) / max(std, sigma_floor).  Returns None if stats is None."""
    if stats is None:
        return None
    sigma = max(stats.std, SIGMA_FLOOR[sensor])
    return float((x - stats.mean) / sigma)


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile of an iterable of values (q in 0–100)."""
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


def percentile_rank(x, days, values, target_day: date, sensor: str) -> Optional[float]:
    """Percentile rank of x in the pooled baseline window (mid-rank for ties).

    Returns None if the baseline is insufficient.
    """
    selected_days, selected_values = _baseline_candidates(days, values, target_day, sensor)
    if not selected_values:
        return None
    x = float(x)
    count_less = sum(1 for v in selected_values if v < x)
    count_equal = sum(
        1 for v in selected_values if math.isclose(v, x, rel_tol=1e-12, abs_tol=1e-12)
    )
    pct = 100.0 * (count_less + 0.5 * count_equal) / len(selected_values)
    return min(100.0, max(0.0, pct))


def to_stress(sensor: str, z: float) -> float:
    """Convert a raw z-score to the stress scale: stress = SENSOR_SIGN * z."""
    return float(SENSOR_SIGN[sensor] * z)


def forecast_stress_interval(fc_or_p10=None, p50=None, p90=None):
    """Convert ndvi_z quantiles to stress-scale quantiles (D1, frozen).

    Two calling conventions are accepted:
      1. forecast_stress_interval(Forecast_obj_or_dict)
         — P3/agent style; reads ndvi_z_p10/p50/p90 from fields/keys.
      2. forecast_stress_interval(p10, p50, p90)
         — P2/ml style; three scalar (or None) positional arguments.

    Returns (stress_p10, stress_p50, stress_p90) where:
        stress_p10 = -ndvi_z_p90   (quantiles invert when sign flips)
        stress_p50 = -ndvi_z_p50
        stress_p90 = -ndvi_z_p10
    Any None value propagates as None in the corresponding output slot.
    """
    # Detect calling convention: if first arg is a dict/object (not scalar/None)
    if fc_or_p10 is None and p50 is None and p90 is None:
        return (None, None, None)

    if isinstance(fc_or_p10, dict):
        p10_val = fc_or_p10.get("ndvi_z_p10")
        p50_val = fc_or_p10.get("ndvi_z_p50")
        p90_val = fc_or_p10.get("ndvi_z_p90")
    elif p50 is None and p90 is None and not isinstance(fc_or_p10, (int, float)):
        # Object with attributes (Forecast model instance)
        p10_val = getattr(fc_or_p10, "ndvi_z_p10", None)
        p50_val = getattr(fc_or_p10, "ndvi_z_p50", None)
        p90_val = getattr(fc_or_p10, "ndvi_z_p90", None)
    else:
        # Three scalar args: forecast_stress_interval(p10, p50, p90)
        p10_val, p50_val, p90_val = fc_or_p10, p50, p90

    def _neg(v):
        return None if v is None else -float(v)

    return (_neg(p90_val), _neg(p50_val), _neg(p10_val))


def modis_end_date(start: date) -> date:
    """MODIS 16-day composite end date: start + 15 days (D5, frozen)."""
    return start + timedelta(days=MODIS_COMPOSITE_DAYS - 1)


def modis_usable(start: date, as_of: date) -> bool:
    """True if the composite's end date <= as_of (no look-ahead, D5)."""
    return modis_end_date(start) <= as_of


# ---------------------------------------------------------------------------
# Compatibility API — pd.Series-based (used by ml/ training pipeline, P2)
# ---------------------------------------------------------------------------


def _doy(d: date) -> int:
    """1-based day-of-year."""
    return d.timetuple().tm_yday


def _in_window(d: date, target_date: date, half: int = BASELINE_HALF_WINDOW_DAYS) -> bool:
    """True if *d* is within ±*half* days (circular DoY) of *target_date*."""
    return circ_doy_dist(d, target_date) <= half


def baseline_window_dates(target_date: date, series: pd.Series) -> pd.Series:
    """Return the subset of *series* in the ±10-day circular DoY window,
    strictly before ``target_date.year``.

    Sensor start-year filtering is NOT applied here — pass the already-filtered
    series in, or use ``_pool`` / ``baseline_stats`` which apply it.
    """
    if series.empty:
        return series.iloc[0:0]
    idx = pd.to_datetime(series.index).date
    mask = np.array(
        [d.year < target_date.year and _in_window(d, target_date) for d in idx],
        dtype=bool,
    )
    return series.iloc[mask]


def _pool(series: pd.Series, target_date: date, sensor: str):
    """Pool non-NaN observations from the ±10-day window in valid baseline years.

    Returns (np.ndarray of values, n_years: int).
    """
    grp = baseline_group(sensor)
    start_year = BASELINE_START_YEAR[grp]
    if series.empty:
        return np.array([]), 0
    idx_dates = pd.to_datetime(series.index).date
    mask = np.array(
        [
            start_year <= d.year < target_date.year and _in_window(d, target_date)
            for d in idx_dates
        ],
        dtype=bool,
    )
    windowed = series.iloc[mask].dropna()
    if windowed.empty:
        return np.array([]), 0
    values = windowed.values.astype(float)
    years = {pd.to_datetime(d).year for d in windowed.index}
    return values, len(years)


def compute_z(
    value: float,
    series: pd.Series,
    target_date: date,
    sensor: str,
) -> tuple[Optional[float], int, int]:
    """z-score of *value* vs the pooled pd.Series baseline.

    Returns (z | None, n_years, n_obs).  z is None when data is insufficient.
    Uses the same rules as ``baseline_stats`` (min years, min obs, sigma floor).
    """
    grp = baseline_group(sensor)
    min_yrs = MIN_BASELINE_YEARS[grp]
    sigma_floor = SIGMA_FLOOR[sensor]

    values, n_years = _pool(series, target_date, sensor)
    n_obs = len(values)
    if n_years < min_yrs or n_obs < MIN_BASELINE_OBS:
        return None, n_years, n_obs

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n_obs > 1 else 0.0
    std = max(std, sigma_floor)
    return float((value - mean) / std), n_years, n_obs


def compute_percentile(
    value: float,
    series: pd.Series,
    target_date: date,
    sensor: str,
) -> Optional[float]:
    """Percentile rank (0–100) of *value* in the pooled pd.Series baseline.

    Returns None if the baseline is insufficient (same thresholds as compute_z).
    """
    grp = baseline_group(sensor)
    min_yrs = MIN_BASELINE_YEARS[grp]

    values, n_years = _pool(series, target_date, sensor)
    n_obs = len(values)
    if n_years < min_yrs or n_obs < MIN_BASELINE_OBS:
        return None
    return float(np.mean(values < value) * 100)


def rain_normal_14d(daily_precip: pd.Series, target_date: date) -> Optional[float]:
    """Mean of 14-day precipitation totals ending on *target_date*'s DoY
    across all prior baseline years (2010 … target_date.year-1).

    A year is skipped if its 14-day block has fewer than 14 rows or any NaN.
    Returns None when no valid years exist.
    """
    if daily_precip.empty:
        return None
    s = daily_precip.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    start_yr = BASELINE_START_YEAR["weather"]
    totals: list[float] = []
    for yr in range(start_yr, target_date.year):
        try:
            end_d = dt.date(yr, target_date.month, target_date.day)
        except ValueError:
            continue  # target_date is Feb 29 and yr is not a leap year
        start_d = end_d - dt.timedelta(days=13)
        block = s.loc[str(start_d): str(end_d)]
        if len(block) < 14 or block.isna().any():
            continue
        totals.append(float(block.sum()))
    return float(np.mean(totals)) if totals else None


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Primary (list-based) API
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
    # Compatibility (Series-based) API
    "baseline_window_dates",
    "compute_z",
    "compute_percentile",
    "rain_normal_14d",
]
