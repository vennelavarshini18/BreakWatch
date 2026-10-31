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
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BASELINE_HALF_WINDOW_DAYS,
    BASELINE_START_YEAR,
    MIN_BASELINE_OBS,
    MIN_BASELINE_YEARS,
    SIGMA_FLOOR,
    baseline_group,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DOY_YEAR = 365  # fixed-length year for circular arithmetic


def _doy(date: dt.date) -> int:
    """Return 1-based day-of-year (1–366 in leap years, 1–365 otherwise)."""
    return date.timetuple().tm_yday


def _circular_distance(doy_a: int, doy_b: int) -> int:
    """Absolute circular distance between two DoYs on a 365-day circle."""
    diff = abs(doy_a - doy_b) % _DOY_YEAR
    return min(diff, _DOY_YEAR - diff)


def _in_window(date: dt.date, target_date: dt.date, half: int = BASELINE_HALF_WINDOW_DAYS) -> bool:
    """True if *date* falls within ±*half* days (circular DoY) of *target_date*."""
    return _circular_distance(_doy(date), _doy(target_date)) <= half


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def baseline_window_dates(
    target_date: dt.date,
    series: pd.Series,
) -> pd.Series:
    """Return the subset of *series* whose index dates fall in the ±10-day
    circular DoY window strictly before ``target_date.year``.

    Parameters
    ----------
    target_date:
        The date we are computing anomalies for.  Only years < target_date.year
        are used (no strict start-year filter here — callers apply that via the
        full series passed in).
    series:
        A ``pd.Series`` with a ``DatetimeIndex`` or ``Index`` of ``datetime.date``
        values.  Values may be float with NaN for missing.

    Returns
    -------
    pd.Series
        Filtered subset (may be empty).  The caller decides what to do with it.
    """
    if series.empty:
        return series.iloc[0:0]

    # Normalise index to datetime.date objects for reliable comparison
    idx = pd.to_datetime(series.index).date

    mask = np.array(
        [
            d.year < target_date.year and _in_window(d, target_date)
            for d in idx
        ],
        dtype=bool,
    )
    return series.iloc[mask]


def _pool(
    series: pd.Series,
    target_date: dt.date,
    sensor: str,
) -> tuple[np.ndarray, int]:
    """Pool observations from the ±10-day window across valid baseline years.

    Returns
    -------
    values : np.ndarray
        Non-NaN pooled values.
    n_years : int
        Number of distinct calendar years represented in *values*.
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
    windowed = series.iloc[mask]
    valid = windowed.dropna()

    if valid.empty:
        return np.array([]), 0

    values = valid.values.astype(float)
    years = {pd.to_datetime(d).year for d in valid.index}
    return values, len(years)


def compute_z(
    value: float,
    series: pd.Series,
    target_date: dt.date,
    sensor: str,
) -> tuple[Optional[float], int, int]:
    """Compute a baseline z-score for *value* against historical *series*.

    Uses the pooled ±10-day circular DoY window across years strictly before
    ``target_date.year`` (and not before the sensor's start year).

    Parameters
    ----------
    value:
        The observed scalar to score.
    series:
        Historical time series, ``pd.Series`` indexed by date.
    target_date:
        The observation date (determines year filter and DoY window).
    sensor:
        One of the ``Sensor`` literals from contracts.py (e.g. ``"modis_ndvi"``).

    Returns
    -------
    (z, n_years, n_obs)
        *z* is ``None`` when there is insufficient baseline data.
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

    z = (value - mean) / std
    return float(z), n_years, n_obs


def compute_percentile(
    value: float,
    series: pd.Series,
    target_date: dt.date,
    sensor: str,
) -> Optional[float]:
    """Return the percentile rank (0–100) of *value* within the pooled baseline.

    Uses the same window, year filter, and minimum thresholds as ``compute_z``.
    Returns ``None`` if the baseline is insufficient.
    """
    grp = baseline_group(sensor)
    min_yrs = MIN_BASELINE_YEARS[grp]

    values, n_years = _pool(series, target_date, sensor)
    n_obs = len(values)

    if n_years < min_yrs or n_obs < MIN_BASELINE_OBS:
        return None

    # Fraction of pooled values strictly less than *value* → percentile 0–100
    pct = float(np.mean(values < value) * 100)
    return pct


def rain_normal_14d(
    daily_precip: pd.Series,
    target_date: dt.date,
) -> Optional[float]:
    """Mean of the 14-day rolling precipitation total ending on *target_date*'s
    day-of-year, computed across years strictly before ``target_date.year``.

    Parameters
    ----------
    daily_precip:
        ``pd.Series`` of daily precipitation (mm), indexed by date.  Missing
        days should be NaN — any year-window with NaN in its 14-day block is
        excluded from the mean.
    target_date:
        Determines the calendar day-of-year end of the window and the year
        filter.

    Returns
    -------
    float or None
        Mean 14-day total in mm, or ``None`` if no valid years exist.

    Notes
    -----
    The 14-day window ends on the same day-of-year as *target_date* (inclusive)
    and starts 13 days earlier.  This matches the ``rain_deficit_14d_mm`` field
    in ``Climatology`` which is ``normal - observed``.
    """
    if daily_precip.empty:
        return None

    # Work with a sorted DatetimeIndex for efficient slicing
    s = daily_precip.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    target_doy = _doy(target_date)
    # Build a 14-day DOY range (not circular — this is a contiguous window ending at target_doy)
    window_doys = set(range(target_doy - 13, target_doy + 1))
    start_yr = BASELINE_START_YEAR["weather"]

    totals: list[float] = []
    years_seen = range(start_yr, target_date.year)

    for yr in years_seen:
        # Slice the 14-day block for this year
        try:
            end_d = dt.date(yr, target_date.month, target_date.day)
        except ValueError:
            # target_date is Feb 29 and yr is not a leap year — skip
            continue
        start_d = end_d - dt.timedelta(days=13)

        block = s.loc[str(start_d) : str(end_d)]
        if len(block) < 14 or block.isna().any():
            continue
        totals.append(float(block.sum()))

    if not totals:
        return None
    return float(np.mean(totals))


def forecast_stress_interval(
    ndvi_z_p10: Optional[float],
    ndvi_z_p50: Optional[float],
    ndvi_z_p90: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Convert ndvi_z quantiles to stress-scale quantiles.

    Decision D1 (frozen):
        stress = -1 * ndvi_z   (SENSOR_SIGN for ndvi = -1)
        Quantiles invert:
            stress_p10 = -ndvi_z_p90   (best stress = worst ndvi)
            stress_p50 = -ndvi_z_p50
            stress_p90 = -ndvi_z_p10   (worst stress = best ndvi)

    Any None input propagates as None in the corresponding output position.
    """
    def _neg(v: Optional[float]) -> Optional[float]:
        return None if v is None else -v

    stress_p10 = _neg(ndvi_z_p90)
    stress_p50 = _neg(ndvi_z_p50)
    stress_p90 = _neg(ndvi_z_p10)
    return stress_p10, stress_p50, stress_p90
