"""
common/features.py — Feature engineering for the BreakWatch ML pipeline.

Pure functions only: no network, no system clock, no global state.
All thresholds and baseline constants come from config.py.
Z-scores and percentiles delegate to common/baseline.py — no duplication.

Feature name contract
---------------------
``feature_names()`` and the keys returned by ``build_features()`` are kept in
lock-step via FEATURE_NAMES.  Training and serving both call ``feature_names()``;
any rename here requires a model retrain.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from common.baseline import (
    baseline_stats,
    modis_usable,
    percentile_rank,
    zscore,
)
from config import (
    BASELINE_START_YEAR,
    SM_DAILY_MIN_HOURS,
    STALE_DAYS,
)

# Days after which an NDVI composite is considered too stale to use.
# Separate from config.STALE_DAYS (which is for sensor obs, not NDVI features).
STALE_NDVI_DAYS: int = 20

# Minimum present rows for windowed aggregates — never fill with a guess.
_ET0_MIN_DAYS: int = 12   # out of 14
_TMAX_MIN_DAYS: int = 6   # out of 7

# Ordered feature keys — single source of truth.
FEATURE_NAMES: tuple[str, ...] = (
    "rain_deficit_14d_mm",
    "days_since_rain_gt_2p5mm",
    "sm_shallow_pctl",
    "sm_root_pctl",
    "ndvi_z",
    "ndvi_z_slope_3comp",
    "et0_sum_14d",
    "tmax_mean_7d",
    "doy_sin",
    "doy_cos",
)


# ---------------------------------------------------------------------------
# 1. Hourly → daily aggregation
# ---------------------------------------------------------------------------


def aggregate_hourly_to_daily(
    hourly,
    how: str = "mean",
    tz: str = "Asia/Kolkata",
) -> pd.Series | pd.DataFrame:
    """Aggregate a timestamp-indexed Series or DataFrame to IST calendar days.

    Parameters
    ----------
    hourly:
        ``pd.Series`` or ``pd.DataFrame`` indexed by timezone-aware or naive
        UTC timestamps.  Naive timestamps are assumed UTC and converted to IST.
    how:
        ``"mean"`` (default) or ``"sum"``.
    tz:
        Target timezone for grouping; default ``"Asia/Kolkata"`` (IST, UTC+5:30).

    Returns
    -------
    Same type as input (Series or DataFrame), indexed by ``datetime.date``.
    A day with **fewer than** ``SM_DAILY_MIN_HOURS`` (18) non-NaN values becomes
    ``NaN`` — the threshold is applied per column for DataFrames.

    Notes
    -----
    The IST day boundary is at UTC+5:30, so midnight IST = 18:30 UTC the
    previous day.  An hour that falls on a different IST date than the UTC date
    is correctly assigned to the IST date.
    """
    if hourly is None:
        raise TypeError("hourly must be a pandas Series or DataFrame, not None")
    if hourly.empty:
        return hourly.iloc[0:0]

    is_series = isinstance(hourly, pd.Series)

    # ── Normalise to IST ────────────────────────────────────────────────────
    idx = pd.to_datetime(hourly.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ist_idx = idx.tz_convert(tz)
    # Group key: IST calendar date (no tz info needed for date)
    ist_dates = ist_idx.normalize().tz_localize(None)  # midnight IST, tz-naive

    if is_series:
        df = hourly.copy()
        df.index = ist_dates
        if how == "sum":
            daily = df.groupby(df.index).sum()
        else:
            daily = df.groupby(df.index).mean()
        # Count non-NaN per day
        counts = hourly.copy()
        counts.index = ist_dates
        valid_counts = counts.notna().groupby(counts.index).sum()
        daily[valid_counts < SM_DAILY_MIN_HOURS] = float("nan")
        daily.index = pd.to_datetime(daily.index).date
        return daily
    else:
        df = hourly.copy()
        df.index = ist_dates
        valid_counts = df.notna().groupby(df.index).sum()
        if how == "sum":
            daily = df.groupby(df.index).sum()
        else:
            daily = df.groupby(df.index).mean()
        for col in daily.columns:
            mask = valid_counts[col] < SM_DAILY_MIN_HOURS
            daily.loc[mask, col] = float("nan")
        daily.index = pd.to_datetime(daily.index).date
        return daily


# ---------------------------------------------------------------------------
# 2. Dry-spell helpers (public; also imported by ml/triggers.py)
# ---------------------------------------------------------------------------


def consecutive_dry_days(
    daily_precip: pd.Series,
    as_of: date,
    threshold_mm: float = 2.5,
) -> tuple[int, Optional[date], Optional[date]]:
    """Count consecutive days with precipitation below *threshold_mm*, ending on *as_of*.

    Walks backwards from *as_of* using only rows with ``day <= as_of``.

    A day ends the run if:
    * ``precip >= threshold_mm`` — genuine rain event.
    * ``precip`` is NaN / the date is absent from the series — missing data.
      In this case the run length up to that point is returned, and the
      caller can detect missing-data termination by checking whether
      ``run_start`` is the day after the gap.

    Parameters
    ----------
    daily_precip:
        ``pd.Series`` indexed by ``datetime.date`` (or parseable strings),
        values are precipitation in mm.  Only rows with index <= *as_of* are used.
    as_of:
        The reference date (inclusive end of the run).
    threshold_mm:
        Days with ``precip >= threshold_mm`` break the run (default 2.5 mm).

    Returns
    -------
    (run_length, run_start, run_end)
        * ``run_length``: number of consecutive sub-threshold days (0 if *as_of*
          itself has rain or is missing).
        * ``run_start``: first date of the dry run (earliest), or ``None`` if
          ``run_length == 0``.
        * ``run_end``: last date of the dry run (*as_of* if ``run_length > 0``),
          or ``None``.
    """
    # Build a dict of date → precip for O(1) lookup
    s = daily_precip.copy()
    idx_dates = pd.to_datetime(s.index).date
    # Only use days on or before as_of
    day_map: dict[date, float] = {}
    for d, v in zip(idx_dates, s.values):
        if d <= as_of:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            day_map[d] = fv

    count = 0
    cursor = as_of
    while True:
        if cursor not in day_map:
            # Missing day — ends the run
            break
        v = day_map[cursor]
        if math.isnan(v) or v >= threshold_mm:
            # Rain day or explicit NaN — ends the run
            break
        count += 1
        cursor -= timedelta(days=1)

    if count == 0:
        return 0, None, None
    run_end = as_of
    run_start = as_of - timedelta(days=count - 1)
    return count, run_start, run_end


def days_since_rain_gt_2p5mm(
    daily_precip: pd.Series,
    as_of: date,
    lookback: int = 60,
) -> int:
    """Count days since the last day with precip > 2.5 mm, capped at *lookback*.

    Walks backwards from *as_of*.  A missing or absent day also stops the walk
    (conservative: treat unknown data as a boundary, not as dry).

    Cap: if no qualifying rain day is found within *lookback* days, returns
    *lookback*.  This prevents unbounded integers in feature vectors.

    Parameters
    ----------
    daily_precip:
        ``pd.Series`` indexed by ``datetime.date``, values in mm.
    as_of:
        Reference date (inclusive).
    lookback:
        Maximum days to search backwards; returned value is capped here.

    Returns
    -------
    int
        Days since last rain > 2.5 mm, in [0, lookback].
    """
    s = daily_precip.copy()
    idx_dates = pd.to_datetime(s.index).date
    day_map: dict[date, float] = {}
    for d, v in zip(idx_dates, s.values):
        if d <= as_of:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            day_map[d] = fv

    count = 0
    cursor = as_of
    while count < lookback:
        if cursor not in day_map:
            break
        v = day_map[cursor]
        if not math.isnan(v) and v > 2.5:
            break
        count += 1
        cursor -= timedelta(days=1)
    return min(count, lookback)


# ---------------------------------------------------------------------------
# 3. Internal helpers for build_features
# ---------------------------------------------------------------------------


def _to_date(val) -> Optional[date]:
    """Coerce val to datetime.date; return None on failure.

    ``datetime.date`` is a superclass of ``datetime.datetime``.  We want to
    accept a bare ``date`` but pass timestamps through ``pd.to_datetime`` so
    that the ``.date()`` method always strips the time component correctly.
    """
    if val is None:
        return None
    # pd.Timestamp is a subclass of datetime.datetime (which is a subclass of date),
    # so check for the exact date type to avoid treating timestamps as bare dates.
    if type(val) is date:  # noqa: E721 — intentional exact-type check
        return val
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _rain_normal_from_weather(weather: pd.DataFrame, as_of: date) -> Optional[float]:
    """14-day rain normal using the weather DataFrame (2010-baseline years).

    Re-implements the logic from common/baseline.rain_normal_14d adapted for
    the columnar weather DataFrame format used in build_features.
    """
    if weather.empty or "precip_mm" not in weather.columns:
        return None
    start_yr = BASELINE_START_YEAR["weather"]
    totals: list[float] = []
    for yr in range(start_yr, as_of.year):
        try:
            end_d = date(yr, as_of.month, as_of.day)
        except ValueError:
            continue  # Feb 29 in non-leap year
        start_d = end_d - timedelta(days=13)
        block = weather[(weather["day"] >= start_d) & (weather["day"] <= end_d)]
        if len(block) < 14 or block["precip_mm"].isna().any():
            continue
        totals.append(float(block["precip_mm"].sum()))
    return float(np.mean(totals)) if totals else None


def _observed_rain_14d(weather: pd.DataFrame, as_of: date) -> float:
    """Observed 14-day precipitation sum ending on as_of (treat NaN as 0)."""
    start_d = as_of - timedelta(days=13)
    block = weather[(weather["day"] >= start_d) & (weather["day"] <= as_of)]
    return float(block["precip_mm"].fillna(0).sum())


def _usable_modis(modis_ndvi: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return MODIS rows whose end_date <= as_of, sorted by end_date ascending."""
    df = modis_ndvi.copy()
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    # D5: composite usable only if end_date <= as_of
    df = df[df["end_date"] <= as_of].sort_values("end_date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4. build_features — the main feature vector for Model A
# ---------------------------------------------------------------------------


def build_features(
    weather_daily: pd.DataFrame,
    modis_ndvi: pd.DataFrame,
    as_of: date,
) -> dict[str, Optional[float]]:
    # Guard: allow old-style call build_features(as_of, weather, modis) where
    # the first argument is a date and the last is a DataFrame.
    # This keeps backward compatibility with P3's existing test calls.
    if isinstance(weather_daily, date) and isinstance(modis_ndvi, pd.DataFrame):
        # Signature was: build_features(as_of, weather, modis)
        weather_daily, modis_ndvi, as_of = modis_ndvi, as_of, weather_daily  # type: ignore[assignment]
    """Build the Model A feature dict for a single *as_of* date.

    All inputs are treated as the full historical record; this function
    enforces the look-ahead guard itself (rows after *as_of* are silently
    ignored — changing them does not affect the output).

    Parameters
    ----------
    weather_daily:
        DataFrame with columns: ``day``, ``precip_mm``, ``et0_mm``, ``tmax_c``,
        ``sm_shallow``, ``sm_root``.  Full history 2010-onward.
    modis_ndvi:
        DataFrame with columns: ``start_date``, ``end_date``, ``ndvi``.
        Baselines are aligned by *end_date* (D5, frozen).
    as_of:
        Reference date.  Only data with day/end_date <= as_of is used.

    Returns
    -------
    dict matching ``feature_names()`` exactly.  Any feature whose inputs are
    insufficient is ``None``; features are **never** filled with a default or guess.
    """
    as_of = _to_date(as_of) or as_of

    # ── Weather: enforce look-ahead guard ───────────────────────────────────
    wx = pd.DataFrame() if weather_daily is None else weather_daily.copy()
    if not wx.empty:
        wx["day"] = pd.to_datetime(wx["day"]).dt.date
        wx = wx[wx["day"] <= as_of].sort_values("day").reset_index(drop=True)

    # Build a precip Series for the dry-spell helpers
    precip_series: pd.Series = (
        wx.set_index("day")["precip_mm"] if not wx.empty and "precip_mm" in wx.columns
        else pd.Series(dtype=float)
    )

    # ── MODIS: enforce look-ahead guard (end_date <= as_of) ─────────────────
    mdv = pd.DataFrame() if modis_ndvi is None else modis_ndvi.copy()
    # Normalise: if MODIS has a 'day' column instead of start_date/end_date
    # (backward-compat with P3's older format), treat 'day' as end_date and
    # derive start_date as end_date - 15 days (MODIS_COMPOSITE_DAYS - 1).
    if not mdv.empty and "day" in mdv.columns and "end_date" not in mdv.columns:
        mdv["end_date"] = pd.to_datetime(mdv["day"]).dt.date
        mdv["start_date"] = mdv["end_date"].apply(lambda d: d - timedelta(days=15))
    mdv = _usable_modis(mdv, as_of) if not mdv.empty else mdv

    # ── Feature 1: rain_deficit_14d_mm ──────────────────────────────────────
    rain_normal = _rain_normal_from_weather(wx, as_of)
    if rain_normal is not None:
        rain_deficit_14d_mm: Optional[float] = rain_normal - _observed_rain_14d(wx, as_of)
    else:
        rain_deficit_14d_mm = None

    # ── Feature 2: days_since_rain_gt_2p5mm ─────────────────────────────────
    dsrg: Optional[int] = (
        days_since_rain_gt_2p5mm(precip_series, as_of)
        if not precip_series.empty else None
    )

    # ── Features 3 & 4: sm percentiles ──────────────────────────────────────
    sm_shallow_pctl: Optional[float] = None
    sm_root_pctl: Optional[float] = None
    if not wx.empty:
        # Use the most recent on-or-before as_of value for each SM column
        for col, feat_var in [("sm_shallow", "sm_shallow_pctl"), ("sm_root", "sm_root_pctl")]:
            if col not in wx.columns:
                continue
            col_series = wx.set_index("day")[col].dropna()
            if col_series.empty:
                continue
            latest_val = float(col_series.iloc[-1])
            days_list = list(col_series.index)
            vals_list = list(col_series.values.astype(float))
            pct = percentile_rank(latest_val, days_list, vals_list, as_of, "soil_moist")
            if feat_var == "sm_shallow_pctl":
                sm_shallow_pctl = pct
            else:
                sm_root_pctl = pct

    # ── Features 5 & 6: ndvi_z and ndvi_z_slope_3comp ───────────────────────
    ndvi_z: Optional[float] = None
    ndvi_z_slope_3comp: Optional[float] = None

    if not mdv.empty:
        # Latest usable composite
        latest = mdv.iloc[-1]
        latest_end: date = latest["end_date"]
        latest_val_ndvi: float = float(latest["ndvi"])

        # Staleness check: composite end_date must be within STALE_NDVI_DAYS
        if (as_of - latest_end).days <= STALE_NDVI_DAYS:
            stats = baseline_stats(
                list(mdv["end_date"]),
                list(mdv["ndvi"].astype(float)),
                latest_end,
                "modis_ndvi",
            )
            ndvi_z = zscore(latest_val_ndvi, stats, "modis_ndvi")

        # Slope across last 3 composites — need exactly 3 usable z-scores
        tail3 = mdv.tail(3)
        z_scores: list[float] = []
        for _, row in tail3.iterrows():
            end_d: date = row["end_date"]
            val: float = float(row["ndvi"])
            st = baseline_stats(
                list(mdv["end_date"]),
                list(mdv["ndvi"].astype(float)),
                end_d,
                "modis_ndvi",
            )
            z_i = zscore(val, st, "modis_ndvi")
            if z_i is not None:
                z_scores.append(z_i)
        if len(z_scores) == 3:
            # Least-squares slope per composite step (x = 0, 1, 2)
            xs = np.array([0.0, 1.0, 2.0])
            ys = np.array(z_scores)
            ndvi_z_slope_3comp = float(np.polyfit(xs, ys, 1)[0])

    # ── Feature 7: et0_sum_14d ───────────────────────────────────────────────
    et0_sum_14d: Optional[float] = None
    if not wx.empty and "et0_mm" in wx.columns:
        start_d = as_of - timedelta(days=13)
        et0_block = wx[(wx["day"] >= start_d) & (wx["day"] <= as_of)]["et0_mm"]
        n_present = int(et0_block.notna().sum())
        if n_present >= _ET0_MIN_DAYS:
            et0_sum_14d = float(et0_block.sum(min_count=1))

    # ── Feature 8: tmax_mean_7d ──────────────────────────────────────────────
    tmax_mean_7d: Optional[float] = None
    if not wx.empty and "tmax_c" in wx.columns:
        start_d7 = as_of - timedelta(days=6)
        tmax_block = wx[(wx["day"] >= start_d7) & (wx["day"] <= as_of)]["tmax_c"]
        n_present7 = int(tmax_block.notna().sum())
        if n_present7 >= _TMAX_MIN_DAYS:
            tmax_mean_7d = float(tmax_block.mean())

    # ── Features 9 & 10: doy_sin, doy_cos ───────────────────────────────────
    doy_angle = 2.0 * math.pi * (as_of.timetuple().tm_yday / 365.0)

    return {
        "rain_deficit_14d_mm": rain_deficit_14d_mm,
        "days_since_rain_gt_2p5mm": dsrg,
        "sm_shallow_pctl": sm_shallow_pctl,
        "sm_root_pctl": sm_root_pctl,
        "ndvi_z": ndvi_z,
        "ndvi_z_slope_3comp": ndvi_z_slope_3comp,
        "et0_sum_14d": et0_sum_14d,
        "tmax_mean_7d": tmax_mean_7d,
        "doy_sin": float(math.sin(doy_angle)),
        "doy_cos": float(math.cos(doy_angle)),
    }


# ---------------------------------------------------------------------------
# 5. feature_names() — single source of truth for training and serving
# ---------------------------------------------------------------------------


def feature_names() -> list[str]:
    """Return the ordered list of feature keys produced by ``build_features``.

    Both the training pipeline and the serving path call this function so that
    column order is never assumed from position.
    """
    return list(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 6. climatology_features — helper for P1's ObsBundle.climatology
# ---------------------------------------------------------------------------


def climatology_features(weather_df: pd.DataFrame, as_of) -> dict:
    """Build the Climatology field dict from the weather history.

    Used by P1 to populate ``ObsBundle.climatology`` without reimplementing
    the baseline logic.  Returns ``None`` for any field that cannot be computed.
    """
    as_of = _to_date(as_of) or pd.to_datetime(as_of).date()
    wx = pd.DataFrame() if weather_df is None else weather_df.copy()
    if not wx.empty:
        wx["day"] = pd.to_datetime(wx["day"]).dt.date
        wx = wx[wx["day"] <= as_of].sort_values("day").reset_index(drop=True)

    if wx.empty:
        return {
            "sm_root_percentile": None,
            "sm_shallow_percentile": None,
            "rain_deficit_14d_mm": None,
            "days_since_rain_gt_2p5mm": None,
            "n_years": 0,
            "baseline_years": [],
            "evidence_ids": [],
        }

    sm_root_pctl = None
    sm_shallow_pctl = None
    for col, key in [("sm_root", "sm_root_pctl"), ("sm_shallow", "sm_shallow_pctl")]:
        if col not in wx.columns:
            continue
        col_s = wx.set_index("day")[col].dropna()
        if col_s.empty:
            continue
        pct = percentile_rank(
            float(col_s.iloc[-1]),
            list(col_s.index),
            list(col_s.values.astype(float)),
            as_of,
            "soil_moist",
        )
        if key == "sm_root_pctl":
            sm_root_pctl = pct
        else:
            sm_shallow_pctl = pct

    precip_s = (
        wx.set_index("day")["precip_mm"]
        if "precip_mm" in wx.columns
        else pd.Series(dtype=float)
    )
    rain_normal = _rain_normal_from_weather(wx, as_of)
    rain_deficit = (
        (rain_normal - _observed_rain_14d(wx, as_of)) if rain_normal is not None else None
    )
    days_since = days_since_rain_gt_2p5mm(precip_s, as_of) if not precip_s.empty else None
    years = sorted({d.year for d in wx["day"].tolist() if d.year < as_of.year})

    return {
        "sm_root_percentile": sm_root_pctl,
        "sm_shallow_percentile": sm_shallow_pctl,
        "rain_deficit_14d_mm": rain_deficit,
        "days_since_rain_gt_2p5mm": days_since,
        "n_years": len(years),
        "baseline_years": years,
        "evidence_ids": [],
    }


# ---------------------------------------------------------------------------
# Aliases for backward compat (P3's existing imports)
# ---------------------------------------------------------------------------

hourly_to_daily = aggregate_hourly_to_daily

__all__ = [
    "FEATURE_NAMES",
    "STALE_NDVI_DAYS",
    "aggregate_hourly_to_daily",
    "hourly_to_daily",
    "consecutive_dry_days",
    "days_since_rain_gt_2p5mm",
    "build_features",
    "feature_names",
    "climatology_features",
]
