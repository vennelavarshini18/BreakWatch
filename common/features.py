from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from common.baseline import baseline_stats, percentile_rank, zscore
from config import BASELINE_START_YEAR, SM_DAILY_MIN_HOURS

FEATURE_NAMES = (
    "rain_deficit_14d_mm",
    "days_since_rain_gt_2p5mm",
    "sm_shallow_pctl",
    "sm_root_pctl",
    "ndvi_z",
    "ndvi_z_slope3",
    "et0_14d_mm",
    "tmax_7d_mean_c",
    "doy_sin",
    "doy_cos",
)


def aggregate_hourly_to_daily(df_hourly, value_cols, tz="Asia/Kolkata"):
    """Daily mean by IST day, NaN where fewer than configured valid hours are present."""
    if df_hourly is None or df_hourly.empty:
        return pd.DataFrame()
    df = df_hourly.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(tz).dt.tz_localize(None)
        else:
            ts = ts.dt.tz_localize(tz).dt.tz_localize(None)
        df["day"] = ts
    elif "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    else:
        return pd.DataFrame()
    df = df.dropna(subset=["day"]).copy()
    valid_hours = df[value_cols].notna().groupby(df["day"]).sum()
    daily = df.groupby("day")[value_cols].mean()
    for col in value_cols:
        if col in daily.columns:
            daily.loc[valid_hours[col] < SM_DAILY_MIN_HOURS, col] = np.nan
    daily = daily.reset_index().rename(columns={"day": "day"})
    daily["day"] = pd.to_datetime(daily["day"]).dt.date
    return daily


def _as_date_list(values):
    out = []
    for value in values:
        if value is None:
            continue
        try:
            parsed = pd.to_datetime(value).date()
        except Exception:
            continue
        out.append(parsed)
    return out


def _fifteen_day_window(as_of: date):
    start = as_of - timedelta(days=13)
    return [start + timedelta(days=i) for i in range(14)]


def _rain_deficit_14d_mm(weather_df, as_of: date) -> float | None:
    if weather_df is None or weather_df.empty:
        return None
    weather = weather_df.copy()
    weather["day"] = pd.to_datetime(weather["day"]).dt.date
    weather = weather[(weather["day"] <= as_of)].sort_values("day")
    window_dates = _fifteen_day_window(as_of)
    obs_total = 0.0
    for d in window_dates:
        row = weather[weather["day"] == d]
        if row.empty:
            continue
        value = pd.to_numeric(row["precip_mm"], errors="coerce").iloc[0]
        if pd.notna(value):
            obs_total += float(value)
    prior_years: list[float] = []
    start_year = BASELINE_START_YEAR["weather"]
    for year in range(start_year, as_of.year):
        year_totals = 0.0
        matched_days = 0
        for d in window_dates:
            try:
                match_day = date(year, d.month, d.day)
            except ValueError:
                continue
            row = weather[weather["day"] == match_day]
            if row.empty:
                continue
            value = pd.to_numeric(row["precip_mm"], errors="coerce").iloc[0]
            if pd.notna(value):
                year_totals += float(value)
                matched_days += 1
        if matched_days > 0:
            prior_years.append(year_totals)
    if not prior_years:
        return None
    return float(np.mean(prior_years) - obs_total)


def _days_since_rain_gt_2p5mm(weather_df, as_of: date) -> int | None:
    if weather_df is None or weather_df.empty:
        return None
    weather = weather_df.copy()
    weather["day"] = pd.to_datetime(weather["day"]).dt.date
    weather = weather[(weather["day"] <= as_of)].sort_values("day")
    day_map = {
        row["day"]: float(row["precip_mm"]) if pd.notna(row["precip_mm"]) else math.nan
        for _, row in weather.iterrows()
    }
    count = 0
    cursor = as_of
    while True:
        if cursor not in day_map:
            break
        value = day_map[cursor]
        if pd.notna(value) and value > 2.5:
            break
        count += 1
        if cursor <= date(1970, 1, 1):
            break
        cursor = cursor - timedelta(days=1)
    return count


def _latest_value_for_day(weather_df, target_day: date, col: str):
    weather = weather_df.copy()
    weather["day"] = pd.to_datetime(weather["day"]).dt.date
    mask = weather["day"] <= target_day
    subset = weather.loc[mask].sort_values("day")
    if subset.empty:
        return None
    value = pd.to_numeric(subset[col], errors="coerce").iloc[-1]
    return None if pd.isna(value) else float(value)


def _percentile_for_sensor(value: float, days, values, target_day: date, sensor: str) -> float | None:
    return percentile_rank(value, days, values, target_day, sensor)


def build_features(as_of, weather_df, ndvi_df) -> dict[str, float | None]:
    """Build a feature dict suitable for downstream model scoring."""
    as_of = pd.to_datetime(as_of).date()
    weather = weather_df.copy() if weather_df is not None else pd.DataFrame()
    ndvi = ndvi_df.copy() if ndvi_df is not None else pd.DataFrame()
    if not weather.empty:
        weather["day"] = pd.to_datetime(weather["day"]).dt.date
        weather = weather[weather["day"] <= as_of].sort_values("day")
    if not ndvi.empty:
        ndvi["day"] = pd.to_datetime(ndvi["day"]).dt.date
        ndvi = ndvi[ndvi["day"] <= as_of].sort_values("day")

    rain_deficit_14d_mm = _rain_deficit_14d_mm(weather, as_of)
    days_since_rain = _days_since_rain_gt_2p5mm(weather, as_of)

    sm_root_latest = _latest_value_for_day(weather, as_of, "sm_root")
    sm_shallow_latest = _latest_value_for_day(weather, as_of, "sm_shallow")
    sm_root_pctl = None
    sm_shallow_pctl = None
    if sm_root_latest is not None and not weather.empty:
        sm_root_pctl = _percentile_for_sensor(sm_root_latest, weather["day"], weather["sm_root"], as_of, "soil_moist")
    if sm_shallow_latest is not None and not weather.empty:
        sm_shallow_pctl = _percentile_for_sensor(sm_shallow_latest, weather["day"], weather["sm_shallow"], as_of, "soil_moist")

    ndvi_z = None
    ndvi_z_slope3 = None
    if not ndvi.empty:
        latest_row = ndvi.iloc[-1]
        latest_day = latest_row["day"]
        latest_ndvi = float(latest_row["ndvi"])
        stats = baseline_stats(ndvi["day"], ndvi["ndvi"], latest_day, "modis_ndvi")
        ndvi_z = zscore(latest_ndvi, stats, "modis_ndvi") if stats is not None else None
        recent = ndvi[ndvi["day"] <= as_of].tail(3).copy()
        z_scores = []
        for _, row in recent.iterrows():
            day_value = row["day"]
            val = float(row["ndvi"])
            stats_i = baseline_stats(ndvi["day"], ndvi["ndvi"], day_value, "modis_ndvi")
            z_i = zscore(val, stats_i, "modis_ndvi") if stats_i is not None else None
            if z_i is not None:
                z_scores.append(float(z_i))
        if len(z_scores) >= 2:
            xs = np.arange(len(z_scores), dtype=float)
            slope = np.polyfit(xs, z_scores, 1)[0]
            ndvi_z_slope3 = float(slope)

    et0_window = weather[(weather["day"] >= as_of - timedelta(days=13)) & (weather["day"] <= as_of)]
    et0_14d_mm = float(et0_window["et0_mm"].sum()) if not et0_window.empty else None
    tmax_window = weather[(weather["day"] >= as_of - timedelta(days=6)) & (weather["day"] <= as_of)]
    tmax_7d_mean_c = float(tmax_window["tmax_c"].mean()) if not tmax_window.empty else None

    doy_angle = 2 * math.pi * (as_of.timetuple().tm_yday / 365.0)
    features = {
        "rain_deficit_14d_mm": rain_deficit_14d_mm,
        "days_since_rain_gt_2p5mm": days_since_rain,
        "sm_shallow_pctl": sm_shallow_pctl,
        "sm_root_pctl": sm_root_pctl,
        "ndvi_z": ndvi_z,
        "ndvi_z_slope3": ndvi_z_slope3,
        "et0_14d_mm": et0_14d_mm,
        "tmax_7d_mean_c": tmax_7d_mean_c,
        "doy_sin": float(math.sin(doy_angle)),
        "doy_cos": float(math.cos(doy_angle)),
    }
    return features


def climatology_features(weather_df, as_of) -> dict:
    """Build climatology fields consistent with contracts.Climatology."""
    as_of = pd.to_datetime(as_of).date()
    weather = weather_df.copy() if weather_df is not None else pd.DataFrame()
    if not weather.empty:
        weather["day"] = pd.to_datetime(weather["day"]).dt.date
        weather = weather[weather["day"] <= as_of].sort_values("day")

    if weather.empty:
        return {
            "sm_root_percentile": None,
            "sm_shallow_percentile": None,
            "rain_deficit_14d_mm": None,
            "days_since_rain_gt_2p5mm": None,
            "n_years": 0,
            "baseline_years": [],
            "evidence_ids": [],
        }

    sm_root_latest = _latest_value_for_day(weather, as_of, "sm_root")
    sm_shallow_latest = _latest_value_for_day(weather, as_of, "sm_shallow")
    sm_root_pctl = None
    sm_shallow_pctl = None
    if sm_root_latest is not None:
        sm_root_pctl = _percentile_for_sensor(sm_root_latest, weather["day"], weather["sm_root"], as_of, "soil_moist")
    if sm_shallow_latest is not None:
        sm_shallow_pctl = _percentile_for_sensor(sm_shallow_latest, weather["day"], weather["sm_shallow"], as_of, "soil_moist")

    rain_deficit = _rain_deficit_14d_mm(weather, as_of)
    days_since = _days_since_rain_gt_2p5mm(weather, as_of)
    years = sorted({d.year for d in weather["day"].tolist() if d.year < as_of.year})
    return {
        "sm_root_percentile": sm_root_pctl,
        "sm_shallow_percentile": sm_shallow_pctl,
        "rain_deficit_14d_mm": rain_deficit,
        "days_since_rain_gt_2p5mm": days_since,
        "n_years": len(years),
        "baseline_years": years,
        "evidence_ids": [],
    }


hourly_to_daily = aggregate_hourly_to_daily

__all__ = [
    "FEATURE_NAMES",
    "aggregate_hourly_to_daily",
    "hourly_to_daily",
    "build_features",
    "climatology_features",
]
