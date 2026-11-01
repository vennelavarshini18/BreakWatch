"""geo/quality.py - observation quality gate (P1).

Pure functions: no network, no clock. The date comes from the as_of argument.

Rules (thresholds come from config.py):
  * Optical (s2_ndvi, s2_ndmi) observation with valid_frac < OPTICAL_MIN_VALID_FRAC is cloudy -> dropped.
  * If no fresh optical observation is left and a fresh SAR (s1_*) one exists -> event "optical_cloudy->sar".
    If no SAR exists either -> event "optical_cloudy:no_sar_available". Nothing is ever invented.
  * A sensor whose newest observation is older than STALE_DAYS (MODIS: MODIS_STALE_DAYS) is stale.
    If observations exist but none is fresh, GateResult.stale is True (the agent then reports insufficient data).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from config import OPTICAL_MIN_VALID_FRAC, STALE_DAYS

OPTICAL_SENSORS = ("s2_ndvi", "s2_ndmi")
SAR_SENSORS = ("s1_vh", "s1_vv")
MODIS_STALE_DAYS = 20          # mirrors common.features.STALE_NDVI_DAYS (a 16-day composite is slow by nature)


@dataclass
class GateResult:
    obs: list                                   # observations that passed the gate (inputs are not modified)
    events: list[str] = field(default_factory=list)
    stale: bool = False                         # True only when observations exist but none is fresh
    fresh_sensors: list[str] = field(default_factory=list)
    switched_to_sar: bool = False


def _stale_limit(sensor: str) -> int:
    return MODIS_STALE_DAYS if sensor == "modis_ndvi" else STALE_DAYS


def quality_gate(obs, as_of: dt.date) -> GateResult:
    kept: list = []
    events: list[str] = []
    n_cloudy = 0
    for o in obs:
        if o.day > as_of:                       # look-ahead guard; the ObsBundle validator would reject it anyway
            continue
        if o.sensor in OPTICAL_SENSORS and o.valid_frac < OPTICAL_MIN_VALID_FRAC:
            n_cloudy += 1
            continue
        kept.append(o)

    latest: dict[str, dt.date] = {}
    for o in kept:
        if o.sensor not in latest or o.day > latest[o.sensor]:
            latest[o.sensor] = o.day

    fresh: list[str] = []
    for sensor in sorted(latest):
        age = (as_of - latest[sensor]).days
        if age <= _stale_limit(sensor):
            fresh.append(sensor)
        else:
            events.append(f"stale:{sensor}:{age}d")

    if n_cloudy:
        events.append(f"optical_cloudy:{n_cloudy}_obs_dropped")

    switched = False
    optical_fresh = any(s in OPTICAL_SENSORS for s in fresh)
    sar_fresh = any(s in SAR_SENSORS for s in fresh)
    if n_cloudy and not optical_fresh:
        if sar_fresh:
            events.append("optical_cloudy->sar")
            switched = True
        else:
            events.append("optical_cloudy:no_sar_available")

    return GateResult(obs=kept, events=events, stale=bool(latest) and not fresh,
                      fresh_sensors=fresh, switched_to_sar=switched)