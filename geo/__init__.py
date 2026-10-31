"""geo package (P1). Kept import-light so `python -m geo.weather` works without double-import warnings."""
from __future__ import annotations


def build_observations(field):
    """S1 version: weather + climatology + ensemble (live). Optical/SAR not built yet -> explicit events, no fake obs.
    field: contracts.FieldInput -> contracts.ObsBundle"""
    from contracts import ObsBundle
    from geo.weather import centroid_of, get_weather

    lat, lon = centroid_of(field.polygon)
    wb = get_weather(lat, lon, field.as_of, field.mode)
    events = list(wb.events) + ["optical_unavailable:not_implemented", "sar_unavailable:not_implemented"]
    return ObsBundle(field=field, obs=[], weather_daily=wb.days, rain_ensemble=wb.rain_ensemble,
                     climatology=wb.climatology, cropland_frac=None, events=events, evidence=wb.evidence)