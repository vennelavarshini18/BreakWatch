"""geo package (P1). Kept import-light so `python -m geo.weather` works without double-import warnings."""
from __future__ import annotations


def build_observations(field):
    """S1 version: weather + climatology + ensemble (live). Optical/SAR not built yet -> explicit events, no fake obs.
    The quality gate runs on whatever sensor observations exist (none yet), so S2 loaders only need to fill `obs`.
    field: contracts.FieldInput -> contracts.ObsBundle"""
    from contracts import ObsBundle
    from geo.quality import quality_gate
    from geo.weather import centroid_of, get_weather

    lat, lon = centroid_of(field.polygon)
    wb = get_weather(lat, lon, field.as_of, field.mode)
    gate = quality_gate([], field.as_of)
    events = list(wb.events) + ["optical_unavailable:not_implemented", "sar_unavailable:not_implemented"] + gate.events
    return ObsBundle(field=field, obs=gate.obs, weather_daily=wb.days, rain_ensemble=wb.rain_ensemble,
                     climatology=wb.climatology, cropland_frac=None, events=events, evidence=wb.evidence)