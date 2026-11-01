"""geo package (P1). Kept import-light so `python -m geo.weather` works without double-import warnings."""
from __future__ import annotations


def build_observations(field):
    """S1+ version: weather + climatology + ensemble (live) + MODIS NDVI obs. Optical/SAR not built yet -> explicit
    events, no fake obs.  field: contracts.FieldInput -> contracts.ObsBundle"""
    from contracts import ObsBundle
    from geo.weather import EvidenceFactory, centroid_of, get_weather

    lat, lon = centroid_of(field.polygon)
    factory = EvidenceFactory()                       # one factory -> unique evidence ids across modules
    wb = get_weather(lat, lon, field.as_of, field.mode, factory=factory)
    events = list(wb.events)

    obs = []
    try:
        from geo.modis import get_modis_obs
        obs, modis_events = get_modis_obs(field, factory)
        events += modis_events
    except NotImplementedError:
        events.append("module_unavailable:geo.modis")
    except Exception as e:                            # network/STAC/raster failure -> explicit event, no fake obs
        events.append(f"modis_failed:{type(e).__name__}")

    events += ["optical_unavailable:not_implemented", "sar_unavailable:not_implemented"]
    return ObsBundle(field=field, obs=obs, weather_daily=wb.days, rain_ensemble=wb.rain_ensemble,
                     climatology=wb.climatology, cropland_frac=None, events=events,
                     evidence=list(factory.rows))