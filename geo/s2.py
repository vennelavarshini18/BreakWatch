"""geo/s2.py - Sentinel-2 L2A de-risk spike (P1, script-level).

Planetary Computer STAC -> B04/B08/B11/SCL -> polygon median NDVI/NDMI + valid_frac per solar day.
Run:  python -m geo.s2 --lat 18.0 --lon 79.6 --as-of 2025-08-20 --days 30
Reflectance: (DN + offset) * 0.0001, offset from STAC raster:bands (else -1000 for processing baseline >= 04.00).
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BAD_SCL = (0, 1, 3, 8, 9, 10, 11)       # nodata, saturated, cloud shadow, cloud med/high, cirrus, snow


def square_polygon(lat: float, lon: float, size_m: float = 300.0) -> dict:
    dlat = size_m / 2 / 111_320.0
    dlon = size_m / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return {"type": "Polygon", "coordinates": [[[lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
                                                [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
                                                [lon - dlon, lat - dlat]]]}


def _offset_for(item) -> float:
    try:
        off = item.assets["B04"].extra_fields["raster:bands"][0].get("offset")
        if off is not None:
            return float(off)
    except Exception:
        pass
    try:
        return -1000.0 if float(item.properties.get("s2:processing_baseline", "0")) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def s2_timeseries(polygon: dict, start: dt.date, end: dt.date, max_cloud: float = 90.0,
                  resolution: int = 20) -> list[dict]:
    import numpy as np
    import odc.stac
    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401  (registers .rio)
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shp_transform

    geom = shape(polygon)
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["sentinel-2-l2a"], intersects=mapping(geom),
                                datetime=f"{start.isoformat()}/{end.isoformat()}",
                                query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    if not items:
        return []
    ds = odc.stac.load(items, bands=["B04", "B08", "B11", "SCL"], resolution=resolution,
                       groupby="solar_day", chunks={}, bbox=geom.bounds).compute()

    to_ds = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True).transform
    inside = geometry_mask([mapping(shp_transform(to_ds, geom))], out_shape=(ds.sizes["y"], ds.sizes["x"]),
                           transform=ds.rio.transform(), all_touched=True, invert=True)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return []

    item_times = np.array([np.datetime64(it.datetime.astimezone(dt.timezone.utc).replace(tzinfo=None), "ns")
                           for it in items])
    best: dict[dt.date, dict] = {}
    for i, t in enumerate(ds.time.values):
        item = items[int(np.argmin(np.abs(item_times - t)))]
        off = _offset_for(item)
        sl = ds.isel(time=i)
        b04, b08, b11 = (sl[b].values.astype("float32") for b in ("B04", "B08", "B11"))
        valid = inside & ~np.isin(sl["SCL"].values, BAD_SCL) & (b04 > 0) & (b08 > 0) & (b11 > 0)
        row = {"day": dt.date.fromisoformat(str(np.datetime_as_string(t, unit="D"))),
               "valid_frac": float(valid.sum()) / n_inside, "ndvi": None, "ndmi": None}
        if valid.any():
            r04, r08, r11 = ((b[valid] + off) * 1e-4 for b in (b04, b08, b11))
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi, ndmi = (r08 - r04) / (r08 + r04), (r08 - r11) / (r08 + r11)
            row["ndvi"], row["ndmi"] = float(np.nanmedian(ndvi)), float(np.nanmedian(ndmi))
        if row["day"] not in best or row["valid_frac"] > best[row["day"]]["valid_frac"]:
            best[row["day"]] = row
    return [best[d] for d in sorted(best)]


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--as-of", required=True, help="YYYY-MM-DD or 'today'")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--size-m", type=float, default=300.0)
    a = ap.parse_args()
    if a.as_of == "today":
        from api.clock import today_utc
        as_of = today_utc()
    else:
        as_of = dt.date.fromisoformat(a.as_of)
    t0 = time.perf_counter()
    rows = s2_timeseries(square_polygon(a.lat, a.lon, a.size_m), as_of - dt.timedelta(days=a.days), as_of)
    print(f"{'date':<12}{'ndvi':>8}{'ndmi':>8}{'valid_frac':>12}")
    for r in rows:
        f = lambda v: f"{v:8.3f}" if v is not None else "    None"
        print(f"{r['day']!s:<12}{f(r['ndvi'])}{f(r['ndmi'])}{r['valid_frac']:12.2f}")
    print(f"{len(rows)} scenes-days, latency {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    _main()