"""Offline tests for geo/modis.py (STAC search and raster reads are patched; tests/ only)."""
import datetime as dt

import pandas as pd

import geo.modis as M
from common.baseline import compute_z
from contracts import FieldInput

POLY = {"type": "Polygon", "coordinates": [[[79.60, 18.00], [79.61, 18.00], [79.61, 18.01], [79.60, 18.01], [79.60, 18.00]]]}
ENDS = [(7, 20), (8, 5), (8, 21)]


def _value(ed: dt.date) -> float:
    return 0.5 + 0.02 * ((ed.year * 3 + ed.month) % 5)


def _items():
    items = []
    for yr in range(2010, 2026):
        for m, d in ENDS:
            ed = dt.date(yr, m, d)
            sd = ed - dt.timedelta(days=15)
            items.append({"id": f"MOD13Q1.{sd}", "properties": {"start_datetime": f"{sd}T00:00:00Z",
                                                                "end_datetime": f"{ed}T23:59:59Z"}})
    ed = dt.date(2025, 9, 6)                              # ends AFTER as_of -> must be excluded
    sd = ed - dt.timedelta(days=15)
    items.append({"id": f"MOD13Q1.{sd}", "properties": {"start_datetime": f"{sd}T00:00:00Z",
                                                        "end_datetime": f"{ed}T23:59:59Z"}})
    return items


def _fake_extract(item, lon, lat):
    ed = dt.date.fromisoformat(item["properties"]["end_datetime"][:10])
    return {"ndvi": _value(ed), "n_good_pixels": 9, "quality_ok": True}, True


def test_modis_obs_end_date_no_lookahead_and_z_matches_compute_z(monkeypatch):
    as_of = dt.date(2025, 8, 25)
    monkeypatch.setattr(M, "_search_items", lambda lon, lat, a: _items())
    monkeypatch.setattr(M, "_extract", _fake_extract)
    field = FieldInput(polygon=POLY, as_of=as_of, mode="replay")

    obs, events = M.get_modis_obs(field)

    assert [o.day for o in obs] == [dt.date(2025, 7, 20), dt.date(2025, 8, 5), dt.date(2025, 8, 21)]
    assert all(o.day <= as_of for o in obs)
    assert all(o.sensor == "modis_ndvi" and o.z is not None for o in obs)

    series = pd.Series({pd.Timestamp(dt.date(y, m, d)): _value(dt.date(y, m, d))
                        for y in range(2010, 2026) for m, d in ENDS})
    for o in obs:
        expected, n_years, _ = compute_z(o.value, series, o.day, "modis_ndvi")
        assert o.z == expected
        assert o.n_baseline == n_years == 15


def test_modis_composite_without_good_pixels_is_skipped_not_faked(monkeypatch):
    as_of = dt.date(2025, 8, 25)
    monkeypatch.setattr(M, "_search_items", lambda lon, lat, a: _items())

    def extract(item, lon, lat):
        rec, cached = _fake_extract(item, lon, lat)
        if item["properties"]["end_datetime"].startswith("2025-08-21"):
            rec = {"ndvi": float("nan"), "n_good_pixels": 0, "quality_ok": False}
        return rec, cached

    monkeypatch.setattr(M, "_extract", extract)
    obs, events = M.get_modis_obs(FieldInput(polygon=POLY, as_of=as_of, mode="replay"))
    assert dt.date(2025, 8, 21) not in [o.day for o in obs]
    assert "modis_composite_unusable:2025-08-21" in events