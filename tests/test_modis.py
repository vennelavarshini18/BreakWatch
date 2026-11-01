"""Tests for ml/train/pull_modis_cube.py and ml/train/check_data.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.train.pull_modis_cube import _deduplicate_items, load_manifest, save_manifest
from ml.train.check_data import check_weather, check_modis, eligible_points


def test_deduplicate_items():
    """Duplicate items sharing the same start_datetime are collapsed to one per date."""
    items = [
        {"id": "tile1_d1", "properties": {"start_datetime": "2020-01-01T00:00:00Z"}},
        {"id": "tile2_d1", "properties": {"start_datetime": "2020-01-01T00:00:00Z"}},
        {"id": "tile1_d2", "properties": {"start_datetime": "2020-01-17T00:00:00Z"}},
    ]
    deduped = _deduplicate_items(items)
    assert len(deduped) == 2
    dates = [i["properties"]["start_datetime"][:10] for i in deduped]
    assert dates == ["2020-01-01", "2020-01-17"]


def test_manifest_roundtrip(tmp_path, monkeypatch):
    """Manifest save and load operations preserve contents."""
    m_file = tmp_path / "modis_manifest.json"
    monkeypatch.setattr("ml.train.pull_modis_cube.MANIFEST_PATH", m_file)

    assert load_manifest() == {}
    save_manifest({"pt_0001": "done", "pt_0002": "done"})
    loaded = load_manifest()
    assert loaded["pt_0001"] == "done"
    assert loaded["pt_0002"] == "done"


def test_check_missing_point():
    """Non-existent points return graceful missing indicators."""
    w = check_weather("non_existent_pt_9999")
    assert w["weather_ok"] is False
    assert w["weather_rows"] == 0

    m = check_modis("non_existent_pt_9999")
    assert m["modis_ok"] is False
    assert m["modis_rows"] == 0


def test_eligible_points_filtering():
    """Eligible points only include points with completed weather files."""
    pts = eligible_points()
    assert isinstance(pts, list)
    assert len(pts) > 0
    # Every eligible point must have an existing weather parquet file
    weather_dir = Path(__file__).resolve().parents[1] / "data" / "train_raw" / "weather"
    for pt in pts:
        assert (weather_dir / f"{pt['id']}.parquet").exists()


def test_existing_modis_parquets():
    """If MODIS files have been pulled, verify their schema, types, and value ranges."""
    modis_dir = Path(__file__).resolve().parents[1] / "data" / "train_raw" / "modis"
    files = list(modis_dir.glob("*.parquet"))
    if not files:
        pytest.skip("No MODIS parquet files generated yet")

    required_cols = {"point_id", "start_date", "end_date", "ndvi", "n_good_pixels", "quality_ok"}
    for f in files:
        df = pd.read_parquet(f)
        assert required_cols.issubset(set(df.columns))
        assert len(df) > 0

        # Validate dates
        start = pd.to_datetime(df["start_date"])
        end = pd.to_datetime(df["end_date"])
        assert (end >= start).all(), f"End date before start date found in {f}"

        # Validate NDVI values
        valid_ndvi = df["ndvi"].dropna()
        if len(valid_ndvi) > 0:
            assert (valid_ndvi >= -0.2).all()
            assert (valid_ndvi <= 1.0).all()

        # Validate good pixel counts
        assert (df["n_good_pixels"] >= 0).all()
