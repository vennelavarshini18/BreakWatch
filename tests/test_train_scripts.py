"""Tests for ml/train/sample_points.py and ml/train/pull_weather.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from ml.train.sample_points import _assign_district, _stratified_sample
from ml.train.pull_weather import (
    PARQUET_COLUMNS,
    _load_manifest,
    _process_response,
    _update_manifest,
)


def test_assign_district():
    """Point inside polygon is correctly assigned, outside falls back to nearest."""
    poly_a = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
    poly_b = Polygon([(3, 0), (3, 2), (5, 2), (5, 0)])
    districts = [
        {"name": "DistrictA", "geom": poly_a},
        {"name": "DistrictB", "geom": poly_b},
    ]

    # Inside DistrictA
    assert _assign_district(1.0, 1.0, districts) == "DistrictA"
    # Inside DistrictB
    assert _assign_district(1.0, 4.0, districts) == "DistrictB"
    # Closer to DistrictA
    assert _assign_district(1.0, 2.2, districts) == "DistrictA"


def test_stratified_sample():
    """Stratified sampling preserves proportions and seed reproducibility."""
    poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    districts = [{"name": "SingleDist", "geom": poly}]

    cells = [
        {"lat": 1.0 + i * 0.1, "lon": 1.0 + i * 0.1, "cropland_frac": 0.8}
        for i in range(50)
    ]

    sampled_1 = _stratified_sample(cells, districts, n=10, seed=42)
    sampled_2 = _stratified_sample(cells, districts, n=10, seed=42)

    assert len(sampled_1) == 10
    assert sampled_1 == sampled_2
    assert sampled_1[0]["id"] == "pt_0001"
    assert sampled_1[0]["district"] == "SingleDist"


def test_process_response_weather():
    """Open-Meteo payload is processed into exact PARQUET_COLUMNS with IST aggregation."""
    # 24 hours for 1 day: 2024-06-01
    times = pd.date_range("2024-06-01 00:00", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    payload = {
        "daily": {
            "time": ["2024-06-01"],
            "precipitation_sum": [12.5],
            "et0_fao_evapotranspiration": [4.2],
            "temperature_2m_max": [35.0],
        },
        "hourly": {
            "time": times,
            "soil_moisture_0_to_7cm": [0.20] * 24,
            "soil_moisture_7_to_28cm": [0.30] * 24,
            "soil_moisture_28_to_100cm": [0.40] * 24,
        },
    }

    df = _process_response(payload)
    assert list(df.columns) == PARQUET_COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert str(row["date"]) == "2024-06-01"
    assert row["precip_mm"] == 12.5
    assert row["et0_mm"] == 4.2
    assert row["tmax_c"] == 35.0
    # 0-7 cm
    assert pytest.approx(row["sm_shallow"], rel=1e-3) == 0.20
    # mean of 7-28 cm (0.30) and 28-100 cm (0.40) = 0.35
    assert pytest.approx(row["sm_root"], rel=1e-3) == 0.35


def test_manifest_roundtrip(tmp_path, monkeypatch):
    """Manifest tracks completed points correctly."""
    manifest_file = tmp_path / "manifest.json"
    monkeypatch.setattr("ml.train.pull_weather.MANIFEST_JSON", manifest_file)

    assert _load_manifest() == set()

    done = {"pt_0001", "pt_0002"}
    _update_manifest(done)

    loaded = _load_manifest()
    assert loaded == done

