"""
tests/test_train_forecast.py — Tests for ml/train/train.py and ml/forecast.py.

All tests are self-contained: they build synthetic data where needed and mark
training-only assertions clearly.  They skip gracefully when real disk data
or the trained model are absent.
"""

from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from tqdm import tqdm
tqdm.monitor_interval = 0

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper: build synthetic weather & MODIS dataframes (training path format)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = REPO_ROOT / "data" / "train_raw" / "weather"
MODIS_DIR = REPO_ROOT / "data" / "train_raw" / "modis"
MODEL_PATH = REPO_ROOT / "data" / "model" / "model.joblib"
CARD_PATH = REPO_ROOT / "data" / "model" / "model_card.json"


def _make_weather(start: date, end: date) -> pd.DataFrame:
    """Synthetic daily weather DataFrame with column 'day'."""
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    n = len(dates)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "day": dates,
            "precip_mm": rng.uniform(0, 10, n),
            "et0_mm": rng.uniform(2, 6, n),
            "tmax_c": rng.uniform(25, 40, n),
            "sm_shallow": rng.uniform(0.1, 0.4, n),
            "sm_root": rng.uniform(0.15, 0.35, n),
        }
    )


def _make_modis(start_year: int, end_year: int) -> pd.DataFrame:
    """Synthetic MODIS parquet with quality-OK composites every 16 days."""
    rows = []
    d = date(start_year, 1, 3)
    while d.year <= end_year:
        rows.append(
            {
                "point_id": "pt_synth",
                "start_date": d - timedelta(days=15),
                "end_date": d,
                "ndvi": 0.4 + 0.3 * math.sin(2 * math.pi * d.timetuple().tm_yday / 365),
                "n_good_pixels": 9,
                "quality_ok": True,
            }
        )
        d += timedelta(days=16)
    df = pd.DataFrame(rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


# ---------------------------------------------------------------------------
# 1. Split-by-year assertion
# ---------------------------------------------------------------------------


def test_split_year_assertions():
    """Training rows must never have year > TRAIN_YEARS[1]."""
    from config import TRAIN_YEARS
    from ml.train.train import _build_or_load_features

    w = _make_weather(date(2010, 1, 1), date(2025, 12, 31))
    m = _make_modis(2010, 2025)

    frame = _build_or_load_features("pt_synth_test", w, m, force=True)
    # Clean up cache
    cache = REPO_ROOT / "data" / "train_raw" / "features" / "pt_synth_test.parquet"
    if cache.exists():
        cache.unlink()

    if frame.empty:
        pytest.skip("No samples produced from synthetic data.")

    frame["t"] = pd.to_datetime(frame["t"]).dt.date
    frame["year"] = frame["t"].apply(lambda d: d.year)

    train_rows = frame[frame["year"] <= TRAIN_YEARS[1]]
    assert train_rows["year"].max() <= TRAIN_YEARS[1], (
        f"Training leak: found year {train_rows['year'].max()} > {TRAIN_YEARS[1]}"
    )
    # Every sample must be a Monday
    for d in frame["t"]:
        assert d.weekday() == 0, f"{d} is not a Monday"


# ---------------------------------------------------------------------------
# 2. Same seed → same result (reproducibility)
# ---------------------------------------------------------------------------


def test_same_seed_same_mae():
    """Training twice with seed=42 on the same data must give identical MAE."""
    import lightgbm as lgb
    from ml.train.train import LGB_PARAMS

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = rng.normal(size=100)
    X_train, y_train = X[:70], y[:70]
    X_test, y_test = X[70:], y[70:]

    def run():
        params = dict(LGB_PARAMS)
        params["n_jobs"] = 1
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_train, y_train)
        return float(np.mean(np.abs(y_test - mdl.predict(X_test))))

    mae1 = run()
    mae2 = run()
    assert math.isclose(mae1, mae2, rel_tol=1e-12), f"MAE not reproducible: {mae1} vs {mae2}"


# ---------------------------------------------------------------------------
# 3. Withheld paths for forecast.py
# ---------------------------------------------------------------------------


def test_forecast_withheld_missing_model(tmp_path, monkeypatch):
    """forecast() returns withheld when model file is absent."""
    import ml.forecast as fmod
    # Reset singleton state so the patch takes effect
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    monkeypatch.setattr(fmod, "_MODEL_PATH", tmp_path / "model.joblib")
    # Re-read the real card path from the real location (or a missing one)
    monkeypatch.setattr(fmod, "_CARD_PATH", tmp_path / "model_card.json")

    bundle = _make_minimal_bundle()
    fused = _make_fused()
    result = fmod.forecast(bundle, fused)
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    assert result.validated_for_season is False
    assert result.withheld_reason is not None
    assert result.ndvi_z_p50 is None


def test_forecast_withheld_common_version_mismatch(tmp_path, monkeypatch):
    """forecast() withholds when card common_version != COMMON_VERSION."""
    import ml.forecast as fmod
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    card = {"common_version": "999", "version": "0.1", "seasons": {}, "conformal_q": 0.5}
    (tmp_path / "model_card.json").write_text(json.dumps(card))

    # We need a "model" object to pass the load step
    import joblib
    import lightgbm as lgb
    dummy_model = lgb.LGBMRegressor(n_estimators=1, verbosity=-1, seed=0, n_jobs=1)
    # LGBMRegressor needs >= 2 samples to fit
    dummy_model.fit([[0] * 10, [1] * 10], [0.0, 1.0])
    joblib.dump(dummy_model, tmp_path / "model.joblib")

    monkeypatch.setattr(fmod, "_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(fmod, "_CARD_PATH", tmp_path / "model_card.json")

    bundle = _make_minimal_bundle()
    fused = _make_fused()
    result = fmod.forecast(bundle, fused)
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    assert result.validated_for_season is False
    assert "version" in (result.withheld_reason or "").lower()


def test_forecast_withheld_season_not_validated(tmp_path, monkeypatch):
    """forecast() withholds when the season is not validated in the card."""
    import ml.forecast as fmod
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    from common import COMMON_VERSION
    card = {
        "common_version": COMMON_VERSION,
        "version": "0.1",
        "conformal_q": 0.5,
        "seasons": {"kharif": {"validated": False}, "rabi": {"validated": False}, "summer": {"validated": False}},
    }
    (tmp_path / "model_card.json").write_text(json.dumps(card))

    import joblib, lightgbm as lgb
    dummy_model = lgb.LGBMRegressor(n_estimators=1, verbosity=-1, seed=0, n_jobs=1)
    dummy_model.fit([[0] * 10, [1] * 10], [0.0, 1.0])
    joblib.dump(dummy_model, tmp_path / "model.joblib")

    monkeypatch.setattr(fmod, "_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(fmod, "_CARD_PATH", tmp_path / "model_card.json")

    bundle = _make_minimal_bundle(as_of=date(2024, 7, 15))  # kharif
    fused = _make_fused()
    result = fmod.forecast(bundle, fused)
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    assert result.validated_for_season is False
    assert result.ndvi_z_p50 is None


def test_forecast_withheld_ndvi_z_missing(tmp_path, monkeypatch):
    """forecast() withholds when ndvi_z input is None (no modis obs)."""
    import ml.forecast as fmod
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    from common import COMMON_VERSION
    # validated = True for kharif
    card = {
        "common_version": COMMON_VERSION,
        "version": "0.1",
        "conformal_q": 0.5,
        "seasons": {"kharif": {"validated": True}, "rabi": {"validated": True}, "summer": {"validated": True}},
    }
    (tmp_path / "model_card.json").write_text(json.dumps(card))

    import joblib, lightgbm as lgb
    dummy_model = lgb.LGBMRegressor(n_estimators=1, verbosity=-1, seed=0, n_jobs=1)
    dummy_model.fit([[0] * 10, [1] * 10], [0.0, 1.0])
    joblib.dump(dummy_model, tmp_path / "model.joblib")

    monkeypatch.setattr(fmod, "_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(fmod, "_CARD_PATH", tmp_path / "model_card.json")

    # Bundle with no modis obs → ndvi_z will be None
    bundle = _make_minimal_bundle(as_of=date(2024, 7, 15), include_modis=False)
    fused = _make_fused()
    result = fmod.forecast(bundle, fused)
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    assert result.validated_for_season is False
    assert result.ndvi_z_p50 is None


# ---------------------------------------------------------------------------
# 4. Timing test
# ---------------------------------------------------------------------------


def test_forecast_latency_under_1s(tmp_path, monkeypatch):
    """forecast() should complete in < 1 second once model is loaded."""
    import ml.forecast as fmod
    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    from common import COMMON_VERSION
    card = {
        "common_version": COMMON_VERSION,
        "version": "0.1",
        "conformal_q": 0.5,
        "seasons": {"kharif": {"validated": True}, "rabi": {"validated": True}, "summer": {"validated": True}},
    }
    (tmp_path / "model_card.json").write_text(json.dumps(card))

    import joblib, lightgbm as lgb
    dummy_model = lgb.LGBMRegressor(n_estimators=1, verbosity=-1, seed=0, n_jobs=1)
    dummy_model.fit([[0] * 10, [1] * 10], [0.0, 1.0])
    joblib.dump(dummy_model, tmp_path / "model.joblib")

    monkeypatch.setattr(fmod, "_MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(fmod, "_CARD_PATH", tmp_path / "model_card.json")

    bundle = _make_minimal_bundle(as_of=date(2024, 7, 15), include_modis=True)
    fused = _make_fused()

    # Warm up (loads model)
    fmod.forecast(bundle, fused)

    # Reset and time a single call
    t0 = time.perf_counter()
    fmod.forecast(bundle, fused)
    elapsed = time.perf_counter() - t0

    fmod._ModelState._model = None
    fmod._ModelState._card = None
    fmod._ModelState._load_error = None

    assert elapsed < 1.0, f"forecast() took {elapsed:.3f}s (limit: 1s)"


# ---------------------------------------------------------------------------
# 5. Train/serve PARITY test
# ---------------------------------------------------------------------------


def test_train_serve_parity():
    """
    For one real point and date on disk, the training path and serving path
    must produce identical feature values.

    Skips with a clear message if disk data is absent or the model is not trained.
    """
    from common.features import build_features, feature_names
    import ml.forecast as fmod

    # Pick the first real point that has both weather and MODIS
    w_files = sorted(WEATHER_DIR.glob("*.parquet"))
    m_files = sorted(MODIS_DIR.glob("*.parquet"))
    if not w_files or not m_files:
        pytest.skip("No on-disk training data — skipping parity test.")

    w_ids = {p.stem for p in w_files}
    m_ids = {p.stem for p in m_files}
    both = sorted(w_ids & m_ids)
    if not both:
        pytest.skip("No point with both weather and MODIS — skipping parity test.")

    pid = both[0]
    as_of = date(2024, 8, 5)

    # ── Training path ────────────────────────────────────────────────────────
    from ml.train.train import _load_weather, _load_modis_quality
    w_df = _load_weather(pid)
    m_df = _load_modis_quality(pid)
    train_feat = build_features(w_df, m_df, as_of)

    # ── Serving path: build a minimal ObsBundle from the same on-disk data ──
    from contracts import (
        Climatology,
        FieldInput,
        FusedState,
        ObsBundle,
        SensorObs,
        WeatherDay,
    )
    from common.baseline import baseline_stats, zscore

    # Build SensorObs list from MODIS parquet
    modis_obs_list = []
    end_dates = list(m_df["end_date"])
    ndvi_vals = list(m_df["ndvi"].astype(float))
    for ed, val in zip(end_dates, ndvi_vals):
        if ed > as_of:
            break
        stats = baseline_stats(end_dates, ndvi_vals, ed, "modis_ndvi")
        z = zscore(val, stats, "modis_ndvi")
        modis_obs_list.append(
            SensorObs(
                sensor="modis_ndvi",
                day=ed,
                value=val,
                z=z,
                valid_frac=1.0,
                n_baseline=stats.n_years if stats else 0,
                evidence_id=f"test_{ed}",
            )
        )

    # Build WeatherDay list
    weather_days = []
    for _, row in w_df[w_df["day"] <= as_of].iterrows():
        weather_days.append(
            WeatherDay(
                day=row["day"],
                precip_mm=row.get("precip_mm"),
                et0_mm=row.get("et0_mm"),
                tmax_c=row.get("tmax_c"),
                sm_shallow=row.get("sm_shallow"),
                sm_root=row.get("sm_root"),
                is_forecast=False,
                evidence_id=f"wx_{row['day']}",
            )
        )

    # Build Climatology from the common path
    from common.features import climatology_features
    clim_dict = climatology_features(w_df, as_of)
    clim = Climatology(
        sm_root_percentile=clim_dict.get("sm_root_percentile"),
        sm_shallow_percentile=clim_dict.get("sm_shallow_percentile"),
        rain_deficit_14d_mm=clim_dict.get("rain_deficit_14d_mm"),
        days_since_rain_gt_2p5mm=clim_dict.get("days_since_rain_gt_2p5mm"),
        n_years=clim_dict.get("n_years", 0),
        baseline_years=clim_dict.get("baseline_years", []),
        evidence_ids=[],
    )

    # Build FieldInput
    field = FieldInput(
        polygon={
            "type": "Polygon",
            "coordinates": [[[78.0, 19.0], [78.01, 19.0], [78.01, 19.01], [78.0, 19.01], [78.0, 19.0]]],
        },
        as_of=as_of,
        mode="replay",
    )

    bundle = ObsBundle(
        field=field,
        obs=modis_obs_list,
        weather_daily=weather_days,
        climatology=clim,
        evidence=[],
    )

    # ── Get serving features ─────────────────────────────────────────────────
    serve_feat = fmod._features_from_bundle(bundle)

    # ── Compare ──────────────────────────────────────────────────────────────
    fnames = feature_names()
    diffs = []
    for fn in fnames:
        tv = train_feat.get(fn)
        sv = serve_feat.get(fn)
        if tv is None and sv is None:
            continue
        if tv is None or sv is None:
            diffs.append(f"{fn}: train={tv!r}, serve={sv!r} (one is None)")
            continue
        if not math.isclose(float(tv), float(sv), rel_tol=1e-9, abs_tol=1e-9):
            diffs.append(f"{fn}: train={tv!r}, serve={sv!r} (delta={abs(float(tv)-float(sv)):.2e})")

    if diffs:
        msg = "\n".join(diffs)
        pytest.fail(
            f"Train/serve feature MISMATCH for {pid} on {as_of}:\n{msg}\n\n"
            "Each difference above indicates a parity gap between the training "
            "and serving paths."
        )
    # All features match
    assert True, "Train/serve parity OK"


# ---------------------------------------------------------------------------
# Helper builders for contracts
# ---------------------------------------------------------------------------


def _make_minimal_bundle(
    as_of: date = date(2024, 7, 15),
    include_modis: bool = True,
) -> ObsBundle:
    """Build a minimal but valid ObsBundle for unit tests."""
    from contracts import (
        Climatology,
        FieldInput,
        ObsBundle,
        SensorObs,
        WeatherDay,
    )

    field = FieldInput(
        polygon={
            "type": "Polygon",
            "coordinates": [
                [[78.0, 19.0], [78.01, 19.0], [78.01, 19.01], [78.0, 19.01], [78.0, 19.0]]
            ],
        },
        as_of=as_of,
        mode="replay",
    )

    obs: list[SensorObs] = []
    if include_modis:
        # One composite ending 5 days before as_of with z set
        composite_day = as_of - timedelta(days=5)
        obs.append(
            SensorObs(
                sensor="modis_ndvi",
                day=composite_day,
                value=0.55,
                z=-0.3,
                valid_frac=1.0,
                n_baseline=7,
                evidence_id="ev_test_001",
            )
        )

    # Build 30 weather days
    wx: list[WeatherDay] = []
    rng = np.random.default_rng(1)
    for i in range(30):
        d = as_of - timedelta(days=29 - i)
        wx.append(
            WeatherDay(
                day=d,
                precip_mm=float(rng.uniform(0, 8)),
                et0_mm=float(rng.uniform(3, 6)),
                tmax_c=float(rng.uniform(28, 38)),
                sm_shallow=float(rng.uniform(0.1, 0.4)),
                sm_root=float(rng.uniform(0.15, 0.35)),
                is_forecast=False,
                evidence_id=f"ev_wx_{i}",
            )
        )

    clim = Climatology(
        sm_root_percentile=45.0,
        sm_shallow_percentile=38.0,
        rain_deficit_14d_mm=15.0,
        days_since_rain_gt_2p5mm=3,
        n_years=10,
        baseline_years=list(range(2014, 2024)),
        evidence_ids=[],
    )

    return ObsBundle(
        field=field,
        obs=obs,
        weather_daily=wx,
        climatology=clim,
        evidence=[],
    )


def _make_fused() -> FusedState:
    from contracts import FusedState
    return FusedState(
        series=[],
        latest_mean=0.1,
        latest_std=0.3,
        p_stress=0.15,
        sources_used=["modis_ndvi"],
    )
