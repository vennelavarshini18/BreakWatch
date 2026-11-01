"""
ml/forecast.py — serve Model A NDVI-z 14-day forecast.
=======================================================

forecast(bundle: ObsBundle, fused: FusedState) -> Forecast

Never raises.  Returns a Forecast with:
  - validated_for_season=True and numbers filled when all gates pass.
  - validated_for_season=False and numbers=None when withheld.

Withheld when:
  1. Model file missing (data/model/model.joblib).
  2. Card common_version != common.COMMON_VERSION.
  3. The season of bundle.field.as_of is not validated in the card.
  4. ndvi_z input feature is None.

Feature assembly (serving path)
--------------------------------
Features come from the ObsBundle — no external calls:
  - sm_shallow_pctl, sm_root_pctl: from bundle.climatology
    (sm_shallow_percentile, sm_root_percentile).
  - rain_deficit_14d_mm, days_since_rain_gt_2p5mm: from bundle.climatology.
  - ndvi_z, ndvi_z_slope_3comp: from bundle.obs (SensorObs with sensor
    "modis_ndvi", day <= as_of); z is already computed by geo/modis.py using
    common.baseline.compute_z — we do not recompute it.
  - et0_sum_14d, tmax_mean_7d: from bundle.weather_daily (is_forecast=False,
    day <= as_of).
  - doy_sin, doy_cos: computed from as_of (same formula as build_features).

Column order is always feature_names() so the model receives the right columns.

Latency target: < 1 second (model loaded once at module import).
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
tqdm.monitor_interval = 0

from common import COMMON_VERSION
from common.features import feature_names
from common.ledger import ledger_add
from config import season_of
from contracts import (
    Forecast,
    FusedState,
    ObsBundle,
)

# ---------------------------------------------------------------------------
# Model / card paths  (resolved once at import)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _REPO_ROOT / "data" / "model" / "model.joblib"
_CARD_PATH = _REPO_ROOT / "data" / "model" / "model_card.json"

# Horizon must match what was used during training.
_HORIZON_DAYS = 14


# ---------------------------------------------------------------------------
# Lazy-loaded singleton: model + card are loaded once on first forecast call.
# ---------------------------------------------------------------------------


class _ModelState:
    _model = None
    _card: Optional[dict] = None
    _load_error: Optional[str] = None

    @classmethod
    def load(cls) -> tuple[object | None, dict | None, str | None]:
        """Load model and card if not yet loaded; return (model, card, error)."""
        if cls._load_error is not None:
            return None, None, cls._load_error
        if cls._model is not None:
            return cls._model, cls._card, None

        if not _MODEL_PATH.exists():
            cls._load_error = "model file missing"
            return None, None, cls._load_error
        if not _CARD_PATH.exists():
            cls._load_error = "model card missing"
            return None, None, cls._load_error

        try:
            cls._model = joblib.load(_MODEL_PATH)
        except Exception as exc:
            cls._load_error = f"model load error: {exc}"
            return None, None, cls._load_error

        try:
            cls._card = json.loads(_CARD_PATH.read_text())
        except Exception as exc:
            cls._load_error = f"card load error: {exc}"
            cls._model = None
            return None, None, cls._load_error

        return cls._model, cls._card, None


def _withheld(reason: str, card: Optional[dict] = None) -> Forecast:
    version = (card.get("version", "unknown") if card else "unknown")
    return Forecast(
        horizon_days=_HORIZON_DAYS,
        ndvi_z_p10=None,
        ndvi_z_p50=None,
        ndvi_z_p90=None,
        prob_event=None,
        model_version=version,
        validated_for_season=False,
        withheld_reason=reason,
        evidence_ids=[],
    )


# ---------------------------------------------------------------------------
# Feature assembly from ObsBundle (serving path)
# ---------------------------------------------------------------------------


def _features_from_bundle(bundle: ObsBundle) -> dict[str, Optional[float]]:
    """
    Assemble the feature dict from an ObsBundle.

    Mirrors build_features() exactly but reads from the pre-computed bundle
    fields instead of raw parquets.  Column order is set by feature_names().
    """
    as_of = bundle.field.as_of
    clim = bundle.climatology

    # ── 1. rain_deficit_14d_mm ───────────────────────────────────────────────
    rain_deficit_14d_mm: Optional[float] = clim.rain_deficit_14d_mm

    # ── 2. days_since_rain_gt_2p5mm ─────────────────────────────────────────
    days_since: Optional[float] = (
        float(clim.days_since_rain_gt_2p5mm)
        if clim.days_since_rain_gt_2p5mm is not None
        else None
    )

    # ── 3 & 4. sm percentiles ────────────────────────────────────────────────
    # Climatology stores sm_shallow_percentile / sm_root_percentile (geo/weather.py)
    sm_shallow_pctl: Optional[float] = clim.sm_shallow_percentile
    sm_root_pctl: Optional[float] = clim.sm_root_percentile

    # ── 5 & 6. ndvi_z and ndvi_z_slope_3comp ────────────────────────────────
    # SensorObs with sensor "modis_ndvi"; day is the composite END date (start+15).
    # z is already computed by geo/modis.py using common.baseline.compute_z.
    modis_obs = sorted(
        [o for o in bundle.obs if o.sensor == "modis_ndvi" and o.day <= as_of],
        key=lambda o: o.day,
    )

    ndvi_z: Optional[float] = None
    ndvi_z_slope_3comp: Optional[float] = None

    if modis_obs:
        latest = modis_obs[-1]
        ndvi_z = latest.z  # None if baseline insufficient

        # 3-composite slope from the last 3 available z-scores
        recent3 = [o for o in modis_obs[-3:] if o.z is not None]
        if len(recent3) == 3:
            zs = np.array([o.z for o in recent3], dtype=float)
            xs = np.array([0.0, 1.0, 2.0])
            ndvi_z_slope_3comp = float(np.polyfit(xs, zs, 1)[0])

    # ── 7. et0_sum_14d ───────────────────────────────────────────────────────
    et0_sum_14d: Optional[float] = None
    wx_obs = [
        w for w in bundle.weather_daily if not w.is_forecast and w.day <= as_of
    ]
    wx_obs.sort(key=lambda w: w.day)
    start14 = as_of - timedelta(days=13)
    et0_window = [w.et0_mm for w in wx_obs if w.day >= start14 and w.et0_mm is not None]
    if len(et0_window) >= 12:  # _ET0_MIN_DAYS from features.py
        et0_sum_14d = float(sum(et0_window))

    # ── 8. tmax_mean_7d ──────────────────────────────────────────────────────
    tmax_mean_7d: Optional[float] = None
    start7 = as_of - timedelta(days=6)
    tmax_window = [w.tmax_c for w in wx_obs if w.day >= start7 and w.tmax_c is not None]
    if len(tmax_window) >= 6:  # _TMAX_MIN_DAYS from features.py
        tmax_mean_7d = float(sum(tmax_window) / len(tmax_window))

    # ── 9 & 10. doy_sin, doy_cos ────────────────────────────────────────────
    doy_angle = 2.0 * math.pi * (as_of.timetuple().tm_yday / 365.0)
    doy_sin = float(math.sin(doy_angle))
    doy_cos = float(math.cos(doy_angle))

    return {
        "rain_deficit_14d_mm": rain_deficit_14d_mm,
        "days_since_rain_gt_2p5mm": days_since,
        "sm_shallow_pctl": sm_shallow_pctl,
        "sm_root_pctl": sm_root_pctl,
        "ndvi_z": ndvi_z,
        "ndvi_z_slope_3comp": ndvi_z_slope_3comp,
        "et0_sum_14d": et0_sum_14d,
        "tmax_mean_7d": tmax_mean_7d,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }


# ---------------------------------------------------------------------------
# Main forecast function
# ---------------------------------------------------------------------------


def forecast(bundle: ObsBundle, fused: FusedState) -> Forecast:
    """
    Produce an NDVI-z 14-day forecast from the ObsBundle.

    Never raises.  Returns a withheld Forecast when gates fail.
    Evidence is added to the current ledger context.
    """
    try:
        return _forecast_inner(bundle, fused)
    except Exception as exc:
        return _withheld(f"internal error: {type(exc).__name__}: {exc}")


def _forecast_inner(bundle: ObsBundle, fused: FusedState) -> Forecast:
    # ── Load model + card ────────────────────────────────────────────────────
    model, card, load_err = _ModelState.load()
    if load_err is not None:
        return _withheld(load_err)

    # ── Gate 1: common_version must match ────────────────────────────────────
    card_cv = str(card.get("common_version", ""))
    if card_cv != str(COMMON_VERSION):
        return _withheld(
            f"common version mismatch: card={card_cv!r}, runtime={COMMON_VERSION!r}",
            card,
        )

    # ── Gate 2: season must be validated in the card ─────────────────────────
    as_of = bundle.field.as_of
    try:
        season = season_of(as_of.month)
    except ValueError:
        return _withheld(f"invalid month {as_of.month}", card)

    card_seasons: dict = card.get("seasons", {})
    season_info = card_seasons.get(season, {})
    if not season_info.get("validated", False):
        return _withheld(
            f"season '{season}' not validated for production use (gate failed during training)",
            card,
        )

    # ── Assemble features ────────────────────────────────────────────────────
    feat_dict = _features_from_bundle(bundle)

    # ── Gate 3: ndvi_z must be present ──────────────────────────────────────
    if feat_dict.get("ndvi_z") is None:
        return _withheld("ndvi_z input missing (no baseline-eligible MODIS composite)", card)

    # ── Build feature array in the exact column order ────────────────────────
    fnames = feature_names()
    x = np.array([[feat_dict.get(fn) for fn in fnames]], dtype=object)
    # LightGBM handles NaN natively; convert None -> np.nan
    x = x.astype(float)

    # ── Predict ──────────────────────────────────────────────────────────────
    p50 = float(model.predict(x)[0])
    q = float(card.get("conformal_q", 1.0))
    p10 = p50 - q
    p90 = p50 + q

    # prob_event = P(NDVI-z <= -1) under Normal(p50, q/1.2816)
    # Assumption documented in train.py: residuals approx Normal, q ≈ 1.2816 * σ
    sigma_approx = q / 1.2816
    prob_event = float(norm.cdf((-1.0 - p50) / sigma_approx))

    # ── Evidence ─────────────────────────────────────────────────────────────
    ev_ids: list[str] = []
    source = "ml.forecast:model_a"

    def _ev(key: str, value, unit: str, note: str = "") -> str:
        import datetime as _dt
        # Use a fixed-epoch timestamp (not system clock) for determinism in tests.
        # We use the bundle's field.as_of date at midnight UTC as a proxy.
        fetched = _dt.datetime.combine(as_of, _dt.time.min, tzinfo=_dt.timezone.utc)
        e = ledger_add(
            key,
            round(value, 4) if isinstance(value, float) else value,
            unit=unit,
            source=source,
            obs_date=as_of,
            quality="cached",
            note=note,
            fetched_at=fetched,
        )
        return e.id

    ev_ids.append(_ev("ndvi_z_input", feat_dict["ndvi_z"], "z", f"as_of={as_of}"))
    ev_ids.append(_ev("ndvi_z_p50_forecast", p50, "z", f"horizon={_HORIZON_DAYS}d"))
    ev_ids.append(_ev("ndvi_z_p10_forecast", p10, "z", "80pct conformal lower"))
    ev_ids.append(_ev("ndvi_z_p90_forecast", p90, "z", "80pct conformal upper"))
    ev_ids.append(_ev("prob_event", prob_event, "probability", "P(ndvi_z<=-1 at t+14d)"))

    return Forecast(
        horizon_days=_HORIZON_DAYS,
        ndvi_z_p10=round(p10, 4),
        ndvi_z_p50=round(p50, 4),
        ndvi_z_p90=round(p90, 4),
        prob_event=round(prob_event, 4),
        model_version=str(card.get("version", "unknown")),
        validated_for_season=True,
        withheld_reason=None,
        evidence_ids=ev_ids,
    )

