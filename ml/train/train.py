"""
ml/train/train.py — Train Model A: LightGBM NDVI-z 14-day forecast.
====================================================================

Usage:
    python -m ml.train.train          # full run from disk files
    python -m ml.train.train --debug  # single point, quick sanity

Data requirements
-----------------
- Eligible points: both data/train_raw/weather/<id>.parquet and
  data/train_raw/modis/<id>.parquet must exist.  If fewer than 4 districts
  are covered, the script warns clearly but continues.
- If a pull_modis_cube process is still running, the script aborts.

Target rule (recorded here and in model_card.json)
---------------------------------------------------
Weekly sample dates t (every Monday), 2015–2025.
Target = NDVI z at t + 14 days, obtained by linearly interpolating between
the z-scores of consecutive quality-passing MODIS composites aligned to their
END dates.  This interpolated series is used **only** for the target; the
feature ndvi_z at time t uses the ordinary production path (build_features).
A sample is dropped when:
  - the target cannot be formed (no quality-OK composite end_date on both
    sides of t+14), or
  - the feature ndvi_z is None at time t (no baseline yet).
Event = target_z <= -1.0.

Split (from config.py)
-----------------------
train  2015–2020  (years <= TRAIN_YEARS[1])
calib  2021–2022  (CALIB_YEARS[0] <= year <= CALIB_YEARS[1])
test   2023–2025  (years >= TEST_YEARS[0])

No tuning on test years.

Conformal interval
------------------
q = finite-sample 80% quantile of |calib residual|.
p10 = p50 − q,  p90 = p50 + q.
prob_event = Normal CDF((-1 − p50) / (q / 1.2816)).
Assumption: residuals are approximately Normal; q is the half-width of an
approximate 80% interval, so q ≈ 1.2816 × σ_residual under Normality, hence
σ_residual ≈ q / 1.2816 is used to convert to P(target <= -1).

MODIS quality_ok filter
-----------------------
Only composites with quality_ok == True are used for both features and the
target series.  Serving (geo/modis.py) applies the same filter: line 121
`if not rec.get("quality_ok") or ndvi != ndvi: continue`.
So training and serving use identical data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import COMMON_VERSION
from common.features import build_features, feature_names
from config import (
    CALIB_YEARS,
    COVERAGE_MIN,
    N_TEST_MIN,
    SEASONS,
    SKILL_MIN,
    TEST_YEARS,
    TRAIN_YEARS,
    season_of,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = REPO_ROOT / "data" / "train_raw" / "weather"
MODIS_DIR = REPO_ROOT / "data" / "train_raw" / "modis"
FEATURES_DIR = REPO_ROOT / "data" / "train_raw" / "features"
MODEL_DIR = REPO_ROOT / "data" / "model"
PLOTS_DIR = MODEL_DIR / "plots"
MODEL_PATH = MODEL_DIR / "model.joblib"
CARD_PATH = MODEL_DIR / "model_card.json"
TRAIN_POINTS_JSON = REPO_ROOT / "data" / "train_points.json"

HORIZON_DAYS = 14       # target is ndvi_z at t + HORIZON_DAYS
EVENT_THRESHOLD = -1.0  # z <= this => event=1

# LightGBM hyperparameters (fixed; no tuning on test)
LGB_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "verbosity": -1,
    "n_jobs": 1,
}

# ---------------------------------------------------------------------------
# Guard: abort if pull process is running
# ---------------------------------------------------------------------------


def _check_no_pull_running(allow_running: bool = False) -> None:
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        procs = [
            l
            for l in result.stdout.splitlines()
            if "pull_" in l and "grep" not in l and "train.py" not in l
        ]
    except Exception:
        procs = []
    if procs:
        for p in procs:
            print(f"  Running: {p.strip()}")
        if not allow_running:
            raise SystemExit(
                "\n⚠️  A pull_modis_cube process is still running (shown above).\n"
                "Wait for it to finish or pass --allow-running-pull to train with\n"
                "points currently completed on disk.\n"
            )
        else:
            print(
                "\n⚠️  Proceeding with completed points on disk despite running pull.\n"
            )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _eligible_points() -> list[dict]:
    """Points that have BOTH a completed weather and MODIS parquet."""
    w_ids = {p.stem for p in WEATHER_DIR.glob("*.parquet")}
    m_ids = {p.stem for p in MODIS_DIR.glob("*.parquet")}
    both = w_ids & m_ids
    with open(TRAIN_POINTS_JSON) as f:
        pts = json.load(f)
    return [p for p in pts if p["id"] in both]


def _load_weather(pid: str) -> pd.DataFrame:
    """Load weather parquet; rename 'date' -> 'day' as build_features expects."""
    df = pd.read_parquet(WEATHER_DIR / f"{pid}.parquet")
    if "date" in df.columns and "day" not in df.columns:
        df = df.rename(columns={"date": "day"})
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df.sort_values("day").reset_index(drop=True)


def _load_modis_quality(pid: str) -> pd.DataFrame:
    """Load MODIS parquet, keep only quality_ok == True rows."""
    df = pd.read_parquet(MODIS_DIR / f"{pid}.parquet")
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    return df[df["quality_ok"]].copy().sort_values("end_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target series: linear interpolation of NDVI z between composites
# ---------------------------------------------------------------------------


def _build_z_series(modis_df: pd.DataFrame) -> pd.Series:
    """
    Build a daily NDVI-z series by linearly interpolating between
    the z-scores of consecutive quality-OK MODIS composites.

    The z-scores are computed by build_features (which uses baseline_stats /
    zscore from common.baseline), but we cannot call build_features with
    future data for the target.  Instead we compute the z-score of each
    composite's end_date value directly using only earlier-year baseline data
    — exactly the same protocol build_features uses internally.

    The resulting series is used ONLY for the target label; not for features.
    """
    from common.baseline import baseline_stats, zscore

    # We have all quality-ok rows; compute z for each composite end_date.
    end_dates = list(modis_df["end_date"])
    ndvi_vals = list(modis_df["ndvi"].astype(float))

    z_points: list[tuple[date, float]] = []
    for ed, val in zip(end_dates, ndvi_vals):
        stats = baseline_stats(end_dates, ndvi_vals, ed, "modis_ndvi")
        z = zscore(val, stats, "modis_ndvi")
        if z is not None:
            z_points.append((ed, z))

    if not z_points:
        return pd.Series(dtype=float)

    # Build sparse date-indexed series and interpolate to daily
    z_dates, z_vals = zip(*z_points)
    sparse = pd.Series(
        list(z_vals),
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in z_dates]),
        dtype=float,
    )
    # Daily range covering the whole history
    if len(z_dates) < 2:
        return sparse
    daily_idx = pd.date_range(sparse.index.min(), sparse.index.max(), freq="D")
    daily = sparse.reindex(daily_idx).interpolate(method="linear")
    daily.index = [d.date() for d in daily.index]
    return daily


def _get_target(z_daily: pd.Series, t: date) -> Optional[float]:
    """Return interpolated NDVI z at t + HORIZON_DAYS; None if not available."""
    target_d = t + timedelta(days=HORIZON_DAYS)
    if target_d not in z_daily:
        return None
    v = z_daily[target_d]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Feature caching (per-point, on disk)
# ---------------------------------------------------------------------------


def _feature_cache_path(pid: str) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURES_DIR / f"{pid}.parquet"


def _build_or_load_features(
    pid: str,
    weather_df: pd.DataFrame,
    modis_df: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns = feature_names() + ['t', 'target_z', 'event'].

    Loads from the on-disk feature cache if present and not forced.
    """
    cache = _feature_cache_path(pid)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    fnames = feature_names()
    z_daily = _build_z_series(modis_df)

    # Sample dates: every Monday 2015-01-01..2025-12-31
    t = date(2015, 1, 5)  # First Monday on or after 2015-01-01
    while t.weekday() != 0:
        t += timedelta(days=1)

    rows = []
    while t <= date(2025, 12, 31):
        feat = build_features(weather_df, modis_df, t)

        # ndvi_z must be non-None (cannot form a reliable feature row)
        if feat.get("ndvi_z") is None:
            t += timedelta(weeks=1)
            continue

        target_z = _get_target(z_daily, t)
        if target_z is None:
            t += timedelta(weeks=1)
            continue

        row = {"t": t, "point_id": pid}
        for fn in fnames:
            row[fn] = feat.get(fn)
        row["target_z"] = target_z
        row["event"] = int(target_z <= EVENT_THRESHOLD)
        rows.append(row)

        t += timedelta(weeks=1)

    if not rows:
        df = pd.DataFrame(columns=["t", "point_id"] + fnames + ["target_z", "event"])
    else:
        df = pd.DataFrame(rows)

    df.to_parquet(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _mae(y_true, y_pred) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(np.array(y_true)[mask] - np.array(y_pred)[mask])))


def _coverage(y_true, p10, p90) -> float:
    mask = np.isfinite(y_true) & np.isfinite(p10) & np.isfinite(p90)
    if mask.sum() == 0:
        return float("nan")
    inside = (np.array(y_true)[mask] >= np.array(p10)[mask]) & (
        np.array(y_true)[mask] <= np.array(p90)[mask]
    )
    return float(inside.mean())


def _auroc(y_true, y_score) -> Optional[float]:
    y_t = np.array(y_true)
    y_s = np.array(y_score)
    mask = np.isfinite(y_s)
    y_t, y_s = y_t[mask], y_s[mask]
    if len(np.unique(y_t)) < 2:
        return None
    return float(roc_auc_score(y_t, y_s))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save_plots(test_df: pd.DataFrame, model_name: str = "Model A") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping plots")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_df["target_z"], test_df["p50"], alpha=0.4, s=8)
    lim = max(abs(test_df["target_z"].max()), abs(test_df["target_z"].min()), 3)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("Actual NDVI-z @ t+14d")
    ax.set_ylabel("Predicted p50")
    ax.set_title(f"{model_name} — Predicted vs Actual (test 2023–2025)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "predicted_vs_actual.png", dpi=100)
    plt.close(fig)

    # 2. MAE per season (model vs baselines)
    seasons = list(SEASONS.keys())
    model_mae, persist_mae, clim_mae = [], [], []
    for s in seasons:
        sub = test_df[test_df["season"] == s]
        if len(sub) == 0:
            model_mae.append(float("nan"))
            persist_mae.append(float("nan"))
            clim_mae.append(float("nan"))
            continue
        model_mae.append(_mae(sub["target_z"], sub["p50"]))
        persist_mae.append(_mae(sub["target_z"], sub["ndvi_z"]))
        clim_mae.append(_mae(sub["target_z"], np.zeros(len(sub))))
    x = np.arange(len(seasons))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, model_mae, width, label="Model A", color="steelblue")
    ax.bar(x, persist_mae, width, label="Persistence", color="orange")
    ax.bar(x + width, clim_mae, width, label="Climatology", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("MAE (NDVI-z)")
    ax.set_title(f"{model_name} — MAE per season (test 2023–2025)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mae_per_season.png", dpi=100)
    plt.close(fig)

    # 3. Coverage per season
    coverages = []
    for s in seasons:
        sub = test_df[test_df["season"] == s]
        if len(sub) == 0:
            coverages.append(float("nan"))
            continue
        coverages.append(_coverage(sub["target_z"], sub["p10"], sub["p90"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(seasons, coverages, color=["green" if c >= COVERAGE_MIN else "red" for c in coverages])
    ax.axhline(COVERAGE_MIN, color="black", ls="--", lw=1, label=f"min={COVERAGE_MIN}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("80%-interval coverage")
    ax.set_title(f"{model_name} — Coverage per season (test 2023–2025)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "coverage_per_season.png", dpi=100)
    plt.close(fig)

    print(f"  Plots saved to {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def main(debug: bool = False, force_rebuild: bool = False, allow_running_pull: bool = False) -> None:
    t0 = time.time()

    # ── Guard: no running pull process ──────────────────────────────────────
    _check_no_pull_running(allow_running=allow_running_pull)

    # ── Eligible points ──────────────────────────────────────────────────────
    points = _eligible_points()
    if debug:
        points = points[:1]

    dist_counts = Counter(p["district"] for p in points)
    n_pts = len(points)
    n_districts = len(dist_counts)

    print(f"\n{'='*60}")
    print(f"BreakWatch — Model A training")
    print(f"{'='*60}")
    print(f"Eligible points (both weather + MODIS): {n_pts}")
    print(f"Districts covered: {n_districts}")
    for dist, cnt in sorted(dist_counts.items()):
        print(f"  {dist}: {cnt} point(s)")

    if n_districts < 4:
        print(
            f"\n⚠️  WARNING: Only {n_districts} district(s) covered "
            f"(< 4 required for good spatial diversity).\n"
            f"   The model is valid but expect poor generalisation to other districts.\n"
            f"   Wait for the MODIS pull to complete more points before production use.\n"
        )

    # ── Load, build features, build target ─────────────────────────────────
    print("\nBuilding per-point feature tables (cached to data/train_raw/features/)...")
    all_frames: list[pd.DataFrame] = []

    for pt in points:
        pid = pt["id"]
        district = pt["district"]
        print(f"  {pid} ({district}) ...", end="", flush=True)

        w_df = _load_weather(pid)
        m_df = _load_modis_quality(pid)

        frame = _build_or_load_features(pid, w_df, m_df, force=force_rebuild)
        frame["district"] = district
        all_frames.append(frame)
        print(f" {len(frame)} samples")

    df = pd.concat(all_frames, ignore_index=True)
    df["t"] = pd.to_datetime(df["t"]).dt.date
    df["year"] = df["t"].apply(lambda d: d.year)
    df["month"] = df["t"].apply(lambda d: d.month)
    df["season"] = df["month"].map(season_of)

    fnames = feature_names()
    for col in fnames + ["target_z", "event"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"\nTotal samples: {len(df)} across {n_pts} points")
    print(f"Feature columns ({len(fnames)}): {fnames}")
    print(f"Any None in features: {df[fnames].isnull().any().any()}")

    # ── Train / calib / test splits ─────────────────────────────────────────
    train_mask = (df["year"] >= TRAIN_YEARS[0]) & (df["year"] <= TRAIN_YEARS[1])
    calib_mask = (df["year"] >= CALIB_YEARS[0]) & (df["year"] <= CALIB_YEARS[1])
    test_mask = (df["year"] >= TEST_YEARS[0]) & (df["year"] <= TEST_YEARS[1])

    train_df = df[train_mask].copy()
    calib_df = df[calib_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\nSplit sizes:")
    print(f"  train  {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]}: {len(train_df)} rows")
    print(f"  calib  {CALIB_YEARS[0]}–{CALIB_YEARS[1]}: {len(calib_df)} rows")
    print(f"  test   {TEST_YEARS[0]}–{TEST_YEARS[1]}: {len(test_df)} rows")

    # ── Assert no training row has year > TRAIN_YEARS[1] ────────────────────
    assert (
        train_df["year"].max() <= TRAIN_YEARS[1]
    ), f"Look-ahead leak: train has year {train_df['year'].max()} > {TRAIN_YEARS[1]}"
    # Assert temporal integrity: each sample's features only use data <= t
    # (enforced by build_features's look-ahead guard; we document this assertion)
    # The feature pipeline's as_of == t guarantees no future data is used.
    assert all(
        d <= date(TRAIN_YEARS[1], 12, 31) for d in train_df["t"]
    ), "Train split contains rows past the training window"

    # ── Train LightGBM ───────────────────────────────────────────────────────
    print(f"\nTraining LightGBM (n_estimators={LGB_PARAMS['n_estimators']}, "
          f"seed={LGB_PARAMS['seed']}) ...")
    X_train = train_df[fnames].values.astype(float)
    y_train = train_df["target_z"].values.astype(float)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    print("  Training done.")

    # ── Conformal calibration ────────────────────────────────────────────────
    X_calib = calib_df[fnames].values.astype(float)
    y_calib = calib_df["target_z"].values.astype(float)
    p50_calib = model.predict(X_calib)
    abs_residuals = np.abs(y_calib - p50_calib)
    # Finite-sample 80% quantile of |calib residuals|
    q = float(np.quantile(abs_residuals[np.isfinite(abs_residuals)], 0.80))
    print(f"  Conformal q (80th pct |calib residual|) = {q:.4f}")

    # ── Evaluate on test ─────────────────────────────────────────────────────
    X_test = test_df[fnames].values.astype(float)
    y_test = test_df["target_z"].values.astype(float)
    p50_test = model.predict(X_test)
    p10_test = p50_test - q
    p90_test = p50_test + q
    # prob_event: P(target <= -1) under Normal approx with σ ≈ q/1.2816
    # Assumption: residuals are approx Normal; q = 1.2816 * σ_resid
    sigma_approx = q / 1.2816
    prob_event_test = norm.cdf((-1.0 - p50_test) / sigma_approx)

    test_df = test_df.copy()
    test_df["p50"] = p50_test
    test_df["p10"] = p10_test
    test_df["p90"] = p90_test
    test_df["prob_event_pred"] = prob_event_test

    # ── Per-season metrics ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Per-season evaluation (test {TEST_YEARS[0]}–{TEST_YEARS[1]})")
    print(f"{'='*60}")
    header = (
        f"{'Season':<10} {'N':>5} {'MAE_model':>10} {'MAE_persist':>12} "
        f"{'MAE_clim':>10} {'Skill':>7} {'Cover':>7} {'AUROC':>7} {'Pass?':>6}"
    )
    print(header)
    print("-" * len(header))

    season_metrics: dict[str, dict] = {}

    for season in SEASONS:
        sub = test_df[test_df["season"] == season]
        n = len(sub)

        if n == 0:
            print(f"{season:<10} {'0':>5}  (no samples)")
            season_metrics[season] = {
                "n": 0, "mae_model": None, "mae_persistence": None,
                "mae_climatology": None, "skill": None,
                "coverage_80pct": None, "auroc": None, "validated": False,
            }
            continue

        yt = sub["target_z"].values
        yp = sub["p50"].values
        y_persist = sub["ndvi_z"].values.astype(float)  # persistence baseline
        y_clim = np.zeros(len(sub))                      # climatology baseline (predict 0)

        mae_m = _mae(yt, yp)
        mae_p = _mae(yt, y_persist)
        mae_c = _mae(yt, y_clim)
        skill = 1.0 - mae_m / mae_p if mae_p > 0 else float("nan")
        cov = _coverage(yt, sub["p10"].values, sub["p90"].values)
        auroc = _auroc(sub["event"].values, sub["prob_event_pred"].values)
        auroc_str = f"{auroc:.3f}" if auroc is not None else " None"

        validated = bool(
            skill >= SKILL_MIN
            and cov >= COVERAGE_MIN
            and n >= N_TEST_MIN
        )
        pass_str = "✓ PASS" if validated else "✗ FAIL"

        print(
            f"{season:<10} {n:>5} {mae_m:>10.4f} {mae_p:>12.4f} "
            f"{mae_c:>10.4f} {skill:>7.3f} {cov:>7.3f} {auroc_str:>7} {pass_str:>6}"
        )

        if not validated:
            reasons = []
            if skill < SKILL_MIN:
                reasons.append(f"skill={skill:.3f} < {SKILL_MIN}")
            if cov < COVERAGE_MIN:
                reasons.append(f"coverage={cov:.3f} < {COVERAGE_MIN}")
            if n < N_TEST_MIN:
                reasons.append(f"n={n} < {N_TEST_MIN}")
            print(f"  ↳ Gate FAILED: {', '.join(reasons)}")

        season_metrics[season] = {
            "n": n,
            "mae_model": round(mae_m, 6),
            "mae_persistence": round(mae_p, 6),
            "mae_climatology": round(mae_c, 6),
            "skill": round(float(skill), 6),
            "coverage_80pct": round(cov, 6),
            "auroc": round(auroc, 6) if auroc is not None else None,
            "validated": validated,
        }

    any_validated = any(v["validated"] for v in season_metrics.values())
    print(f"\nAny season validated: {any_validated}")

    # ── Save model ───────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    model_size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nModel saved: {MODEL_PATH}  ({model_size_mb:.2f} MB)")
    if model_size_mb > 20:
        print(f"  ⚠️  Model file exceeds 20 MB ({model_size_mb:.1f} MB).")

    # ── Write model card ─────────────────────────────────────────────────────
    # Keep existing keys; write common_version (from common.COMMON_VERSION,
    # never hardcoded); do NOT add a top-level "forecast" key; do NOT add
    # timestamps from the system clock.
    existing_card: dict = {}
    if CARD_PATH.exists():
        try:
            existing_card = json.loads(CARD_PATH.read_text())
        except Exception:
            existing_card = {}

    card = {
        "name": existing_card.get("name", "BreakWatch model card"),
        "version": existing_card.get("version", "0.1"),
        "status": "trained",
        "training_window": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
        "notes": existing_card.get("notes", ""),
        # ── written by train.py ────────────────────────────────────────────
        "common_version": COMMON_VERSION,      # read by agent/graph.py
        "feature_names": fnames,
        "horizon_days": HORIZON_DAYS,
        "conformal_q": round(q, 6),
        "split_years": {
            "train": list(TRAIN_YEARS),
            "calib": list(CALIB_YEARS),
            "test": list(TEST_YEARS),
        },
        "target_rule": (
            "Weekly Monday samples t in 2015-2025. "
            "Target = NDVI-z at t+14d from linear interpolation of z-scores "
            "of quality-OK MODIS composites between their END dates. "
            "Event = target_z <= -1.0. "
            "Samples dropped if ndvi_z feature is None or target cannot be formed."
        ),
        "districts_used": sorted(dist_counts.keys()),
        "n_points": n_pts,
        "n_rows_train": int(len(train_df)),
        "n_rows_calib": int(len(calib_df)),
        "n_rows_test": int(len(test_df)),
        "seasons": season_metrics,
        # NOTE: no top-level "forecast" key (agent/graph.py reads it for stress
        # interval; leaving absent so that path returns {} and withholds numbers
        # until a proper stress model is trained separately).
    }

    CARD_PATH.write_text(json.dumps(card, indent=2))
    print(f"Model card written: {CARD_PATH}")
    print("  Changed keys vs old card:")
    old_keys = set(existing_card.keys())
    new_keys = set(card.keys())
    for k in sorted(new_keys - old_keys):
        print(f"    + {k}: {card[k]!r}")
    for k in sorted(old_keys & new_keys):
        if existing_card.get(k) != card[k]:
            print(f"    ~ {k}: {existing_card.get(k)!r} → {card[k]!r}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _save_plots(test_df)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BreakWatch Model A (LightGBM NDVI-z forecast).")
    parser.add_argument("--debug", action="store_true", help="Use only first point for a quick sanity check.")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild per-point feature cache from scratch.")
    parser.add_argument("--allow-running-pull", action="store_true", help="Allow training on disk data even if pull_ is running.")
    args = parser.parse_args()
    main(debug=args.debug, force_rebuild=args.force_rebuild, allow_running_pull=args.allow_running_pull)

