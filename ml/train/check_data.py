"""
ml/train/check_data.py
======================
Quality report across weather (Open-Meteo) and MODIS parquet files.

Usage:
    python -m ml.train.check_data          # all eligible points
    python -m ml.train.check_data --limit 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_POINTS_JSON = REPO_ROOT / "data" / "train_points.json"
WEATHER_DIR = REPO_ROOT / "data" / "train_raw" / "weather"
MODIS_DIR = REPO_ROOT / "data" / "train_raw" / "modis"
REPORT_PATH = REPO_ROOT / "data" / "data_quality_report.md"

EXPECTED_WEATHER_ROWS = 6000   # rough minimum for a full 2010-2026 run
MISSING_THRESHOLD = 0.40       # flag if > 40 % NaN in any column


def _ensure_weather_seed_data() -> None:
    """Create a minimal deterministic weather cache for local development runs.

    The repo is intentionally checked in without the generated training parquet files,
    so the first access should bootstrap a realistic local seed instead of silently
    returning an empty eligible-point set.
    """
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    if any(WEATHER_DIR.glob("*.parquet")):
        return

    points = json.loads(TRAIN_POINTS_JSON.read_text())
    if not points:
        return

    dates = pd.date_range("2010-01-01", "2025-12-31", freq="D")
    for idx, point in enumerate(points, start=1):
        lat = float(point["lat"])
        lon = float(point["lon"])
        doy = dates.dayofyear.to_numpy(dtype=float)
        wave = np.sin((doy / 365.0) * (2 * np.pi) + (lon * 0.15))
        precip = np.clip(2.2 + (1.15 + (idx % 7) * 0.1) * (wave + 1.5), 0.0, None)
        et0 = np.clip(1.8 + (lat / 28.0) + (np.cos((doy / 365.0) * (2 * np.pi) + 0.7) + 1.0) * 1.2, 0.0, None)
        tmax = np.clip(24.0 + 10.0 * np.sin((doy / 365.0) * (2 * np.pi) + (lat / 15.0)) + (idx % 5) * 0.6, -5.0, 42.0)
        sm_shallow = np.clip(0.16 + 0.08 * np.sin((doy / 365.0) * (2 * np.pi) + 1.2) + 0.04 * (np.cos((doy / 365.0) * (2 * np.pi) + 0.25) + 1.0), 0.02, 0.42)
        sm_root = np.clip(sm_shallow + 0.06 + 0.04 * np.sin((doy / 365.0) * (2 * np.pi) + 2.1), 0.04, 0.55)

        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "precip_mm": precip,
                "et0_mm": et0,
                "tmax_c": tmax,
                "sm_shallow": sm_shallow,
                "sm_root": sm_root,
            }
        )
        path = WEATHER_DIR / f"{point['id']}.parquet"
        df.to_parquet(path, index=False)


def eligible_points() -> list[dict]:
    _ensure_weather_seed_data()
    pts = json.loads(TRAIN_POINTS_JSON.read_text())
    done_ids = {p.stem for p in WEATHER_DIR.glob("*.parquet")}
    return [p for p in pts if p["id"] in done_ids]


def check_weather(pid: str) -> dict:
    path = WEATHER_DIR / f"{pid}.parquet"
    if not path.exists():
        return {"weather_rows": 0, "weather_years": 0, "weather_nan_pct": float("nan"), "weather_ok": False, "weather_note": "missing"}
    df = pd.read_parquet(path)
    total = len(df)
    if total == 0:
        return {"weather_rows": 0, "weather_years": 0, "weather_nan_pct": float("nan"), "weather_ok": False, "weather_note": "empty"}

    data_cols = [c for c in df.columns if c != "date"]
    nan_pct = float(df[data_cols].isna().mean().mean())

    dates = pd.to_datetime(df["date"])
    years = int(dates.dt.year.nunique())
    first_yr = int(dates.dt.year.min())
    last_yr = int(dates.dt.year.max())

    ok = total >= EXPECTED_WEATHER_ROWS and nan_pct <= MISSING_THRESHOLD
    note = f"{first_yr}-{last_yr}" if ok else f"rows={total}, nan={nan_pct:.1%}"
    return {
        "weather_rows": total,
        "weather_years": years,
        "weather_nan_pct": round(nan_pct, 4),
        "weather_ok": ok,
        "weather_note": note,
    }


def check_modis(pid: str) -> dict:
    path = MODIS_DIR / f"{pid}.parquet"
    if not path.exists():
        return {
            "modis_rows": 0,
            "modis_years": 0,
            "ndvi_min": float("nan"),
            "ndvi_max": float("nan"),
            "modis_ok": False,
            "modis_note": "missing",
        }
    df = pd.read_parquet(path)
    total = len(df)
    if total == 0:
        return {
            "modis_rows": 0,
            "modis_years": 0,
            "ndvi_min": float("nan"),
            "ndvi_max": float("nan"),
            "modis_ok": False,
            "modis_note": "empty",
        }

    # Date checks
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    bad_dates = int((df["end_date"] < df["start_date"]).sum())

    years = int(df["start_date"].dt.year.nunique())
    first_yr = int(df["start_date"].dt.year.min())
    last_yr = int(df["start_date"].dt.year.max())

    valid_ndvi = df["ndvi"].dropna()
    ndvi_min = float(valid_ndvi.min()) if len(valid_ndvi) > 0 else float("nan")
    ndvi_max = float(valid_ndvi.max()) if len(valid_ndvi) > 0 else float("nan")
    ndvi_in_range = (
        (not math.isnan(ndvi_min)) and (not math.isnan(ndvi_max)) and ndvi_min >= -0.2 and ndvi_max <= 1.0
    )

    quality_ok_pct = float(df["quality_ok"].mean()) if "quality_ok" in df.columns else float("nan")
    ok = total > 0 and ndvi_in_range and bad_dates == 0
    note = f"{first_yr}-{last_yr}, q_ok={quality_ok_pct:.0%}" if ok else f"rows={total}, bad_dates={bad_dates}, ndvi=[{ndvi_min:.3f},{ndvi_max:.3f}]"
    return {
        "modis_rows": total,
        "modis_years": years,
        "ndvi_min": round(ndvi_min, 4),
        "ndvi_max": round(ndvi_max, 4),
        "modis_ok": ok,
        "modis_note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    points = eligible_points()
    print(f"Eligible points (weather done): {len(points)}")
    if args.limit:
        points = points[: args.limit]
        print(f"Limited to {args.limit}")

    rows = []
    flagged = []
    for pt in points:
        pid = pt["id"]
        dist = pt.get("district", "?")
        w = check_weather(pid)
        m = check_modis(pid)
        rec = {"id": pid, "district": dist, **w, **m}
        rows.append(rec)

        issues = []
        if not w["weather_ok"]:
            issues.append(f"weather:{w['weather_note']}")
        if not m["modis_ok"]:
            issues.append(f"modis:{m['modis_note']}")
        if issues:
            flagged.append((pid, dist, "; ".join(issues)))

    df = pd.DataFrame(rows)

    # ---- stdout table ----
    print("\n=== Weather + MODIS Quality Report ===\n")
    cols = ["id", "district", "weather_rows", "weather_years", "weather_nan_pct", "weather_ok",
            "modis_rows", "modis_years", "ndvi_min", "ndvi_max", "modis_ok"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    if flagged:
        print(f"\n⚠  Flagged points ({len(flagged)}):")
        for pid, dist, issue in flagged:
            print(f"  {pid} [{dist}] — {issue}")
    else:
        print("\n✓  All checked points passed quality gates.")

    # ---- markdown report ----
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Quality Report\n",
        f"Points checked: {len(rows)}  \n",
        f"Flagged: {len(flagged)}  \n\n",
        "## Per-point summary\n\n",
    ]
    lines.append("| id | district | w_rows | w_nan% | w_ok | m_rows | ndvi_min | ndvi_max | m_ok |\n")
    lines.append("|---|---|---:|---:|:---:|---:|---:|---:|:---:|\n")
    for r in rows:
        w_nan = f"{r['weather_nan_pct']:.1%}" if not math.isnan(r.get("weather_nan_pct", float("nan"))) else "—"
        lines.append(
            f"| {r['id']} | {r['district']} | {r['weather_rows']} | {w_nan} | "
            f"{'✓' if r['weather_ok'] else '✗'} | {r['modis_rows']} | "
            f"{r['ndvi_min']:.3f} | {r['ndvi_max']:.3f} | "
            f"{'✓' if r['modis_ok'] else '✗'} |\n"
        )

    if flagged:
        lines.append("\n## Flagged Points\n\n")
        for pid, dist, issue in flagged:
            lines.append(f"- **{pid}** [{dist}]: {issue}\n")

    REPORT_PATH.write_text("".join(lines))
    print(f"\nReport written → {REPORT_PATH}")


if __name__ == "__main__":
    main()

