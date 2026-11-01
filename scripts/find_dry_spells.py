"""
scripts/find_dry_spells.py
==========================
Scan on-disk weather Parquet files across 2024, 2025, and 2026
within the monsoon season window (Jun 15 – Sep 30) for dry spells
(< 2.5 mm rainfall).

Saves results to data/dry_spell_scan.csv.

Usage:
    python scripts/find_dry_spells.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEATHER_DIR = REPO_ROOT / "data" / "train_raw" / "weather"
TRAIN_POINTS_PATH = REPO_ROOT / "data" / "train_points.json"
OUTPUT_CSV_PATH = REPO_ROOT / "data" / "dry_spell_scan.csv"

# Monsoon window per year
START_MONTH_DAY = (6, 15)  # June 15
END_MONTH_DAY = (9, 30)    # September 30
RAIN_THRESHOLD_MM = 2.5
TARGET_YEARS = [2024, 2025, 2026]


def main() -> None:
    # 1. Load valid ICAR plan districts
    import ml.rag as rag
    valid_districts = {d["district"].lower(): d["district"] for d in rag.available_districts()}

    # 2. Load training points metadata
    points_list = json.loads(TRAIN_POINTS_PATH.read_text())
    points_by_id = {p["id"]: p for p in points_list}

    # 3. Locate available weather Parquet files
    parquet_files = sorted(WEATHER_DIR.glob("*.parquet"))
    print(f"Total on-disk weather Parquet files found: {len(parquet_files)}")

    results: list[dict] = []
    skipped_districts: set[str] = set()

    for pfile in parquet_files:
        pid = pfile.stem
        if pid not in points_by_id:
            continue
        pt = points_by_id[pid]
        dist_raw = pt["district"]
        dist_key = dist_raw.lower()

        if dist_key not in valid_districts:
            skipped_districts.add(dist_raw)
            continue

        canonical_district = valid_districts[dist_key].capitalize()

        df = pd.read_parquet(pfile)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")

        for yr in TARGET_YEARS:
            win_start = date(yr, *START_MONTH_DAY)
            win_end = date(yr, *END_MONTH_DAY)

            sub = df.loc[(df.index >= win_start) & (df.index <= win_end)]
            if sub.empty:
                continue

            season_rain = float(sub["precip_mm"].sum())

            # Find longest contiguous run where precip_mm < 2.5 mm
            precip = sub["precip_mm"]
            max_run = 0
            best_s = None
            best_e = None

            cur_run = 0
            cur_s = None

            for d, val in precip.items():
                if pd.isna(val) or val < RAIN_THRESHOLD_MM:
                    if cur_run == 0:
                        cur_s = d
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                        best_s = cur_s
                        best_e = d
                else:
                    cur_run = 0
                    cur_s = None

            results.append({
                "point": pid,
                "district": canonical_district,
                "lat": pt["lat"],
                "lon": pt["lon"],
                "year": yr,
                "longest_run_days": max_run,
                "run_start": str(best_s) if best_s else "",
                "run_end": str(best_e) if best_e else "",
                "season_rain_mm": round(season_rain, 1),
            })

    if skipped_districts:
        print(f"Skipped points in districts not in ICAR corpus: {skipped_districts}")

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(by=["longest_run_days", "season_rain_mm"], ascending=[False, True])

    # Save CSV
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved {len(out_df)} scan records to {OUTPUT_CSV_PATH}")

    # Print top 15 longest dry spells
    print("\n=== Top 15 Longest Dry Spells (Jun 15 - Sep 30) ===")
    print(out_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
