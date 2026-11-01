"""geo/check_bundle.py - end-to-end check of a real ObsBundle (P1).

Run:  python -m geo.check_bundle --lat 18.0 --lon 79.6 --as-of 2025-08-20 --mode replay
      python -m geo.check_bundle --lat 18.0 --lon 79.6 --as-of now --mode live
Calls the real APIs (or the cache) and prints PASS/FAIL per rule. Exit code 1 if any hard check fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

ALLOWED_QUALITY = {"live", "cached", "fallback", "stale"}


def check_bundle(bundle) -> list[tuple[str, bool, str]]:
    """-> [(check name, passed, detail)]. Pure function over an ObsBundle."""
    a = bundle.field.as_of
    mode = bundle.field.mode
    ev_ids = [e.id for e in bundle.evidence]
    ev_set = set(ev_ids)
    fc = [w for w in bundle.weather_daily if w.is_forecast]
    out: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    add("no observation after as_of", all(o.day <= a for o in bundle.obs))
    add("no observed weather after as_of", all(w.day <= a for w in bundle.weather_daily if not w.is_forecast))
    add("evidence ids are unique", len(ev_ids) == len(ev_set))
    add("evidence quality values are valid", all(e.quality in ALLOWED_QUALITY for e in bundle.evidence))
    missing = [w.day.isoformat() for w in bundle.weather_daily if w.evidence_id not in ev_set]
    add("every weather day has an evidence row", not missing, f"missing: {missing[:3]}" if missing else "")
    add("every climatology evidence id exists", all(i in ev_set for i in bundle.climatology.evidence_ids))
    add("every observation has an evidence row", all(o.evidence_id in ev_set for o in bundle.obs))
    add("weather has at least 50 days", len(bundle.weather_daily) >= 50, f"{len(bundle.weather_daily)} rows")
    if mode == "replay":
        add("replay: no forecast rows", not fc, f"{len(fc)} forecast rows")
        add("replay: no rain ensemble", not bundle.rain_ensemble)
    else:
        add("live: 16 forecast rows", len(fc) == 16, f"{len(fc)} forecast rows")
        add("live: rain ensemble present", len(bundle.rain_ensemble) > 0, f"{len(bundle.rain_ensemble)} members")
    c = bundle.climatology
    add("climatology has 3+ baseline years", c.n_years >= 3, f"n_years={c.n_years}")
    add("climatology soil-moisture percentile present", c.sm_root_percentile is not None)
    add("climatology rain deficit present", c.rain_deficit_14d_mm is not None)
    return out


def _main() -> None:
    from contracts import FieldInput
    from geo import build_observations
    from geo.s2 import square_polygon

    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--as-of", required=True, help="YYYY-MM-DD or 'now'")
    ap.add_argument("--mode", choices=["live", "replay"], required=True)
    a = ap.parse_args()
    if a.as_of == "now":
        from api.clock import today_utc
        as_of = today_utc()
    else:
        as_of = dt.date.fromisoformat(a.as_of)
    field = FieldInput(polygon=square_polygon(a.lat, a.lon, 300.0), as_of=as_of, mode=a.mode)
    bundle = build_observations(field)
    results = check_bundle(bundle)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print("events:", bundle.events)
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()