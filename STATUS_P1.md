# STATUS_P1

Owner: P1 (geo-data). Branch: p1-geo. Scope: geo/ and its tests.

## What works

- **geo/http.py**: shared HTTP helper. 8 s timeout, 2 retries (1 s, then 3 s), an event string per retry or fallback.
  Write-through cache in data/cache/ (written only by this module). Always tries live first. If live fails after retries
  it serves the last cached copy with quality="cached", otherwise raises NetworkUnavailable. `fetched_at` is a
  wall-clock stamp from time.time(); it never decides which day is analysed.
- **geo/weather.py**: Open-Meteo weather.
  - Replay: archive endpoint only, days <= as_of, no forecast rows, no ensemble.
  - Live: archive for the settled past, forecast endpoint for the recent gap plus 16 forecast rows, rain ensemble (14 days).
  - Climatology from years strictly before as_of.year: soil-moisture percentiles, 14-day rain deficit, days since rain > 2.5 mm.
  - One Evidence row per group of daily rows and one per derived climatology number.
- **geo/quality.py**: observation quality gate.
  - Optical observation with valid_frac < 0.4 is dropped.
  - No fresh optical left and fresh SAR available -> event `optical_cloudy->sar`.
  - No SAR either -> event `optical_cloudy:no_sar_available`.
  - Newest observation older than 12 days (MODIS: 20 days) -> event `stale:<sensor>:<N>d`, and GateResult.stale is True when observations exist but none is fresh.
- **geo/__init__.py**: `build_observations(field) -> ObsBundle`. Weather, climatology, ensemble (live only), evidence, events, and the quality gate.
  Sensor observations are not built yet, so obs=[] and events carry `optical_unavailable:not_implemented` and `sar_unavailable:not_implemented`.
- **geo/s2.py**: Sentinel-2 spike (Planetary Computer STAC, polygon-median NDVI/NDMI, valid_frac per solar day).
- **geo/check_bundle.py**: end-to-end validator for a real ObsBundle (PASS/FAIL per rule).
- **Repo rule `make lint-time`**: passes. The word "today" and system-clock calls were removed from the geo files.
  The CLI keyword is now `--as-of now`, which resolves the date through api.clock.today_utc().
  The weather test moved to tests/test_geoweather.py.

## How to run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

make lint-time
python -m pytest -q

python -m geo.weather --lat 18.0 --lon 79.6 --as-of 2025-08-20 --mode replay
python -m geo.weather --lat 18.0 --lon 79.6 --as-of now --mode live
python -m geo.s2 --lat 18.0 --lon 79.6 --as-of 2025-08-20 --days 30
python -m geo.check_bundle --lat 18.0 --lon 79.6 --as-of 2025-08-20 --mode replay
python -m geo.check_bundle --lat 18.0 --lon 79.6 --as-of now --mode live
```

## S1 definition-of-done evidence

Paste the real output of each command below. Do not edit the numbers by hand.

### a) Replay, as_of 2025-08-20 (>= 60 daily rows, zero forecast rows, climatology fields non-null)

```
PASTE OUTPUT OF: python -m geo.weather --lat 18.0 --lon 79.6 --as-of 2025-08-20 --mode replay
```

### b) Live (16 forecast rows, ensemble member count printed)

```
PASTE OUTPUT OF: python -m geo.weather --lat 18.0 --lon 79.6 --as-of now --mode live
```

### c) Offline rerun of (a) with the network off (succeeds with quality="cached" or fails cleanly)

```
PASTE OUTPUT (Wi-Fi off) OF: python -m geo.weather --lat 18.0 --lon 79.6 --as-of 2025-08-20 --mode replay
```

### d) build_observations returns a valid ObsBundle in both modes

```
PASTE OUTPUT OF: python -m geo.check_bundle ... --mode replay   (all lines should say PASS)
PASTE OUTPUT OF: python -m geo.check_bundle ... --as-of now --mode live
```

### e) Sentinel-2 spike table and latency (or the documented blocker)

```
PASTE OUTPUT OF: python -m geo.s2 --lat 18.0 --lon 79.6 --as-of 2025-08-20 --days 30
```

### f) Repo checks

```
PASTE OUTPUT OF: make lint-time      (expected: lint-time ok)
PASTE OUTPUT OF: python -m pytest -q
```

## Stubbed (raise NotImplementedError, the agent shows "module unavailable")

- geo/s1.py (Sentinel-1 SAR loader)
- geo/modis.py (see the note below if a real version has been merged)
- geo/baseline.py (per-sensor baselines and z-scores)
- geo/fusion.py (fuse)

No stub returns fake numbers.

## Known issues / blockers

- **UI shows "Insufficient data" and an empty weather panel.** In the agent as last seen, `agent/graph.py` returns
  `weather_daily: []` and does not yet call `geo.build_observations`. That wiring belongs to P3. P1's side can be checked
  on its own with `python -m geo.check_bundle`.
- **Timeout risk in live mode.** Climatology makes about one Open-Meteo call per past year, and `agent/tools.py`
  allows P1 8 seconds. If check_bundle takes longer than 8 s, tell P3.
- **Open-Meteo rate limit.** Running P2's weather pull at the same time can cause HTTP 429 for P1. Run them one at a time.
- Optical and SAR observations are not produced yet, so the quality gate has no real data to act on until the S2/S1 loaders exist.

## Notes for other owners

- **P3:** the UI banner text should map these exact event strings: `optical_cloudy->sar` ("cloudy, switched to SAR"),
  `stale:<sensor>:<N>d` (stale warning), `optical_cloudy:no_sar_available`, `module_unavailable`, `cache_used:<host>`.
  `GateResult.stale` can feed `_risk_from_fused(stale=...)`.
- **P3 (Makefile):** no exemption is needed for geo/http.py. It uses time.time() for fetched_at, which lint-time does not flag.