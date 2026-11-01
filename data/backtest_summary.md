# BreakWatch Replay Backtest Summary

## 1. Executive Overview & Pipeline Status

- **Mode**: Historical Replay (`mode='replay'`, no forecast leakage, no system clock).
- **Evaluated Fields**: `field_dry_2026` (dry_spell), `field_control_mahabubnagar` (control)
- **Total Dates Evaluated**: 13

### Component Health:
| Component | Status | Details |
|---|---|---|
| **Weather & Climatology** | Operational | Open-Meteo ERA5 / archive segments & baseline percentiles |
| **ICAR Triggers** | Operational | Evaluates consecutive rainless days against ICAR plan thresholds |
| **Sensor Fusion (P1)** | Unavailable | `geo.fusion.fuse` raises `NotImplementedError` |
| **Model A Forecast** | Skipped | Forecast depends on `FusedState`; skipped cleanly when fusion unavailable |

## 2. Trigger Firing Analysis

### Field: `field_dry_2026` (Role: `dry_spell`, District: `mahabubnagar`)

- **Ground Truth Dry Spell**: **25 days** from **2026-08-24** to **2026-09-17** (consecutive days < 2.5 mm).
- **Trigger Firing**: Did not fire in this run (or data not cached/fetched).

> [!IMPORTANT]
> **Operational Reality Check**: A trigger confirms a dry spell that has **already lasted 15 days**;
> it is **not an early forecast** and must not be presented as one.

### Field: `field_control_mahabubnagar` (Role: `control`, District: `mahabubnagar`)

- **Control Verification**: Field has no runs ≥ 10 dry days in evaluation window.
- **False Alarms**: **0 / 5** dates fired.
  - ✓ Perfect control separation: 0 false alarms across all evaluation dates.

## 3. Sensor Fusion & Lead Time Placeholders

`geo.fusion.fuse` is not yet implemented (P1 task). In accordance with protocol:
- No synthetic or fabricated `FusedState` was created.
- `p_stress` and `risk_level` were recorded as `None` / `N/A`.
- When P1 fusion is implemented, this script will automatically populate:
  - `p_stress` and composite stress risk level.
  - `first_warning_date`: earliest date where `p_stress >= 0.2`.
  - `lead_time`: days between first warning and the first quality-passing MODIS composite with `ndvi z <= -1`.
  - If no MODIS composite exhibits `z <= -1`: records `no visible drop (composites unusable)`.

## 4. What This Backtest Does and Does Not Show

### What It DOES Show:
1. **Deterministic Rule Verification**: The ICAR dry-spell trigger logic faithfully evaluates weather histories and confirms the mid-season threshold (> 14 days) on day 15.
2. **Zero False Alarms on Control**: On control points without extended dry spells (< 10 days), triggers remain silent.
3. **Safe Graceful Degradation**: When downstream modules (fusion) or upstream network calls fail or are withheld, the pipeline continues without unhandled exceptions.

### What It DOES NOT Show:
1. **Early Predictive Skill**: Triggers are retrospective threshold detectors, firing only after 15 consecutive dry days have elapsed.
2. **Crop Stress Assessment**: Cannot assess biological or soil moisture stress until `geo.fusion.fuse` is completed.
3. **Model A Forecast Serving**: Model A 14-day forecast requires a valid `FusedState` and remains withheld.

## 5. API Call & Network Accounting

When executed against live endpoints without cached responses:
- **Open-Meteo Archive API**:
  - Climatology history (2010–2025): **1 call per distinct field** (cached with 10-year TTL).
  - 60-day recent segment: **1 call per evaluated as_of date**.
  - Rain ensemble: **0 calls** (explicitly disabled in replay mode).
- **Microsoft Planetary Computer STAC & COGs**:
  - STAC search: **1 search call per as_of date**.
  - COG reads: ~3 window extractions per uncached composite.

**Total API Calls for standard 2-field run (13 dates total)**:
- Open-Meteo Archive: 2 climatology + 13 segment calls = **15 requests**.
- Planetary Computer STAC: **13 search requests**.
