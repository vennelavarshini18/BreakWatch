# BreakWatch Disk Trigger Replay Summary

This is a **trigger-only replay on archived weather** loaded from `data/train_raw/weather/<point_id>.parquet`.

- Weather source: `disk archive (Open-Meteo ERA5 download)`.
- No network API, observation builder, fusion, or forecast was run.
- Fusion status: `fusion not run in --from-disk mode`.
- Trigger thresholds were not tuned.
- A fired trigger confirms a dry spell **already 15 days long**; it is **not an early forecast**.
- No stress result or lead-time result exists in this replay.

- **Fields in test set**: 5
- **as_of dates evaluated**: 25

## Per-field results

| Field | as_of dates | Mid-season fires | Missing weather dates |
|---|---:|---:|---:|
| `field_dry_2026` | 8 | 3 | 0 |
| `field_dry_2025` | 6 | 0 | 0 |
| `field_dry_2024` | 6 | 0 | 0 |
| `field_control_mahabubnagar` | 5 | 0 | 0 |
| `field_cloudy_placeholder` | 0 | 0 | 0 |

## Interpretation limits

The trigger is retrospective: a firing confirms a dry spell that has already lasted 15 days and is not an early forecast. Fusion was not run, so no crop-stress probability or stress classification is available. No lead-time result exists because this run did not evaluate a forecast or sensor-fusion onset.
