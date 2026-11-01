# STATUS_P2

## What works

### 1. Common Baseline & Feature Engineering (`common/`)
- **`common/baseline.py`**:
  - Climatology baseline statistical utilities: multi-year normals, rolling percentiles, z-score computation, and non-linear stress mappings (`to_stress`).
  - Look-ahead guard enforcing `as_of` temporal cutoff (no future data leakage).
  - Support for both list-based and pandas Series observation collections.
- **`common/features.py`**:
  - `aggregate_hourly_to_daily`: Hourly to daily aggregation with IST calendar date boundaries (midnight IST = 18:30 UTC previous day, minimum 18 valid hours/day).
  - `consecutive_dry_days`: Backwards scan computing sub-threshold ($< 2.5\text{ mm}$) run length and dates ending at `as_of`.
  - `build_features`: Complete feature vector construction (10 ordered features: rain deficit, days since rain, soil moisture percentiles, NDVI z-scores, ET0 14d sum, Tmax 7d mean, DOY sine/cosine harmonics).

### 2. Training Points Sampling & Weather Ingestion (`ml/train/`)
- **`data/train_points.json`**: 80 stratified cropland points sampled across Telangana using ESA WorldCover (`collection: esa-worldcover`, cropland class 40, fraction $\ge 0.6$) via Planetary Computer STAC (seed 42).
- **`data/boundaries/telangana_districts.json`**: Telangana district boundaries loaded and used for point-in-polygon district attribution.
- **`ml/train/sample_points.py`**: Point sampler with district stratification and nearest-fallback geometry lookup.
- **`ml/train/pull_weather.py`**: Open-Meteo Archive API ingest (2010–2026 daily `precip_mm`, `et0_mm`, `tmax_c`, and hourly soil moisture aggregated to daily `sm_shallow`, `sm_root`). Resumable manifest, backoff retry logic, and polite rate limiting.
- **`data/train_raw/weather/`**: 28 points ingested with 6,142 full daily rows each.

### 3. ICAR Contingency Plans & RAG Retrieval (`ml/rag.py`)
- **`data/corpus/icar/`**: Standardized ICAR contingency plan PDFs for 9 Telangana districts (`adilabad`, `karimnagar`, `khammam`, `mahabubnagar`, `medak`, `nalgonda`, `nizamabad`, `rangareddy`, `warangal`), all verified 2011 plans.
- **`ml/rag.py`**:
  - ChromaDB vector store (`data/model/chroma`) backed by `all-MiniLM-L6-v2` embedding model.
  - Chunk parser segmenting by condition blocks (early-season drought, mid-season drought, terminal drought, unseasonal rain, etc.) into 549 chunk passages.
  - BM25 keyword fallback (`rank_bm25`) for resilient retrieval.
  - `retrieve(query, district, crop=None, condition=None, k=5) -> list[Passage]` with district and condition pre-filtering.

### 4. MODIS 250m NDVI Cube Extraction & Quality Assurance
- **`ml/train/pull_modis_cube.py`**:
  - Ingests 16-day 250m NDVI composites (`collection: modis-13Q1-061`) from Microsoft Planetary Computer STAC.
  - Quality masking using pixel reliability band (`pixel_reliability == 0` for good quality) over a $3 \times 3$ pixel window.
  - Explicit composite start and end dates extracted directly from STAC item metadata (`start_datetime`, `end_datetime`).
  - Filtered to Terra (`MOD13Q1`) to preserve standard 16-day composite spacing ($\sim 23$ composites/year).
  - Resumable manifest tracking at `data/train_raw/modis_manifest.json`.
- **`ml/train/check_data.py`**:
  - Cross-source quality gate auditing weather row completeness, missing value thresholds, date ordering, and NDVI physical value bounds ($[-0.2, 1.0]$).
  - Outputs summary report to `data/data_quality_report.md`.

### 5. ICAR Trigger Evaluation & Test Fields (`ml/triggers.py`, `scripts/`)
- **`ml/triggers.py`**:
  - Implements `check_triggers(bundle: ObsBundle) -> list[Trigger]`.
  - Rule 1 (`mid_season_dry_spell`): Evaluates `> 14` consecutive dry days ($< 2.5\text{ mm}$) per ICAR plan specification (`> 2 consecutive weeks rainless`).
  - Rule 2 (`early_season_dry_spell`): Evaluates `\ge 15` consecutive dry days post-sowing within early season ($15\text{--}20$ days after `sowing_date`).
  - Chunk linkage: Ties triggered rules directly to the field's district ICAR plan chunk ID via RAG.
  - Robust edge handling: explicit checks for missing sowing date, insufficient weather history ($< 14$ days), missing calendar days (flagged as missing data, not rain), and districts lacking contingency plans.
  - Emits Evidence records for run length, start date, and end date via `common.ledger.ledger_add`.
- **`scripts/find_dry_spells.py`**:
  - Monsoon dry-spell scanner (Jun 15 – Sep 30 across 2024, 2025, 2026) generating `data/dry_spell_scan.csv`.
- **`data/test_fields.json`**:
  - 5 ground-truthed test fields (3 dry-spell fields for 2024, 2025, 2026; 1 control field in Mahabubnagar with no run $\ge 10$ days; 1 cloudy placeholder field).
  - GeoJSON polygons ($\sim 5\text{ ha}$) with weekly `as_of_dates` sequences for evaluation.

---

## How to run

### Environment setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

### Run test suite
```bash
PYTHON=.venv/bin/python make test
# or
.venv/bin/python -m pytest -q
```

### Rebuild and test RAG retrieval
```bash
# Rebuild Chroma index from corpus PDFs
.venv/bin/python -m ml.rag --build

# Query RAG
.venv/bin/python -m ml.rag "dry spell after sowing" --district Mahabubnagar
```

### Run Monsoon Dry-Spell Scan
```bash
.venv/bin/python scripts/find_dry_spells.py
```

### Run Data Quality Gate Check
```bash
.venv/bin/python -m ml.train.check_data --limit 5
```

### Resume MODIS NDVI Pull
```bash
nohup .venv/bin/python -m ml.train.pull_modis_cube > data/train_raw/pull_modis.log 2>&1 &
tail -f data/train_raw/pull_modis.log
```

---

## In Progress / Background Jobs

- **MODIS Full Ingest**: `ml.train.pull_modis_cube` is actively running in the background across on-disk weather points.
- **Weather Points Backfill**: 28 points ingested; remaining points awaiting Open-Meteo daily quota reset.

---

## Next Steps

- Train ML baseline models on ingested weather and MODIS feature cubes.
- Connect `ml.triggers` and RAG passage retrieval with the P3 advisory generation pipeline.
