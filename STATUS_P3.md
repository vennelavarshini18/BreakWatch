# STATUS_P3

## Current state
- The repo is in a validated mergeable state for the P3 handoff: contracts, config, common utilities, agent scaffolding, API route flow, and the UI shell are all present and consistent.
- The merged branch includes the latest project history and the repo remains safe when upstream services are unavailable: modules surface explicit unavailable/error states instead of fabricating numbers.
- The final technical issue that surfaced in validation was the district/corpus fallback in the trigger retriever; it was resolved with a safe canonical fallback that keeps the app deterministic while preserving the real-corpus path when PDFs are present.

## What works
- Shared schema and config are live and stable (`contracts.py`, `config.py`, `api/clock.py`, `common/ledger.py`).
- Shared baseline + feature utilities are present and validated (`common/baseline.py`, `common/features.py`).
- Agent core is scaffolded and safe (`agent/tools.py`, `agent/llm.py`, `agent/verifier.py`, `agent/graph.py`).
- API run flow and SSE endpoints are implemented in `api/main.py`.
- UI shell is present in `web/index.html` and `web/app.js` and consumes result JSON without hardcoded values.
- The repo passes the validation gates: `pytest -q` is green, and the no-system-clock lint gate passes when run using the same logic as the Makefile (this Windows environment does not have GNU `make` installed).

## How to run
- python -m venv .venv
- pip install -r requirements.txt
- uvicorn api.main:app --reload
- open http://localhost:8000/health
- open http://localhost:8000/ for the UI

## Stub-safe behavior
- Unavailable modules are intentionally surfaced as unavailable/error states rather than fake values.
- The app rejects bad input with clear validation errors.
- Forecast and advisory logic are guarded when model metadata or a real source dataset is absent.

## Blockers / remaining focus
- No active blocker remains for the P3 work itself in the current state.
- Final product sign-off is still a human validation step against the real field/weather/ML data sources when they are available in the runtime environment.
