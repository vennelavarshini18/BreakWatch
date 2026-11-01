# STATUS_P3

## Current state
- `main` already contains the merged `p1-geo` and `p2-ml` history.
- The repo is in a validated, merged baseline state with the shared contracts, config, ledger, common feature logic, agent scaffolding, API run flow, and UI shell in place.
- The runtime remains safe when upstream modules are unavailable: it produces explicit unavailable/error states instead of fabricating data.

## What works
- Shared schema and config are live and stable (`contracts.py`, `config.py`, `api/clock.py`, `common/ledger.py`).
- Shared baseline + feature utilities are present (`common/baseline.py`, `common/features.py`).
- Agent core is scaffolded (`agent/tools.py`, `agent/llm.py`, `agent/verifier.py`, `agent/graph.py`).
- API run flow and SSE endpoints are implemented in `api/main.py`.
- UI shell is present in `web/index.html` and `web/app.js` and consumes result JSON without hardcoded values.
- The repo passes the required validation gates: `pytest -q` and `make lint-time`.

## How to run
- python -m venv .venv
- pip install -r requirements.txt
- uvicorn api.main:app --reload
- open http://localhost:8000/health
- open http://localhost:8000/ for the UI

## Stub-safe behavior
- Unavailable modules are intentionally surfaced as unavailable/error states rather than fake values.
- The app rejects bad input with clear validation errors.
- Forecast and advisory logic remain guarded when model metadata or real source data are absent.

## Blockers / remaining focus
- No active blocker from P1/P2 is present in the current branch state.
- Final integration polish is limited to end-to-end verification against real field/weather/ML data and final product sign-off.
