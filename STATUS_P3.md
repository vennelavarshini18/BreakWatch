# STATUS_P3

## What works
- P3 repo skeleton created.
- Shared contracts/config/clock/ledger bootstrapped.
- FastAPI health endpoint and startup guard are active.

## How to run
- python -m venv .venv
- pip install -r requirements.txt
- uvicorn api.main:app --reload
- open http://localhost:8000/health

## Stubbed
- Geo, ML, and agent modules are not yet implemented; they intentionally remain unavailable.

## Blockers
- P1/P2 feature modules are pending.
- Real data integrations and advisory generation are not yet wired in.
