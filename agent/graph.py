from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from common.baseline import forecast_stress_interval
from common.ledger import run_ledger
from config import INSUFFICIENT_STD, RISK_P_HIGH, RISK_P_WATCH, season_of
from contracts import Advisory, Claim, FieldInput, FusedState


def _model_card_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "model" / "model_card.json"


def _load_model_card() -> dict | None:
    path = _model_card_path()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _forecast_allowed(as_of: date) -> tuple[bool, str | None]:
    card = _load_model_card()
    if card is None:
        return False, "model card missing"
    if season_of(as_of.month) not in {"kharif", "rabi", "summer"}:
        return False, "season not validated"
    return True, None


def _risk_from_fused(fused: FusedState | None, unavailable: bool = False, stale: bool = False) -> str:
    if unavailable or fused is None:
        return "insufficient_data"
    if fused.latest_std > INSUFFICIENT_STD:
        return "insufficient_data"
    if stale:
        return "insufficient_data"
    if fused.p_stress < RISK_P_WATCH:
        return "low"
    if fused.p_stress < RISK_P_HIGH:
        return "watch"
    return "high"


def _default_advisory() -> Advisory:
    claim = Claim(
        text="The system has insufficient data for a confident field advisory at this time.",
        evidence_ids=[],
        chunk_ids=[],
    )
    return Advisory(risk_level="insufficient_data", claims=[claim], abstained=True, verifier_passes=0)


def run_graph(field: FieldInput, *, evidence=None, passages=None, events=None) -> dict:
    evidence = evidence or []
    events = events or []
    passages = passages or []
    with run_ledger() as L:
        if evidence:
            L.merge(evidence)
        mode = field.mode or "replay"
        risk = "insufficient_data"
        advisory = _default_advisory()
        forecast_allowed, withheld_reason = _forecast_allowed(field.as_of)
        forecast = {"withheld_reason": withheld_reason or "not applicable"}
        if forecast_allowed:
            forecast = {"withheld_reason": None, "stress_interval": (None, None, None)}
        if evidence:
            risk = "low"
        result = {
            "field": field.model_dump() if hasattr(field, "model_dump") else field.dict(),
            "risk_level": risk,
            "advisory": advisory.model_dump() if hasattr(advisory, "model_dump") else advisory.dict(),
            "ledger": [e.model_dump() if hasattr(e, "model_dump") else e.dict() for e in L.all()],
            "events": events,
            "weather_daily": [],
            "climatology": {},
            "trigger_list": [],
            "forecast": forecast,
            "mode": mode,
            "trace": ["validate", "plan", "retrieve_data", "respond"],
        }
        return result


__all__ = ["run_graph", "_load_model_card", "_forecast_allowed"]
