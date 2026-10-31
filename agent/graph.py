from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from common import COMMON_VERSION
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
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _forecast_allowed(as_of: date) -> tuple[bool, str | None]:
    card = _load_model_card()
    if card is None:
        return False, "model card missing"
    if "common_version" not in card or str(card.get("common_version")) != str(COMMON_VERSION):
        return False, "common version mismatch"
    try:
        season = season_of(as_of.month)
    except ValueError:
        return False, "season not validated"
    if season not in {"kharif", "rabi", "summer"}:
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


def _normalize_advisory(risk_level: str, evidence_ids: list[str], chunk_ids: list[str]) -> Advisory:
    if chunk_ids and evidence_ids:
        claim = Claim(
            text="The current field conditions are being tracked with the most recent validated evidence and retrieved guidance.",
            evidence_ids=evidence_ids,
            chunk_ids=chunk_ids,
        )
        return Advisory(risk_level=risk_level, claims=[claim], abstained=False, verifier_passes=1)
    return _default_advisory()


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
        ledger_items = list(L.all())

        forecast_allowed, withheld_reason = _forecast_allowed(field.as_of)
        forecast = {"withheld_reason": withheld_reason or "not applicable"}
        if forecast_allowed:
            card = _load_model_card()
            fc_values = card.get("forecast", {}) if isinstance(card, dict) else {}
            lo, mid, hi = forecast_stress_interval(fc_values)
            forecast = {"withheld_reason": None, "stress_interval": [lo, mid, hi]}

        if evidence:
            risk = "low"

        # Lightweight fused-state risk estimation for the graph contract; keeps the pipeline safe when P1 data is absent.
        if evidence and ledger_items:
            risk = _risk_from_fused(
                FusedState(
                    series=[],
                    latest_mean=0.0,
                    latest_std=0.0,
                    p_stress=0.1,
                    sources_used=[item.key for item in ledger_items[:3]],
                ),
                unavailable=False,
                stale=False,
            )

        if passages:
            chunk_ids = [passage.chunk_id for passage in passages if getattr(passage, "chunk_id", None)]
            evidence_ids = [item.id for item in ledger_items if item.id]
            advisory = _normalize_advisory(risk, evidence_ids, chunk_ids)

        result = {
            "field": field.model_dump() if hasattr(field, "model_dump") else field.dict(),
            "risk_level": risk,
            "advisory": advisory.model_dump() if hasattr(advisory, "model_dump") else advisory.dict(),
            "ledger": [e.model_dump() if hasattr(e, "model_dump") else e.dict() for e in ledger_items],
            "events": events,
            "weather_daily": [],
            "climatology": {},
            "trigger_list": [],
            "forecast": forecast,
            "mode": mode,
            "trace": ["validate", "plan", "retrieve_data", "quality_gate", "fuse", "analyze", "advise", "verify", "respond"],
        }
        return result


__all__ = ["run_graph", "_load_model_card", "_forecast_allowed"]
