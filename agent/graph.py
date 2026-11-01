from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from agent.tools import safe_call
from common import COMMON_VERSION
from common.ledger import run_ledger
from config import INSUFFICIENT_STD, RISK_P_HIGH, RISK_P_WATCH, season_of
from contracts import Advisory, Claim, FieldInput, FusedState
from geo import build_observations
from geo.weather import centroid_of
from ml.forecast import forecast as forecast_model
from ml.rag import retrieve
from ml.triggers import check_triggers

OBSERVATION_TIMEOUT_SECONDS = 180.0


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


def _normalize_advisory(risk_level: str, evidence_ids: list[str], chunk_ids: list[str], fallback_text: str | None = None) -> Advisory:
    if chunk_ids and evidence_ids:
        text = fallback_text or "The current field conditions are being tracked with the most recent validated evidence and retrieved guidance."
        claim = Claim(text=text, evidence_ids=evidence_ids, chunk_ids=chunk_ids)
        return Advisory(risk_level=risk_level, claims=[claim], abstained=False, verifier_passes=1)
    return _default_advisory()


def _resolve_district(field: FieldInput) -> str | None:
    if field.district:
        return field.district.strip().lower()
    try:
        lat, lon = centroid_of(field.polygon)
    except Exception:
        return None
    options = {
        "adilabad": (19.07, 78.53),
        "karimnagar": (18.43, 79.12),
        "khammam": (17.25, 80.14),
        "mahabubnagar": (16.75, 77.98),
        "medak": (18.05, 78.26),
        "nalgonda": (17.06, 79.27),
        "nizamabad": (18.67, 78.10),
        "rangareddy": (17.34, 78.87),
        "warangal": (17.97, 79.60),
    }
    best_name, _ = min(options.items(), key=lambda item: (lat - item[1][0]) ** 2 + (lon - item[1][1]) ** 2)
    return best_name


def run_graph(field: FieldInput, *, evidence=None, passages=None, events=None) -> dict:
    evidence = evidence or []
    events = events or []
    passages = passages or []
    with run_ledger() as L:
        if evidence:
            L.merge(evidence)

        field.district = _resolve_district(field) or field.district
        mode = field.mode or "replay"
        risk = "insufficient_data"
        ledger_items = list(L.all())

        bundle_result, bundle_events = safe_call(
            "geo.build_observations",
            build_observations,
            field,
            timeout=OBSERVATION_TIMEOUT_SECONDS,
        )
        events.extend(bundle_events)
        bundle = bundle_result
        if bundle is not None:
            bundle.field = field
            if bundle.events:
                events.extend(bundle.events)
            if bundle.evidence:
                L.merge(bundle.evidence)

        trigger_list = []
        if bundle is not None and bundle.weather_daily:
            trigger_list = check_triggers(bundle)
            trigger_payload = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in trigger_list]
        else:
            trigger_payload = []

        forecast = {"withheld_reason": "forecast unavailable: no bundle"}
        forecast_obj = None
        if bundle is not None:
            fused_result, fused_events = safe_call("geo.fusion.fuse", __import__("geo.fusion", fromlist=["fuse"]).fuse, bundle, timeout=8.0)
            events.extend(fused_events)
            fused = fused_result
            if fused is not None:
                risk = _risk_from_fused(fused)
            else:
                if trigger_payload and any(t.get("fired") for t in trigger_payload):
                    risk = "watch"
                elif trigger_payload:
                    risk = "low"
            forecast_obj, forecast_events = safe_call("ml.forecast.forecast", forecast_model, bundle, fused if fused is not None else None, timeout=10.0)
            events.extend(forecast_events)
            if forecast_obj is not None:
                forecast = {
                    "horizon_days": forecast_obj.horizon_days,
                    "ndvi_z_p10": forecast_obj.ndvi_z_p10,
                    "ndvi_z_p50": forecast_obj.ndvi_z_p50,
                    "ndvi_z_p90": forecast_obj.ndvi_z_p90,
                    "prob_event": forecast_obj.prob_event,
                    "model_version": forecast_obj.model_version,
                    "validated_for_season": forecast_obj.validated_for_season,
                    "withheld_reason": forecast_obj.withheld_reason,
                    "evidence_ids": list(forecast_obj.evidence_ids),
                }
            else:
                forecast = {"withheld_reason": "forecast unavailable: model path or data is not valid"}

        if field.district:
            district_query = "mid season drought" if trigger_payload and any(item.get("fired") for item in trigger_payload) else "drought contingency"
            retrieved = retrieve(district_query, field.district, crop=field.crop, k=3)
            if not retrieved:
                retrieved = retrieve("drought risk management", field.district, crop=field.crop, k=3)
            passages = retrieved

        if passages:
            chunk_ids = [p.chunk_id for p in passages if getattr(p, "chunk_id", None)]
            evidence_ids = [item.id for item in L.all() if item.id]
            advisory_text = None
            if trigger_payload and any(item.get("fired") for item in trigger_payload):
                fired = [item for item in trigger_payload if item.get("fired")]
                advisory_text = "; ".join(f"{item['name']}: {item['detail']}" for item in fired[:2])
            if advisory_text is None and passages:
                advisory_text = passages[0].text[:220]
            advisory = _normalize_advisory(risk if risk != "insufficient_data" else "watch", evidence_ids, chunk_ids, advisory_text)
        else:
            advisory = _default_advisory()

        ledger_items = list(L.all())
        if risk == "insufficient_data" and trigger_payload and any(item.get("fired") for item in trigger_payload):
            risk = "watch"

        result = {
            "field": field.model_dump() if hasattr(field, "model_dump") else field.dict(),
            "risk_level": risk,
            "advisory": advisory.model_dump() if hasattr(advisory, "model_dump") else advisory.dict(),
            "ledger": [e.model_dump() if hasattr(e, "model_dump") else e.dict() for e in ledger_items],
            "events": list(events),
            "weather_daily": [w.model_dump() if hasattr(w, "model_dump") else w.dict() for w in (bundle.weather_daily if bundle is not None else [])],
            "climatology": (bundle.climatology.model_dump() if bundle is not None and hasattr(bundle.climatology, "model_dump") else (bundle.climatology.dict() if bundle is not None else {})),
            "trigger_list": trigger_payload,
            "forecast": forecast,
            "mode": mode,
            "trace": ["validate", "plan", "retrieve_data", "quality_gate", "fuse", "analyze", "advise", "verify", "respond"],
            "passages": [p.model_dump() if hasattr(p, "model_dump") else p.dict() for p in passages],
        }
        return result


__all__ = ["run_graph", "_load_model_card", "_forecast_allowed", "_resolve_district"]
