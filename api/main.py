from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent.graph import run_graph
from api.clock import _now, resolve_mode
from contracts import FieldInput


if os.getenv("DEMO_MODE") == "1" and os.getenv("USE_FIXTURES") == "1":
    print("ERROR: DEMO_MODE=1 and USE_FIXTURES=1 are not allowed together.")
    raise SystemExit(1)


class RunRequest(BaseModel):
    polygon: dict
    as_of: date
    crop: str | None = None
    sowing_date: date | None = None
    soil: str | None = None
    irrigation_budget_mm: float = Field(0.0, ge=0)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if not isinstance(value, dict):
            raise ValueError("polygon must be a GeoJSON-like object")
        if value.get("type") != "Polygon":
            raise ValueError("polygon must be a GeoJSON Polygon")
        if not value.get("coordinates"):
            raise ValueError("polygon coordinates are required")
        return value

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value):
        try:
            resolve_mode(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return value


app = FastAPI(title="BreakWatch")
app.state.runs = {}


async def _write_event(run_id: str, node: str, status: str, msg: str) -> None:
    store = app.state.runs.setdefault(run_id, {"events": [], "result": None, "done": False})
    store["events"].append({"node": node, "status": status, "msg": msg})


async def _run_pipeline(run_id: str, payload: RunRequest) -> None:
    await _write_event(run_id, "validate", "start", "checking field polygon")
    try:
        field = FieldInput(
            polygon=payload.polygon,
            as_of=payload.as_of,
            crop=payload.crop,
            sowing_date=payload.sowing_date,
            soil=payload.soil,
            irrigation_budget_mm=payload.irrigation_budget_mm,
            mode=resolve_mode(payload.as_of),
        )
        await _write_event(run_id, "validate", "done", "field accepted")
        result = run_graph(field, events=[{"node": "validate", "status": "done", "msg": "field accepted"}])
        app.state.runs[run_id]["result"] = result
        app.state.runs[run_id]["done"] = True
        await _write_event(run_id, "respond", "done", "run complete")
    except Exception as exc:  # pragma: no cover - runtime safety for background task
        app.state.runs[run_id]["done"] = True
        await _write_event(run_id, "respond", "error", str(exc))


@app.post("/runs")
async def create_run(payload: RunRequest):
    run_id = str(uuid.uuid4())
    app.state.runs[run_id] = {"events": [], "result": None, "done": False}
    asyncio.create_task(_run_pipeline(run_id, payload))
    return {"run_id": run_id}


@app.get("/runs/{run_id}/events")
async def runs_events(run_id: str):
    async def stream():
        sent_count = 0
        while True:
            run = app.state.runs.get(run_id)
            if run is None:
                break
            events = run["events"]
            if events:
                for event in events[sent_count:]:
                    payload = {
                        "node": event["node"],
                        "status": event["status"],
                        "msg": event["msg"],
                        "ts": _now().isoformat(),
                    }
                    yield f"event: update\ndata: {json.dumps(payload)}\n\n"
                    sent_count += 1
            if run["done"]:
                break
            await asyncio.sleep(0.1)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/runs/{run_id}")
async def run_result(run_id: str):
    run = app.state.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run["result"] or {"run_id": run_id, "status": "processing"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


web_dir = Path(__file__).resolve().parents[1] / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
