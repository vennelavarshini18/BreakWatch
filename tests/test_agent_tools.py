from datetime import date
from threading import Event
from unittest.mock import patch

from agent.graph import run_graph
from agent.tools import safe_call
from contracts import FieldInput, Passage
from geo.http import NetworkUnavailable


def test_safe_call_returns_when_timeout_expires():
    started = Event()
    release = Event()

    def blocked_call():
        started.set()
        release.wait(timeout=1)

    result, events = safe_call("test.operation", blocked_call, timeout=0.1)
    release.set()

    assert started.is_set()
    assert result is None
    assert events == ["error:test.operation:timeout:after 0.1s"]


def test_run_graph_preserves_icar_citations_when_observations_unavailable():
    polygon = {
        "type": "Polygon",
        "coordinates": [[[78.5, 17.5], [78.6, 17.5], [78.6, 17.6], [78.5, 17.6], [78.5, 17.5]]],
    }
    field = FieldInput(polygon=polygon, as_of=date(2026, 11, 1), mode="live")
    fake_passages = [
        Passage(
            chunk_id="test_chunk_001",
            doc="AP_Rangareddy_2011.pdf",
            district="rangareddy",
            plan_year=2011,
            page=5,
            condition="mid season drought",
            text="test recommendation",
            score=0.9,
        )
    ]
    with patch("agent.graph.build_observations", side_effect=NetworkUnavailable("open-meteo unreachable after retries and no cached copy (HTTP 429)")):
        with patch("agent.graph.retrieve", return_value=fake_passages):
            res = run_graph(field)

    assert res["risk_level"] == "insufficient_data"
    assert any("error:geo.build_observations:NetworkUnavailable" in ev for ev in res["events"])
    claims = res["advisory"]["claims"]
    assert len(claims) == 1
    assert "test_chunk_001" in claims[0]["chunk_ids"]

