from datetime import datetime, timezone

from common.ledger import Ledger, ledger_add, run_ledger
from contracts import Evidence


def test_ledger_ids_are_sequential():
    ledger = Ledger()
    first = ledger.add("rain", 2.5, source="test", quality="live")
    second = ledger.add("rain", 3.0, source="test", quality="live")
    assert first.id == "ev_001"
    assert second.id == "ev_002"
    assert ledger.get("ev_001").value == 2.5


def test_fetched_at_override_is_preserved():
    ledger = Ledger()
    fetched_at = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    evidence = ledger.add("temp", 18.0, source="test", fetched_at=fetched_at)
    assert evidence.fetched_at == fetched_at


def test_context_isolation_between_ledgers():
    first = Ledger()
    second = Ledger()

    with run_ledger(first):
        first_evidence = ledger_add("metric", 1.0, source="test")
        assert first_evidence.id == "ev_001"

    with run_ledger(second):
        second_evidence = ledger_add("metric", 2.0, source="test")
        assert second_evidence.id == "ev_001"
        assert second.get("ev_001").value == 2.0

    assert first.get("ev_001").value == 1.0


def test_merge_keeps_unique_evidence_and_preserves_existing_entries():
    ledger = Ledger()
    first = ledger.add("rain", 2.5, source="test")
    second = Evidence(
        id="ev_099",
        key="temp",
        value=18.0,
        source="test",
        fetched_at=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        quality="live",
    )

    added = ledger.merge([first, second])
    assert [item.id for item in added] == ["ev_099"]
    assert len(ledger.all()) == 2
    assert ledger.get(first.id).value == 2.5
    assert ledger.get(second.id).value == 18.0
