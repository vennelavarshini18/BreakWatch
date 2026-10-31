from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from common.ledger import Ledger as SharedLedger, ledger_add, run_ledger as _shared_run_ledger
from contracts import Evidence


class Ledger(SharedLedger):
    """Wrapper around the shared ledger; the actual system-clock stamping lives in common.ledger."""

    def merge(self, items: Iterable[Evidence]) -> list[Evidence]:
        added: list[Evidence] = []
        for item in items:
            if not isinstance(item, Evidence):
                continue
            if self.get(item.id) is None:
                self._entries[item.id] = item
                self._by_key.setdefault(item.key, []).append(item)
                added.append(item)
        return added

    def lookup(self, evidence_id: str) -> Evidence | None:
        return self.get(evidence_id)


@contextmanager
def run_ledger(ledger: Ledger | None = None):
    """Context manager shared with the common ledger so each run uses one evidence context."""
    if ledger is None:
        with _shared_run_ledger() as L:
            yield L
        return
    with _shared_run_ledger(ledger) as L:
        yield L


__all__ = ["Ledger", "run_ledger", "ledger_add"]
