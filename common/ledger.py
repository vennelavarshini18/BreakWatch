from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterable

from contracts import Evidence


class Ledger:
    """Simple thread-safe evidence ledger for BreakWatch."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: dict[str, Evidence] = {}
        self._by_key: dict[str, list[Evidence]] = {}
        self._next_index = 1

    def _next_id(self) -> str:
        ledger_id = f"ev_{self._next_index:03d}"
        self._next_index += 1
        return ledger_id

    def add(
        self,
        key: str,
        value,
        *,
        unit: str | None = None,
        source: str,
        obs_date=None,
        quality: str = "live",
        note: str | None = None,
        fetched_at: datetime | None = None,
    ) -> Evidence:
        with self._lock:
            evidence = Evidence(
                id=self._next_id(),
                key=key,
                value=value,
                unit=unit,
                source=source,
                fetched_at=fetched_at or datetime.now(timezone.utc),
                obs_date=obs_date,
                quality=quality,
                note=note,
            )
            self._entries[evidence.id] = evidence
            self._by_key.setdefault(key, []).append(evidence)
            return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        with self._lock:
            return self._entries.get(evidence_id)

    def merge(self, items: Iterable[Evidence]) -> list[Evidence]:
        with self._lock:
            added: list[Evidence] = []
            for item in items:
                if not isinstance(item, Evidence):
                    continue
                if item.id in self._entries:
                    continue
                self._entries[item.id] = item
                self._by_key.setdefault(item.key, []).append(item)
                added.append(item)
            return added

    def lookup(self, evidence_id: str) -> Evidence | None:
        return self.get(evidence_id)

    def all(self) -> list[Evidence]:
        with self._lock:
            return list(self._entries.values())

    def by_key(self, key: str) -> list[Evidence]:
        with self._lock:
            return list(self._by_key.get(key, []))


current_ledger: ContextVar[Ledger | None] = ContextVar("current_ledger", default=None)
_default_ledger = Ledger()


@contextmanager
def run_ledger(ledger: Ledger | None = None):
    """Use a ledger for the current context. Call from the main thread or use contextvars.copy_context().run."""
    previous = current_ledger.get()
    active = ledger if ledger is not None else (_default_ledger if previous is None else previous)
    token = current_ledger.set(active)
    try:
        yield active
    finally:
        current_ledger.reset(token)
        if previous is not None:
            current_ledger.set(previous)


def ledger_add(
    key: str,
    value,
    *,
    unit: str | None = None,
    source: str,
    obs_date=None,
    quality: str = "live",
    note: str | None = None,
    fetched_at: datetime | None = None,
) -> Evidence:
    ledger = current_ledger.get() or _default_ledger
    return ledger.add(
        key,
        value,
        unit=unit,
        source=source,
        obs_date=obs_date,
        quality=quality,
        note=note,
        fetched_at=fetched_at,
    )
