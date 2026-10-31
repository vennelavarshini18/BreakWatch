from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    return datetime.now(IST)


def today_utc() -> date:
    """Back-compat alias; returns the IST calendar date for the current moment."""
    return _now().date()


def resolve_mode(as_of: date) -> str:
    today = today_utc()
    if as_of > today:
        raise ValueError(f"as_of {as_of.isoformat()} is in the future; use a date on or before {today.isoformat()}")
    return "live" if as_of == today else "replay"
