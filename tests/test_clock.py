from datetime import date, datetime

import pytest
from zoneinfo import ZoneInfo

import api.clock as clock


def test_future_date_rejected():
    monkey = pytest.MonkeyPatch()
    monkey.setattr(clock, "_now", lambda: datetime(2024, 3, 1, tzinfo=ZoneInfo("Asia/Kolkata")))
    try:
        with pytest.raises(ValueError):
            clock.resolve_mode(date(2024, 3, 2))
    finally:
        monkey.undo()


def test_mode_resolution_uses_today_from_patched_now():
    monkey = pytest.MonkeyPatch()
    monkey.setattr(clock, "_now", lambda: datetime(2024, 3, 1, tzinfo=ZoneInfo("Asia/Kolkata")))
    try:
        assert clock.today_utc() == date(2024, 3, 1)
        assert clock.resolve_mode(date(2024, 3, 1)) == "live"
        assert clock.resolve_mode(date(2024, 2, 29)) == "replay"
    finally:
        monkey.undo()
