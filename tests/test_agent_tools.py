from threading import Event

from agent.tools import safe_call


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
