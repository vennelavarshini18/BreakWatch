"""Unit tests for geo/http.py. Fake HTTP responses are fine here (tests/ only; the app never loads them)."""
import httpx
import pytest

import geo.http as H


class R:
    def __init__(self, code=200, body=None, headers=None):
        self.status_code, self._b, self.text = code, body or {}, "x"
        self.headers = headers or {}

    def json(self):
        return self._b


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(H.time, "sleep", lambda s: None)


def test_live_then_cache_fallback(monkeypatch):
    monkeypatch.setattr(H, "_request", lambda u, p: R(200, {"a": 1}))
    body, q, _, ev = H.get_json("https://api.open-meteo.com/x", {"p": 1})
    assert (body, q, ev) == ({"a": 1}, "live", [])

    def boom(u, p):
        raise httpx.ConnectError("down")
    monkeypatch.setattr(H, "_request", boom)
    body, q, _, ev = H.get_json("https://api.open-meteo.com/x", {"p": 1})
    assert q == "cached" and body == {"a": 1}
    assert ev == ["retry:open-meteo:1", "retry:open-meteo:2", "cache_used:open-meteo"]


def test_no_cache_raises(monkeypatch):
    monkeypatch.setattr(H, "_request", lambda u, p: (_ for _ in ()).throw(httpx.ConnectError("down")))
    with pytest.raises(H.NetworkUnavailable):
        H.get_json("https://api.open-meteo.com/y", {})


def test_4xx_is_api_error(monkeypatch):
    monkeypatch.setattr(H, "_request", lambda u, p: R(400, {"reason": "bad"}))
    with pytest.raises(H.ApiError):
        H.get_json("https://api.open-meteo.com/z", {})


def test_ttl_serves_cache_without_network(monkeypatch):
    monkeypatch.setattr(H, "_request", lambda u, p: R(200, {"a": 2}))
    H.get_json("https://api.open-meteo.com/t", {}, ttl_s=0)
    monkeypatch.setattr(H, "_request", lambda u, p: (_ for _ in ()).throw(AssertionError("no network")))
    body, q, _, ev = H.get_json("https://api.open-meteo.com/t", {}, ttl_s=H.IMMUTABLE_TTL_S)
    assert q == "cached" and ev == ["cache_hit:open-meteo"]


def test_429_honors_retry_after_seconds(monkeypatch):
    responses = iter([R(429, headers={"Retry-After": "2"}), R(200, {"a": 3})])
    sleeps = []
    monkeypatch.setattr(H, "_request", lambda u, p: next(responses))
    monkeypatch.setattr(H.time, "sleep", sleeps.append)

    body, quality, _, events = H.get_json("https://api.open-meteo.com/retry", {})

    assert body == {"a": 3}
    assert quality == "live"
    assert sleeps == [2.0]
    assert events == ["retry:open-meteo:after_2s"]


def test_429_retry_after_http_date_is_respected(monkeypatch):
    from email.utils import formatdate

    now = 1_798_000_000.0
    retry_at = formatdate(now + 7, usegmt=True)
    responses = iter([R(429, headers={"Retry-After": retry_at}), R(200, {"a": 4})])
    sleeps = []
    monkeypatch.setattr(H.time, "time", lambda: now)
    monkeypatch.setattr(H, "_request", lambda u, p: next(responses))
    monkeypatch.setattr(H.time, "sleep", sleeps.append)

    body, quality, _, events = H.get_json("https://api.open-meteo.com/retry-date", {})

    assert body == {"a": 4}
    assert quality == "live"
    assert sleeps == [7.0]
    assert events == ["retry:open-meteo:after_7s"]


def test_long_retry_after_uses_only_fresh_fallback_cache(monkeypatch):
    url = "https://api.open-meteo.com/long-retry"
    params = {"p": 1}
    H._write_cache(H.cache_key(url, params), url, params, {"a": 5}, H.time.time() - 30)
    calls = []
    monkeypatch.setattr(
        H,
        "_request",
        lambda u, p: calls.append((u, p)) or R(429, headers={"Retry-After": "120"}),
    )
    sleeps = []
    monkeypatch.setattr(H.time, "sleep", sleeps.append)

    body, quality, _, events = H.get_json(
        url,
        params,
        ttl_s=10,
        stale_if_error_ttl_s=60,
    )

    assert body == {"a": 5}
    assert quality == "cached"
    assert len(calls) == 1
    assert sleeps == []
    assert events == ["retry_deferred:open-meteo:120s", "cache_used:open-meteo"]


def test_forecast_cache_fallback_expires(monkeypatch):
    url = "https://api.open-meteo.com/expired"
    params = {"p": 1}
    old_fetch = H.time.time() - 120
    H._write_cache(H.cache_key(url, params), url, params, {"a": 6}, old_fetch)
    monkeypatch.setattr(H, "_request", lambda u, p: R(429, headers={"Retry-After": "120"}))
    monkeypatch.setattr(H.time, "sleep", lambda _: pytest.fail("must not retry before Retry-After"))

    with pytest.raises(H.NetworkUnavailable):
        H.get_json(url, params, ttl_s=60, stale_if_error_ttl_s=60)