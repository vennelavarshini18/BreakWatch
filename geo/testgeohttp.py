"""Unit tests for geo/http.py. Fake HTTP responses are fine here (tests/ only; the app never loads them)."""
import httpx
import pytest

import geo.http as H


class R:
    def __init__(self, code=200, body=None):
        self.status_code, self._b, self.text = code, body or {}, "x"

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