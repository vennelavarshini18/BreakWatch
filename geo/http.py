"""geo/http.py - shared HTTP helper (P1).

Rules (from the hackathon plan):
  * 8 s timeout, 2 retries (sleep 1 s, then 3 s), an event string per retry/fallback.
  * Write-through cache in data/cache/ - ONLY this module writes there, never by hand.
  * ALWAYS try live first, unless the caller passes ttl_s > 0 AND a cached copy is younger than ttl_s
    (used for immutable archive calls).
  * If live fails after retries -> serve the last cached copy (quality="cached"), else NetworkUnavailable.
  * No date decisions here: fetched_at is a wall-clock stamp of the fetch (time.time()), never used to pick which day to analyse.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

TIMEOUT_S = 8.0
BACKOFFS_S = (1.0, 3.0)                      # 2 retries
IMMUTABLE_TTL_S = 10 * 365 * 24 * 3600       # "forever" for archive years that cannot change
CACHE_DIR = Path(os.environ.get("BREAKWATCH_CACHE_DIR",
                                Path(__file__).resolve().parents[1] / "data" / "cache"))


class NetworkUnavailable(Exception):
    """Live fetch failed after retries and no cached copy exists."""


class ApiError(Exception):
    """The API rejected the request (HTTP 4xx other than 429). Retrying will not help."""


def host_label(url: str) -> str:
    host = urlparse(url).hostname or url
    return "open-meteo" if "open-meteo" in host else host


def _canon_params(params: dict | None) -> str:
    return json.dumps(sorted((str(k), str(v)) for k, v in (params or {}).items()))


def cache_key(url: str, params: dict | None) -> str:
    return hashlib.sha256((url + _canon_params(params)).encode()).hexdigest()


def _body_sha(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text())
        if rec.get("sha256") != _body_sha(rec["body"]):
            return None                       # corrupted -> ignore
        return rec
    except Exception:
        return None


def _write_cache(key: str, url: str, params: dict | None, body: Any, ts: float) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"url": url, "params": params, "params_hash": hashlib.sha256(_canon_params(params).encode()).hexdigest(),
               "fetched_ts": ts, "sha256": _body_sha(body), "body": body}
        tmp = _cache_path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, _cache_path(key))
    except Exception:
        pass                                  # a cache write failure must never break a live answer


def _request(url: str, params: dict | None) -> httpx.Response:   # separate function so tests can patch it
    return httpx.get(url, params=params, timeout=TIMEOUT_S)


def _to_dt(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)


def get_json(url: str, params: dict | None, ttl_s: int = 0):
    """-> (json, quality, fetched_at, events). quality is 'live' or 'cached'."""
    events: list[str] = []
    host = host_label(url)
    key = cache_key(url, params)
    cached = _read_cache(key)

    if cached and ttl_s > 0 and (time.time() - cached["fetched_ts"]) <= ttl_s:
        return cached["body"], "cached", _to_dt(cached["fetched_ts"]), [f"cache_hit:{host}"]

    last_err: Exception | None = None
    for attempt in range(len(BACKOFFS_S) + 1):
        try:
            r = _request(url, params)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {r.status_code}")
            if 400 <= r.status_code < 500:
                try:
                    reason = r.json().get("reason", r.text[:200])
                except Exception:
                    reason = r.text[:200]
                raise ApiError(f"{host} HTTP {r.status_code}: {reason}")
            body = r.json()
            ts = time.time()
            _write_cache(key, url, params, body, ts)
            return body, "live", _to_dt(ts), events
        except ApiError:
            raise
        except (httpx.HTTPError, ValueError, OSError) as e:
            last_err = e
            if attempt < len(BACKOFFS_S):
                events.append(f"retry:{host}:{attempt + 1}")
                time.sleep(BACKOFFS_S[attempt])

    if cached:
        events.append(f"cache_used:{host}")
        return cached["body"], "cached", _to_dt(cached["fetched_ts"]), events
    raise NetworkUnavailable(f"{host} unreachable after retries and no cached copy ({last_err})")