from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def safe_call(name: str, fn: Callable, *args, timeout: float | None = 8.0, **kwargs) -> tuple[Any | None, list[str]]:
    """Call a function and convert failures into safe event strings without fabricating values."""
    events: list[str] = []
    try:
        if timeout is None:
            return fn(*args, **kwargs), events
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(fn, *args, **kwargs)
            result = future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        except Exception:
            executor.shutdown(wait=True)
            raise
        else:
            executor.shutdown(wait=True)
            return result, events
    except NotImplementedError:
        events.append(f"module_unavailable:{name}")
        return None, events
    except TimeoutError:
        events.append(f"error:{name}:timeout:after {timeout:g}s")
        return None, events
    except Exception as exc:  # pragma: no cover - runtime safety
        reason = exc.__class__.__name__
        events.append(f"error:{name}:{reason}:{exc}")
        return None, events


def wrap_p1(name: str, fn: Callable):
    def runner(*args, **kwargs):
        return safe_call(name, fn, *args, timeout=8.0, **kwargs)
    return runner


def wrap_p2(name: str, fn: Callable):
    def runner(*args, **kwargs):
        return safe_call(name, fn, *args, timeout=8.0, **kwargs)
    return runner
