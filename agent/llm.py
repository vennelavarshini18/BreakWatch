from __future__ import annotations

import json
import os
from typing import Any


def _build_client():
    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover - provider dependency optional for tests
        raise RuntimeError("Anthropic SDK is not installed") from exc
    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not configured")
    return Anthropic(api_key=api_key)


def complete_json(system: str, user: str, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a parsed JSON object from the configured LLM provider."""
    model = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    prompt = user
    if json_schema is not None:
        prompt = f"{user}\nReturn only valid JSON matching this schema:\n{json.dumps(json_schema, sort_keys=True)}"
    client = _build_client()
    text = None
    for _ in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=800,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = getattr(response, "content", [])
            if isinstance(raw, list):
                chunks = []
                for item in raw:
                    if hasattr(item, "text"):
                        chunks.append(item.text)
                    elif isinstance(item, dict):
                        chunks.append(str(item.get("text", "")))
                text = "".join(chunks)
            else:
                text = str(raw)
            data = json.loads(text)
            return data
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    raise ValueError("LLM returned invalid JSON after one retry")
