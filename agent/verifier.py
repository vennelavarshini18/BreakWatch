from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from contracts import Claim


ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_iso_dates(text: str) -> str:
    return ISO_DATE_RE.sub(" ", text)


def _number_precision(raw: str) -> int:
    try:
        text = raw.strip()
        if "." in text:
            return len(text.split(".", 1)[1])
    except Exception:
        return 0
    return 0


def _matches_value(number: float, candidate: float | str | None, raw_text: str) -> bool:
    if candidate is None:
        return False
    try:
        val = float(candidate)
    except (TypeError, ValueError):
        return False
    precision = _number_precision(raw_text)
    if precision == 0:
        tolerance = 0.0
    else:
        tolerance = 10 ** (-precision)
    return abs(number - val) <= tolerance


def _extract_numbers(text: str) -> list[float]:
    cleaned = _strip_iso_dates(text)
    values: list[float] = []
    for token in NUMBER_RE.findall(cleaned):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def verify_claim(claim: Claim, evidence_lookup: dict[str, object], passage_lookup: dict[str, str]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not claim.chunk_ids:
        issues.append("claim missing chunk_id")
    for chunk_id in claim.chunk_ids:
        if chunk_id not in passage_lookup:
            issues.append(f"chunk_id not retrieved: {chunk_id}")
    if not claim.evidence_ids:
        issues.append("claim missing evidence_ids")
    for evidence_id in claim.evidence_ids:
        if evidence_id not in evidence_lookup:
            issues.append(f"evidence not found: {evidence_id}")
    numbers = _extract_numbers(claim.text)
    for num in numbers:
        matched = False
        for ev_id in claim.evidence_ids:
            evidence = evidence_lookup.get(ev_id)
            if evidence is None:
                continue
            raw = getattr(evidence, "value", None)
            if raw is None:
                continue
            candidates: list[str]
            if isinstance(raw, str):
                candidates = [raw]
            elif isinstance(raw, (int, float)):
                candidates = [str(raw), format(float(raw), "g")]
            else:
                candidates = [str(raw), format(float(raw), "g")]
            for candidate in candidates:
                if _matches_value(num, candidate, str(num)):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            for chunk_id in claim.chunk_ids:
                passage_text = passage_lookup.get(chunk_id)
                if passage_text:
                    normalized = _strip_iso_dates(passage_text)
                    if str(num) in normalized or format(float(num), "g") in normalized:
                        matched = True
                        break
        if not matched:
            issues.append(f"number not traceable: {num}")
    return (not issues, issues)


def verify_advice(claims: list[Claim], evidence_lookup: dict[str, object], passage_lookup: dict[str, str]) -> tuple[bool, list[str]]:
    all_issues: list[str] = []
    for claim in claims:
        ok, issues = verify_claim(claim, evidence_lookup, passage_lookup)
        if not ok:
            all_issues.extend(issues)
    return (not all_issues, all_issues)
