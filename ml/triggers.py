"""
ml/triggers.py
==============
Evaluates ICAR contingency plan triggers from an ObsBundle.

Encoded Rules:
1. mid_season_dry_spell:
   - ICAR Plan Wording: "> 2 consecutive weeks rainless (>2.5 mm) period"
   - Threshold Comparison: Uses ">" against 14 days (run_length > 14, i.e., run_length >= 15).
2. early_season_dry_spell:
   - ICAR Plan Wording: "15-20 days dry spell after sowing leading to poor germination/crop stand"
   - Threshold Comparison: Uses ">=" against 15 days (run_length >= 15) post-sowing.
   - Sowing date requirement: bundle.field.sowing_date must be provided; if missing, returns fired=False
     with detail "sowing date needed".

Both rules are tied to the chunk_id of the ICAR contingency plan for the field's district.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd

from common.features import consecutive_dry_days
from common.ledger import ledger_add
from contracts import ObsBundle, Trigger


@lru_cache(maxsize=32)
def _get_district_chunks(dist_key: str) -> dict[str, str]:
    """Retrieve chunk_ids from the ICAR RAG index for a canonical district."""
    import ml.rag as rag

    mid = rag.retrieve(
        "Mid season drought long dry spell consecutive 2 weeks rainless",
        district=dist_key,
        condition="mid season drought",
        k=1,
    )
    early = rag.retrieve(
        "Early season drought dry spell 15 20 days after sowing",
        district=dist_key,
        condition="early season drought",
        k=1,
    )
    return {
        "mid_season": mid[0].chunk_id if mid else "",
        "early_season": early[0].chunk_id if early else "",
    }


def check_triggers(bundle: ObsBundle) -> list[Trigger]:
    """Check dry-spell triggers against the field's weather history and ICAR plan.

    Parameters
    ----------
    bundle:
        The observation bundle containing field parameters, daily weather, and evidence ledger.

    Returns
    -------
    list[Trigger]
        List of evaluated triggers (mid_season_dry_spell and early_season_dry_spell).
    """
    import ml.rag as rag

    field = bundle.field
    as_of = field.as_of
    dist_raw = field.district or ""
    dist_key = dist_raw.strip().lower()

    # Verify district validity against available ICAR plans
    valid_districts = {d["district"].lower() for d in rag.available_districts()}
    district_has_plan = bool(dist_key and dist_key in valid_districts)

    chunks = _get_district_chunks(dist_key) if district_has_plan else {"mid_season": "", "early_season": ""}
    mid_chunk_id = chunks["mid_season"]
    early_chunk_id = chunks["early_season"]

    # 1. Filter weather: only is_forecast=False and day <= as_of
    filtered_weather = [
        w for w in bundle.weather_daily if (not w.is_forecast) and (w.day <= as_of)
    ]

    # Convert to pandas Series indexed by date
    if filtered_weather:
        day_map = {w.day: (float(w.precip_mm) if w.precip_mm is not None else float("nan")) for w in filtered_weather}
        daily_precip = pd.Series(day_map)
    else:
        day_map = {}
        daily_precip = pd.Series(dtype=float)

    # 2. Check for missing weather history (< 14 days total)
    history_len = len(day_map)
    has_sufficient_history = history_len >= 14

    # 3. Check for missing calendar days in the 14-day window ending at as_of
    calendar_lookback = [as_of - timedelta(days=i) for i in range(14)]
    missing_calendar_days = [
        d for d in calendar_lookback if (d not in day_map) or math.isnan(day_map[d])
    ]
    has_missing_data = len(missing_calendar_days) > 0

    # 4. Compute consecutive dry days using common/features.py helper
    if not daily_precip.empty and as_of in day_map and not math.isnan(day_map[as_of]):
        run_length, run_start, run_end = consecutive_dry_days(daily_precip, as_of, threshold_mm=2.5)
    else:
        run_length, run_start, run_end = 0, None, None

    # 5. Create Evidence via common/ledger.py
    ev_len = ledger_add(
        "dry_spell_run_length",
        run_length,
        unit="days",
        source="open-meteo",
        obs_date=as_of,
        quality="live",
        note="consecutive days with precip < 2.5mm ending on as_of",
    )
    ev_start = ledger_add(
        "dry_spell_run_start",
        str(run_start) if run_start else "none",
        source="open-meteo",
        obs_date=run_start or as_of,
        quality="live",
        note="start date of consecutive dry run",
    )
    ev_end = ledger_add(
        "dry_spell_run_end",
        str(run_end) if run_end else "none",
        source="open-meteo",
        obs_date=run_end or as_of,
        quality="live",
        note="end date of consecutive dry run",
    )

    ev_ids = [ev_len.id, ev_start.id, ev_end.id]
    if hasattr(bundle, "evidence") and isinstance(bundle.evidence, list):
        bundle.evidence.extend([ev_len, ev_start, ev_end])

    triggers: list[Trigger] = []

    # ── RULE 1: mid_season_dry_spell ─────────────────────────────────────────
    # Plan wording: "> 2 consecutive weeks rainless (>2.5 mm) period"
    # Threshold comparison: Uses ">" against 14 days (run_length > 14, i.e., run_length >= 15).
    if not district_has_plan:
        mid_fired = False
        mid_detail = f"No ICAR plan exists for district '{dist_raw}'"
    elif not has_sufficient_history:
        mid_fired = False
        mid_detail = f"Fewer than 14 days of weather history available (found {history_len} days)"
    elif has_missing_data:
        missing_fmt = ", ".join(str(d) for d in sorted(missing_calendar_days))
        mid_fired = False
        mid_detail = f"Weather data is missing for {missing_fmt} (data is missing, not that it rained)"
    else:
        # Comparison uses '>' against 14 days
        mid_fired = run_length > 14
        if mid_fired:
            mid_detail = (
                f"Mid-season dry spell detected: {run_length} consecutive dry days "
                f"(< 2.5 mm) from {run_start} to {run_end} (threshold > 14 days)"
            )
        else:
            mid_detail = (
                f"No mid-season dry spell: current dry run is {run_length} days "
                f"(threshold > 14 days)"
            )

    triggers.append(
        Trigger(
            id="mid_season_dry_spell",
            name="Mid-Season Dry Spell (> 2 weeks rainless)",
            fired=mid_fired,
            detail=mid_detail,
            chunk_id=mid_chunk_id if district_has_plan else "",
            evidence_ids=ev_ids,
        )
    )

    # ── RULE 2: early_season_dry_spell ───────────────────────────────────────
    # Plan wording: "15-20 days dry spell after sowing leading to poor germination/crop stand"
    # Threshold comparison: Uses ">=" against 15 days (run_length >= 15) post-sowing.
    if not district_has_plan:
        early_fired = False
        early_detail = f"No ICAR plan exists for district '{dist_raw}'"
    elif field.sowing_date is None:
        early_fired = False
        early_detail = "sowing date needed"
    elif as_of < field.sowing_date:
        early_fired = False
        early_detail = f"as_of ({as_of}) is before sowing date ({field.sowing_date})"
    elif not has_sufficient_history:
        early_fired = False
        early_detail = f"Fewer than 14 days of weather history available (found {history_len} days)"
    elif has_missing_data:
        missing_fmt = ", ".join(str(d) for d in sorted(missing_calendar_days))
        early_fired = False
        early_detail = f"Weather data is missing for {missing_fmt} (data is missing, not that it rained)"
    else:
        # Calculate dry days occurring after sowing
        sowing_d = field.sowing_date
        days_since_sowing = (as_of - sowing_d).days + 1
        dry_days_after_sowing = min(run_length, days_since_sowing)

        # Comparison uses '>=' against 15 days
        early_fired = dry_days_after_sowing >= 15
        if early_fired:
            early_detail = (
                f"Early-season dry spell detected: {dry_days_after_sowing} consecutive dry days "
                f"after sowing ({sowing_d}) ending {as_of} (threshold >= 15 days)"
            )
        else:
            early_detail = (
                f"No early-season dry spell: {dry_days_after_sowing} dry days after sowing "
                f"(threshold >= 15 days)"
            )

    triggers.append(
        Trigger(
            id="early_season_dry_spell",
            name="Early-Season Dry Spell (15-20 days dry spell after sowing)",
            fired=early_fired,
            detail=early_detail,
            chunk_id=early_chunk_id if district_has_plan else "",
            evidence_ids=ev_ids,
        )
    )

    return triggers

