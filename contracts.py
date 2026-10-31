from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CONTRACT_VERSION = "0.2"

Quality = Literal["live", "cached", "fallback", "stale"]
Sensor = Literal["s2_ndmi", "s2_ndvi", "s1_vh", "s1_vv", "modis_ndvi", "soil_moist"]


class FieldInput(BaseModel):
    polygon: dict  # GeoJSON Polygon, EPSG:4326
    as_of: dt.date
    mode: Literal["live", "replay"] = "live"  # set ONLY by api/clock.py
    crop: Optional[Literal["cotton", "maize", "paddy", "redgram"]] = None
    sowing_date: Optional[dt.date] = None
    soil: Optional[Literal["red_sandy", "black_clay", "loam"]] = None
    irrigation_budget_mm: float = Field(0.0, ge=0)
    district: Optional[str] = None  # resolved in agent validate step


class Evidence(BaseModel):  # one ledger row; every shown number has one
    id: str  # "ev_017"
    key: str  # "rain_deficit_14d_mm"
    value: float | str | None
    unit: Optional[str] = None
    source: str  # "open-meteo:archive"
    fetched_at: dt.datetime
    obs_date: Optional[dt.date] = None
    quality: Quality
    note: Optional[str] = None


class SensorObs(BaseModel):
    """MODIS = composite END date (start+15)"""

    sensor: Sensor
    day: dt.date
    value: float
    z: Optional[float] = None  # None if < 3 baseline years
    valid_frac: float
    n_baseline: int
    evidence_id: str


class WeatherDay(BaseModel):
    day: dt.date
    precip_mm: Optional[float] = None
    et0_mm: Optional[float] = None
    tmax_c: Optional[float] = None
    sm_shallow: Optional[float] = None  # m3/m3, 0-7 cm
    sm_root: Optional[float] = None  # m3/m3, mean of 7-28 and 28-100 cm layers
    is_forecast: bool = False
    evidence_id: str


class Climatology(BaseModel):
    sm_root_percentile: Optional[float] = None  # 0-100 vs years < as_of.year, same day-of-year +-7d
    sm_shallow_percentile: Optional[float] = None
    rain_deficit_14d_mm: Optional[float] = None  # normal - observed; positive = deficit
    days_since_rain_gt_2p5mm: Optional[int] = None
    n_years: int = 0
    baseline_years: Optional[list[int]] = None
    evidence_ids: list[str] = []


class ObsBundle(BaseModel):  # P1 -> everyone
    field: FieldInput
    obs: list[SensorObs] = []
    weather_daily: list[WeatherDay] = []
    rain_ensemble: list[list[float]] = []  # [member][day] mm; LIVE mode only
    climatology: Climatology = Climatology()
    cropland_frac: Optional[float] = None
    events: list[str] = []  # "optical_cloudy->sar", "retry:open-meteo:1"
    evidence: list[Evidence] = []

    @model_validator(mode="after")
    def _no_lookahead(self):
        a = self.field.as_of
        for o in self.obs:
            if o.day > a:
                raise ValueError(f"look-ahead: obs {o.sensor} {o.day} > as_of {a}")
        for w in self.weather_daily:
            if w.is_forecast and self.field.mode == "replay":
                raise ValueError("forecast rows not allowed in replay mode")
            if not w.is_forecast and w.day > a:
                raise ValueError(f"look-ahead: weather {w.day} > as_of {a}")
        return self


class FusedState(BaseModel):  # P1
    """stress scale: positive = bad; stress = SENSOR_SIGN*z"""

    series: list[dict]  # day, mean, std, contrib{sensor: weight}
    latest_mean: float  # stress z-scale, higher = more stressed
    latest_std: float
    p_stress: float  # P(state > 1.0)
    onset_date: Optional[dt.date] = None
    sources_used: list[str] = []


class Forecast(BaseModel):  # P2
    """ndvi_z scale: negative = bad; quantiles p10<p50<p90 in ndvi_z"""

    horizon_days: int
    ndvi_z_p10: Optional[float] = None
    ndvi_z_p50: Optional[float] = None
    ndvi_z_p90: Optional[float] = None
    prob_event: Optional[float] = None  # P(NDVI z <= -1 at horizon)
    model_version: str
    validated_for_season: bool
    withheld_reason: Optional[str] = None
    evidence_ids: list[str] = []


class Trigger(BaseModel):  # P2
    id: str
    name: str
    fired: bool
    detail: str  # "17 consecutive days < 2.5 mm"
    chunk_id: str  # ICAR passage defining this trigger
    evidence_ids: list[str] = []


class IrrigationPlan(BaseModel):  # P2, LIVE mode only
    schedule: list[dict]  # day, mm
    stress_days_without: float
    stress_days_with: float
    n_scenarios: int
    objective: str
    evidence_ids: list[str] = []


class Passage(BaseModel):  # P2
    chunk_id: str
    doc: str
    district: str
    plan_year: Optional[int] = None
    condition: str
    crop: Optional[str] = None
    page: int
    text: str
    score: float


class Claim(BaseModel):  # P3 (LLM output unit)
    text: str
    evidence_ids: list[str] = []
    chunk_ids: list[str] = []


class Advisory(BaseModel):  # P3
    risk_level: Literal["low", "watch", "high", "insufficient_data"]
    claims: list[Claim]
    abstained: bool = False
    verifier_passes: int = 0
